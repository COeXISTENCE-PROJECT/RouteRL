import argparse
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import sys
import os


def get_episodes(episodes_folder: str) -> list[int]:
    """Get the episodes data

    Returns:
        sorted_episodes (list[int]): the sorted episodes data
    Raises:
        FileNotFoundError: If the episodes folder does not exist
    """

    eps = list()
    if os.path.exists(episodes_folder):
        for file in os.listdir(episodes_folder):
            episode = int(file.split("ep")[1].split(".csv")[0])
            eps.append(episode)
    else:
        raise FileNotFoundError(f"Episodes folder does not exist!")

    eps = [ep for ep in eps if ep % 5 == 0]  # faster

    return sorted(eps)


def flatten_by_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten a DataFrame by ID.
    """
    # return one row dataframe with all columns renamed to "agent_<id>_<original_column_name>" for each id

    flattened_df = pd.DataFrame()

    columns = []
    values = []

    for id in df["id"]:
        # get the row with the id
        row = df[df["id"] == id]

        # rename the columns
        for column in row.columns:
            if column != "id":
                columns.append(f"agent_{id}_{column}")
                values.append(row[column].values[0])

    # create a new row with the values
    new_row = {}
    for i in range(len(columns)):
        new_row[columns[i]] = values[i]

    flattened_df = pd.DataFrame([new_row])

    return flattened_df


def load_general_SUMO(file) -> pd.DataFrame:
    """
    Load general SUMO output file and return a DataFrame.
    """

    tree = ET.parse(file)  # replace with your actual file path
    root = tree.getroot()

    # Flatten the XML into a single dictionary
    flat_data = {}
    for child in root:
        for key, value in child.attrib.items():
            flat_data[f"{child.tag}_{key}"] = value

    # Convert to a single-row DataFrame
    df = pd.DataFrame([flat_data])

    # remove the columns that are not needed
    cols = [
        "teleports_total",
        "teleports_jam",
        "teleports_yield",
        "teleports_wrongLane",
        "vehicleTripStatistics_count",
        "vehicleTripStatistics_routeLength",
        "vehicleTripStatistics_speed",
        "vehicleTripStatistics_duration",
        "vehicleTripStatistics_waitingTime",
        "vehicleTripStatistics_timeLoss",
        "vehicleTripStatistics_departDelay",
        "vehicleTripStatistics_totalTravelTime",
        "vehicleTripStatistics_totalDepartDelay",
    ]

    df = df[cols]

    try:
        df = df.apply(pd.to_numeric)
    except ValueError:
        pass

    # print(df.shape)
    # print(df)

    return df


def load_detailed_SUMO(file) -> pd.DataFrame:
    """
    Load detailed SUMO output file and return a DataFrame.
    """
    tree = ET.parse(file)  # replace with your file
    root = tree.getroot()

    # Extract all tripinfo elements and their attributes
    data = [trip.attrib for trip in root.findall("tripinfo")]

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # filter out the columns that are not needed
    cols = [
        "id",
        "depart",
        "departDelay",
        "arrival",
        "routeLength",
        "duration",
        "waitingTime",
        "timeLoss",
        "vType",
        "speedFactor",
    ]
    df = df[cols]
    # print(df.shape)

    df = flatten_by_id(df)

    # print(df.shape)

    # keep only the columns that contain words from the cols

    return df


def load_routeRL(file) -> pd.DataFrame:
    """
    Load RouteRL output file and return a DataFrame.
    """

    # load the csv file
    try:
        df = pd.read_csv(file)
    except pd.errors.ParserError:
        print(f"Error parsing file: {file}")
        return pd.DataFrame()

    # convert to numeric
    try:
        df = df.apply(pd.to_numeric)
    except ValueError:
        pass

    df = flatten_by_id(df)

    # print(df.shape)

    return df


def load_episode(results_path: str, episode: int) -> pd.DataFrame:

    if episode % 100 == 0:
        print("loading episode: ", episode)
    SUMO_path = os.path.join(results_path, "SUMO_output")
    RouteRL_path = os.path.join(results_path, "episodes")
    Detectors_path = os.path.join(results_path, "detectors")

    SUMO_files = []
    RouteRL_files = []
    Detectors_files = []

    # find files in the directories

    for root, dirs, files in os.walk(SUMO_path):
        for file in files:
            if file.endswith("_" + str(episode) + ".xml"):
                SUMO_files.append(os.path.join(root, file))

    for root, dirs, files in os.walk(RouteRL_path):
        for file in files:
            if file.endswith("ep" + str(episode) + ".csv"):
                RouteRL_files.append(os.path.join(root, file))

    for root, dirs, files in os.walk(Detectors_path):
        for file in files:
            if file.endswith("ep" + str(episode) + ".csv"):
                Detectors_files.append(os.path.join(root, file))

    dfs = []
    for file in SUMO_files:
        if "detailed" in file:
            df = load_detailed_SUMO(file)
            # print("Detailed SUMO file loaded.")
            dfs.append(df)
        else:
            df = load_general_SUMO(file)
            # print("General SUMO file loaded.")
            dfs.append(df)

    for file in RouteRL_files:
        df = load_routeRL(file)
        # print("RouteRL file loaded.")
        dfs.append(df)

    for file in Detectors_files:
        pass

    for i in range(len(dfs)):
        if i == 0:
            df = dfs[i]
        else:
            df = pd.concat([df, dfs[i]], axis=1)

    # print(df.shape)
    # add first column - "episode"
    df.insert(0, "episode", episode)

    return df


def add_benchmark_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add benchmark columns to the DataFrame.
    """
    n_agents = df["vehicleTripStatistics_count"][0]

    new_columns = {}
    for i in range(n_agents):
        col = (df[f"agent_{i}_action"] != df[f"agent_{i}_action"].shift(1)).astype(int)
        new_columns[f"agent_{i}_action_change"] = col

    # add the new columns to the DataFrame
    df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)

    return df


def collect_to_single_CSV(path, save_path="KPIs.csv"):
    """
    Collect KPIs of the experiment to the single CSV file.
    """

    df = pd.DataFrame()

    episodes = get_episodes(os.path.join(path, "episodes"))

    for i in episodes:
        # add new rows to the DataFrame
        df = pd.concat([df, load_episode(path, i)], ignore_index=True)

    # add benchmark columns
    df = add_benchmark_columns(df)

    df.to_csv(save_path, index=False)

    return df


def get_type_ids(df: pd.DataFrame, type: str) -> list:
    # Keep the last row as a DataFrame, not Series
    df = df.iloc[[-1]]

    type_IDs = [
        col.split("_")[1]
        for col in df.columns
        if col.startswith("agent_")
        and col.endswith("vType")
        and (df[col] == type).any()
    ]

    # Cast to int
    type_IDs = [int(id) for id in type_IDs]

    # print(f"IDs of {type} agents: {type_IDs}")

    return type_IDs


def extract_KPIs(path, config):
    """
    Extract KPIs from the DataFrame.
    """

    training_duration = config["human_learning_episodes"] + config["training_eps"]

    df = pd.read_csv(path)

    CAV_ids = get_type_ids(df, "AV")

    human_ids = get_type_ids(df, "Human")

    testing_frames = df[df["episode"] > training_duration]

    before_mutation = df[df["episode"] <= config["human_learning_episodes"]]

    rho = 0
    for id in CAV_ids:
        rho += testing_frames[f"agent_{id}_duration"].mean()

    rho /= len(CAV_ids)

    tau = 0
    for id in human_ids:
        tau += testing_frames[f"agent_{id}_duration"].mean()
    tau /= len(human_ids)

    tau_b = 0
    for id in human_ids:
        tau_b += before_mutation[f"agent_{id}_duration"].mean()
    tau_b /= len(human_ids)

    mean_TT_humans = np.mean([df[f"agent_{id}_duration"] for id in human_ids])
    mean_TT_CAVs = np.mean(
        [df[f"agent_{id}_duration"] for id in CAV_ids]
    )  # TODO maybe after mutation?

    mean_TT_all = np.mean(df["vehicleTripStatistics_totalTravelTime"])
    mean_TT_all = mean_TT_all / (len(CAV_ids) + len(human_ids))
    # extract KPIs of the experiment

    avg_route_length = np.mean(df["vehicleTripStatistics_routeLength"])

    avg_speed = np.mean(df["vehicleTripStatistics_speed"])

    
    min_human_times = [np.min(df[f"agent_{id}_duration"]) for id in human_ids]

    min_CAV_times = [np.min(df[f"agent_{id}_duration"]) for id in CAV_ids]

    max_human_times = [np.max(df[f"agent_{id}_duration"]) for id in human_ids]

    max_CAV_times = [np.max(df[f"agent_{id}_duration"]) for id in CAV_ids]

    mean_human_diff = np.mean(
        [max_human_times[i] - min_human_times[i] for i in range(len(min_human_times))]
    )
    mean_CAV_diff = np.mean(
        [max_CAV_times[i] - min_CAV_times[i] for i in range(len(min_CAV_times))]
    )

    KPIs = {}

    KPIs["rho"] = rho
    KPIs["tau"] = tau
    KPIs["tau_b"] = tau_b
    KPIs["mean_TT_humans"] = mean_TT_humans
    KPIs["mean_TT_CAVs"] = mean_TT_CAVs
    KPIs["CAV_advantage"] = tau / rho
    KPIs["Effect_of_change"] = tau_b / rho
    KPIs["Effect_of_remaining"] = tau_b / tau
    KPIs["mean_TT_all"] = mean_TT_all
    KPIs["avg_speed"] = avg_speed
    KPIs["avg_route_length"] = avg_route_length
    KPIs["mean_human_diff"] = mean_human_diff
    KPIs["mean_CAV_diff"] = mean_CAV_diff

    # now KPIs that are not a single value but a list of values

    instability_humans = df[
        [f"agent_{id}_action_change" for id in human_ids]
    ].sum(axis=1).tolist()
    instability_CAVs = df[
        [f"agent_{id}_action_change" for id in CAV_ids]
    ].sum(axis=1).tolist()

    avg_time_lost = df["vehicleTripStatistics_timeLoss"] + df["vehicleTripStatistics_departDelay"]
    avg_time_lost = avg_time_lost.tolist()

    vector_KPIs = [
        instability_humans,
        instability_CAVs,
        avg_time_lost,
    ]

    return KPIs, vector_KPIs


def clear_SUMO_files(path, ep_path):
    file_id = 1
    episode = 1

    file_name = "detailed_sumo_stats"

    while True:
        # check if file exists
        file_path = os.path.join(path, f"{file_name}_{episode}.xml")
        if os.path.exists(file_path):
            # read xml file and check if <tripinfos> is empty (no <tripinfo> elements)
            try:
                tree = ET.parse(file_path)
            except ET.ParseError:
                print(f"Error parsing XML file: {file_path}")
                break
            root = tree.getroot()
            if len(root.findall("tripinfo")) == 0:
                # remove the file
                os.remove(file_path)
                # print(f"Removed empty file: {file_path}")
            else:
                # rename to the next file_id
                new_file_path = os.path.join(path, f"{file_name}_{file_id}.xml")
                os.rename(file_path, new_file_path)
                # print(f"Renamed file {file_path} to {new_file_path}")
                file_id += 1
        else:
            break
        episode += 1

    file_id = 1
    episode = 1

    file_name = "sumo_stats"

    while True:
        # check if file exists
        file_path = os.path.join(path, f"{file_name}_{episode}.xml")
        if os.path.exists(file_path):
            # read xml file and check if <vehicle loaded=0>
            try:
                tree = ET.parse(file_path)
            except ET.ParseError:
                print(f"Error parsing XML file: {file_path}")
                break
            root = tree.getroot()
            vehicle = root.find("vehicles")
            if vehicle is not None and vehicle.attrib.get("loaded") == "0":
                # remove the file
                os.remove(file_path)
                # print(f"Removed empty file: {file_path}")
            else:
                # rename to the next file_id
                new_file_path = os.path.join(path, f"{file_name}_{file_id}.xml")
                os.rename(file_path, new_file_path)
                # print(f"Renamed file {file_path} to {new_file_path}")
                file_id += 1
        else:
            break
        episode += 1

    episodes = get_episodes(ep_path)
    # remove SUMO files that are not in the episodes
    for file in os.listdir(path):
        if file.endswith(".xml"):
            episode = int(file.split("_")[-1].split(".")[0])
            if episode not in episodes:
                os.remove(os.path.join(path, file))
                # print(f"Removed file: {file}")


mock_path = "records/gar_aon/gar_aon_43"
records_folder = f"./records"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, required=True)
    args = parser.parse_args()
    exp_id = args.id

    data_path = None
    for root, dirs, files in os.walk(records_folder):
        if exp_id in dirs:
            data_path = os.path.join(root, exp_id)
            break

    # clear_SUMO_files(
    #     os.path.join(data_path, "SUMO_output"), os.path.join(data_path, "episodes")
    # )

    # collect_to_single_CSV(data_path, os.path.join(data_path, "combined_data.csv"))

    KPIs, vector_KPIs = extract_KPIs(
        os.path.join(data_path, "combined_data.csv"),
        {
            "human_learning_episodes": 500,
            "training_eps": 1000,
            "test_eps": 100,
        },
    )

    # save KPIs to csv
    KPIs_df = pd.DataFrame(KPIs, index=[0])
    KPIs_df.to_csv(os.path.join(data_path, "BenchmarkKPIs.csv"), index=False)
    print(KPIs_df)

    # save vector KPIs to csv
    vector_KPIs_df = pd.DataFrame(vector_KPIs).T
    vector_KPIs_df.columns = [
        "instability_humans",
        "instability_CAVs",
        "avg_time_lost",
    ]
    vector_KPIs_df.to_csv(os.path.join(data_path, "VectorKPIs.csv"), index=False)
