import xml.etree.ElementTree as ET
import pandas as pd
import sys
import os

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
    
    # print(df.shape)
    
    df = flatten_by_id(df)
    
    # print(df.shape)
    
    return df

def load_routeRL(file) -> pd.DataFrame:
    """
    Load RouteRL output file and return a DataFrame.
    """
    
    # load the csv file
    df = pd.read_csv(file)
    
    # convert to numeric
    try:
        df = df.apply(pd.to_numeric)
    except ValueError:
        pass
    
    df = flatten_by_id(df)
    
    # print(df.shape)
    
    return df


def load_episode(results_path: str, episode: int) -> pd.DataFrame:
    
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
            print("Detailed SUMO file loaded.")
            dfs.append(df)
        else:
            df = load_general_SUMO(file)
            print("General SUMO file loaded.")
            dfs.append(df)
            
    for file in RouteRL_files:
        df = load_routeRL(file)
        print("RouteRL file loaded.")
        dfs.append(df)

    for file in Detectors_files:
        pass
    
    for i in range(len(dfs)):
        if i == 0:
            df = dfs[i]
        else:
            df = pd.concat([df, dfs[i]], axis=1)
    
    # print(df.shape)
            
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


def collect_to_single_CSV(path, n_episodes, save_path="KPIs.csv"):
    """
    Collect KPIs of the experiment to the single CSV file.
    """
    
    df = pd.DataFrame()
    
    for i in range(1, n_episodes + 1):
        # add new rows to the DataFrame
        df = pd.concat([df, load_episode(path, i)], ignore_index=True)
    
    # add benchmark columns
    df = add_benchmark_columns(df)
    
    df.to_csv(save_path, index=False)
    
    return df


def extract_KPIs(path):
    """
    Extract KPIs from the DataFrame.
    """
    
    df = pd.read_csv(path)
    
    # mean CAV time in last 100 episodes
    rho = df[:-100]
    
    # extract KPIs of the experiment    
    KPIs = {}
    
    return KPIs

def clearSumoFiles(path):
    file_id = 1
    episode = 1
    
    file_name = "detailed_sumo_stats"
    
    while True:
        # check if file exists
        file_path = os.path.join(path, f"{file_name}_{episode}.xml")
        if os.path.exists(file_path):
            # read xml file and check if <tripinfos> is empty (no <tripinfo> elements)
            tree = ET.parse(file_path)
            root = tree.getroot()
            if len(root.findall("tripinfo")) == 0:
                # remove the file
                os.remove(file_path)
                print(f"Removed empty file: {file_path}")
            else:
                # rename to the next file_id
                new_file_path = os.path.join(path, f"{file_name}_{file_id}.xml")
                os.rename(file_path, new_file_path)
                print(f"Renamed file {file_path} to {new_file_path}")
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
            tree = ET.parse(file_path)
            root = tree.getroot()
            vehicle = root.find("vehicles")
            if vehicle is not None and vehicle.attrib.get("loaded") == "0":
                # remove the file
                os.remove(file_path)
                print(f"Removed empty file: {file_path}")
            else:
                # rename to the next file_id
                new_file_path = os.path.join(path, f"{file_name}_{file_id}.xml")
                os.rename(file_path, new_file_path)
                print(f"Renamed file {file_path} to {new_file_path}")
                file_id += 1
        else:
            break
        episode += 1
    
mock_path = "training_records"

if __name__ == "__main__":
    
    clearSumoFiles(mock_path + "/SUMO_output")
    
    collect_to_single_CSV(mock_path, 100)   