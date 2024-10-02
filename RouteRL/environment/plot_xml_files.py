from ..keychain import Keychain as kc
import os
import subprocess


def plot_all_xmls(episode: int) -> None:
    """
    Plot all relevant XML files for a specific episode.

    Args:
        episode (int): The current episode number.
    """
    plot_tripinfo(episode, kc.TRIP_INFO_XML)
    plot_fcd_trajectories(episode, kc.SUMO_FCD)
    plot_fcd_based_speeds(episode, kc.SUMO_FCD)
    #plot_network(episode, kc.NETWORK_XML)
    #plot_summary(episode, kc.SUMMARY_XML)


def plot_tripinfo(episode: int, xml_file: str) -> None:
    """
    Run the plotting script for the tripinfo XML file.

    Args:
        episode (int): The current episode number.
        xml_file (str): The path to the tripinfo XML file to plot.
    """
    directory_path = kc.SAVE_TRIPINFO_XML
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    command = [
        'python', kc.PLOT_XML,
        '-i', 'id',
        '-x', 'depart',
        '-y', 'departDelay',
        '-o', f"{directory_path}/plot-{episode}.png",
        '--scatterplot',
        '--xlabel', 'depart time [s]',
        '--ylabel', 'depart delay [s]',
        '--ylim', '0,40',
        '--xlim', '0,500',
        '--xticks', '0,500,200,10',
        '--yticks', '0,40,5,10',
        '--xgrid',
        '--ygrid',
        '--title', 'depart delay over depart time',
        '--titlesize', '16',
        xml_file
    ]

    run_command(command, episode, "tripinfo")


def plot_fcd_trajectories(episode: int, xml_file: str) -> None:
    """
    Plot all the trajectories over time.

    Args:
        episode (int): The current episode number.
        xml_file (str): The path to the FCD XML file to plot.
    """
    directory_path = kc.SAVE_TRAJECTORIES_XML
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    command = [
        'python', kc.PLOT_TRAJECTORIES,
        '-t', 'xy',
        "--legend",
        '-o', f"{directory_path}/plot-{episode}.png",
        '--filter-ids', '10,20,30,40,50,60,70,80,90,100',
        xml_file
    ]

    run_command(command, episode, "FCD")


def plot_fcd_based_speeds(episode: int, xml_file: str) -> None:
    """
    Plot the FCD based speeds over time.

    Args:
        episode (int): The current episode number.
        xml_file (str): The path to the FCD XML file to plot.
    """
    directory_path = kc.SAVE_FCD_BASED_SPEEDS
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    command = [
        'python', kc.PLOT_TRAJECTORIES,
        '-t', 'ts',
        '-o', f"{directory_path}/plot-{episode}.png",
        '--filter-ids', '10,20,30,40,50,60,70,80,90,100',
        xml_file
    ]

    run_command(command, episode, "FCD")

def plot_network(episode: int, xml_file: str) -> None:
    print(kc.NETWORK_XML)

    command = [
        'python', 'C:/Program Files (x86)/Eclipse/Sumo/tools/visualization/plot_net_speeds.py',
        '-n', kc.NETWORK_XML,
        '--xlim', '1000,25000',
        '--ylim', '2000,26000',
        '--edge-width', '.5',
        '-o', 'speeds2.png',
        '--minV', '0',
        '--maxV', '60',
        '--xticks', '16',
        '--yticks', '16',
        '--xlabel', '[m]',
        '--ylabel', '[m]',
        '--xlabelsize', '16',
        '--ylabelsize', '16',
        '--colormap', 'jet'
    ]

    

    run_command(command, episode, "NETWORK")


def plot_summary(episode, xml_file: str) -> None:
    """
    Run the plotting script for the summary XML file.

    Args:
        xml_file (str): The path to the summary XML file to plot.
    """
    directory_path = kc.SAVE_SUMMARY_XML
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    command = [
        'python', kc.PLOT_XML,
        '-x', 'x',
        '-y', 'y',
        '-o', f"{directory_path}/plot-{episode}.png",
        '--legend',
        xml_file
    ]

    run_command(command, "summary")


def run_command(command: list[str], episode: int | str, plot_type: str = "") -> None:
    """
    Execute a plotting command and handle errors.

    Args:
        command (list[str]): The command to execute.
        episode (int | str): The current episode number or description.
        plot_type (str): The type of plot being generated.
    """
    try:
        subprocess.run(command, check=True)
        print(f"Successfully plotted {plot_type} for episode {episode}.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while plotting {plot_type} for episode {episode}: {e}")
        print("Output:", e.output)
        print("Error Output:", e.stderr)