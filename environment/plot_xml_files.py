from keychain import Keychain as kc
import os
import subprocess
import shutil


def plot_all_xmls(episode: int) -> None:
    """
    Plot all relevant XML files for a specific episode.

    Args:
        episode (int): The current episode number.
    """
    plot_tripinfo(episode, kc.TRIP_INFO_XML)
    plot_fcd(episode, kc.SUMO_FCD)
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
        '--xticks', '0,1200,200,10',
        '--yticks', '0,40,5,10',
        '--xgrid',
        '--ygrid',
        '--title', 'depart delay over depart time',
        '--titlesize', '16',
        xml_file
    ]

    run_command(command, episode, "tripinfo")


def plot_fcd(episode: int, xml_file: str) -> None:
    """
    Run the plotting script for the FCD XML file.

    Args:
        episode (int): The current episode number.
        xml_file (str): The path to the FCD XML file to plot.
    """
    directory_path = kc.SAVE_TRAJECTORIES_XML
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    command = [
        'python', kc.PLOT_XML,
        '-x', 'x',
        '-y', 'y',
        '-o', f"{directory_path}/plot-{episode}.png",
        xml_file
    ]

    run_command(command, episode, "FCD")


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