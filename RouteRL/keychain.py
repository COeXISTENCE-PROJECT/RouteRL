class Keychain:

    """
    This is where we store our file paths, parameter access keys and other constants reliably
    When change needed, just fix it here! Avoid hardcoding...
    """
    
    MODE = "" # empty string if not testing, test otherwise
    # In case of test, make sure to create a test param file named "testparams.json" in the root folder

    ################ PARAMETER ACCESS KEYS ####################

    AGENTS = "agent_parameters"
    RUNNER = "runner_parameters"
    ENVIRONMENT = "environment_parameters"
    SIMULATOR = "simulator_parameters"
    PLOTTER = "plotter_parameters"
    AGENT_GEN = "agent_generation_parameters"
    PATH_GEN = "path_generation_parameters"

    ### Agent parameters
    HUMAN_PARAMETERS = "human_parameters"
    MACHINE_PARAMETERS = "machine_parameters"
    # Common
    ACTION_SPACE_SIZE = "action_space_size"
    MODEL = "model"
    BEHAVIOR = "behavior"
    # Human
    BETA = "beta"
    BETA_RANDOMNESS = "beta_randomness"
    ALPHA_J = "alpha_j"
    ALPHA_ZERO = "alpha_zero"
    # Machine
    OBSERVED_SPAN = "observed_span"

    ### Environment
    NUMBER_OF_DAYS = "number_of_days"

    ### Simulator
    SUMO_TYPE = "sumo_type"
    SEED = 'seed'
    ENV_VAR = "env_var"
    SIMULATION_TIMESTEPS = "simulation_timesteps"

    ### Plotter
    COLORS = "colors"
    LINESTYLES = "linestyles"
    SMOOTH_BY = "smooth_by"
    DEFAULT_WIDTH = "default_width"
    DEFAULT_HEIGHT = "default_height"
    MULTIMODE_WIDTH = "multimode_width"
    MULTIMODE_HEIGHT = "multimode_height"
    DEFAULT_NUM_COLUMNS = "default_num_columns"
    PHASES = "phases"
    PHASE_NAMES = "phase_names"

    ### Agent generation
    NUM_AGENTS = "num_agents"
    AGENT_ATTRIBUTES = "agent_attributes"
    NEW_MACHINES_AFTER_MUTATION = "new_machines_after_mutation"
    CREATE_AGENTS_DATA_PATH = "create_agents_data_path"

    ### Path generation
    NUMBER_OF_PATHS = "number_of_paths"
    ORIGINS = "origins"
    DESTINATIONS = "destinations"
    WEIGHT = "weight"
    NUM_SAMPLES = "num_samples"
    MAX_PATH_LENGTH = "max_path_length"
    ROUTE_UTILITY_COEFFS = "route_utility_coeffs"

    ###########################################################
    

    #################### CONSTANTS ############################

    NOT_AVAILABLE = "N/A"
    
    # Common dataframe headers
    AGENT_ID = "id"
    AGENT_KIND = "kind"
    AGENT_ORIGIN = "origin"
    AGENT_DESTINATION = "destination"
    AGENT_START_TIME = "start_time"
    TO_MUTATE = "to_mutate"
    ACTION = "action"
    SUMO_ACTION = "sumo_action"
    REWARD = "reward"
    COST = "cost"
    STATE = "state"
    Q_TABLE = "q_table"
    ARRIVAL_TIME = "arrival_time"
    COST_TABLE = "cost_table"
    TRAVEL_TIME = "travel_time"
    PATH_INDEX = "path_index"
    FREE_FLOW_TIME = "free_flow_time"
    LAST_SIM_DURATION = "last_sim_duration"
    ORIGINS = "origins"
    DESTINATIONS = "destinations"
    PATH = "path"

    # Agent type encodings
    ALL = "All"
    TYPE_HUMAN = "Human"
    TYPE_MACHINE = "AV"

    # Behavior encodings
    SELFISH = "selfish"
    COMPETITIVE = "competitive"
    COLLABORATIVE = "collaborative"
    SOCIAL = "social"
    ALTRUISTIC = "altruistic"
    MALICIOUS = "malicious"

    # Model encodings
    GAWRON = "gawron"
    CULO = "culo"
    W_AVG = "w_avg"
    
    ###########################################################


    ####################### FILE PATHS ########################

    CONNECTION_FILE_PATH = "connection_file_path"
    EDGE_FILE_PATH = "edge_file_path"
    NOD_FILE_PATH = "nod_file_path"
    ROUTE_FILE_PATH = "route_file_path"
    ROUTE_SAVE_FILE_PATH = "route_save_file_path"
    SUMO_CONFIG_PATH = "sumo_config_path"
    PATHS_CSV_SAVE_DETECTORS = "paths_csv_save_detectors"
    NETWORK_XML = "network_xml"
    DETECTOR_XML_SAVE_PATH = "detector_xml_save_path"
    FREE_FLOW_TIMES_CSV = "free_flow_times_csv"
    TRIP_INFO_XML = "trip_info_xml"
    SUMMARY_XML = "summary_xml"
    SUMO_FCD = "sumo_fcd"
    
    PATHS_CSV_SAVE_PATH = "paths_csv_save_path"
    FIGURES_SAVE_PATH = "figures_save_path"
    AGENTS_DATA_PATH = "agents_data_path"
    SAVE_TRIPINFO_XML = "save_tripinfo_xml"
    SAVE_TRAJECTORIES_XML = "save_trajectories_xml"
    SAVE_FCD_BASED_SPEEDS = "save_fcd_based_speeds"
    SAVE_SUMMARY_XML = "save_summary_xml"

    RECORDS_FOLDER = "records_folder"
    EPISODES_LOGS_FOLDER = "episodes_logs_folder"
    SIMULATION_LENGTH_LOG_FILE_NAME = "simulation_length_log_file_name"
    LOSSES_LOG_FILE_NAME = "losses_log_file_name"
    DETECTOR_LOGS_FOLDER = 'detector_logs_folder'
    PATHS_CSV_FILE_NAME = "paths_csv_file_name"

    PLOTS_FOLDER = "plots_folder"
    REWARDS_PLOT_FILE_NAME = "reward_plot_file_name"
    TRAVEL_TIMES_PLOT_FILE_NAME = "travel_times_plot_file_name"
    TT_DIST_PLOT_FILE_NAME = "tt_dist_plot_file_name"
    FF_TRAVEL_TIME_PLOT_FILE_NAME = "ff_travel_time_plot_file_name"
    FLOWS_PLOT_FILE_NAME = "flows_plot_file_name"
    SIMULATION_LENGTH_PLOT_FILE_NAME = "simulation_length_plot_file_name"
    LOSSES_PLOT_FILE_NAME = "losses_plot_file_name"
    ACTIONS_PLOT_FILE_NAME = "actions_plot_file_name"
    ACTIONS_SHIFTS_PLOT_FILE_NAME = "actions_shifts_plot_file_name"
    MACHINE_AGENTS_EPSILONS_PLOT_FILE_NAME = "machine_agents_epsilons_plot_file_name"
    
    ###########################################################