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
    APPEARANCE_PHASE = "appearance_phase"
    LEARNING_PHASES = "learning_phases"
    BEHAVIOR = "behavior"
    ALPHA = "alpha"
    # Human
    BETA = "beta"
    BETA_RANDOMNESS = "beta_randomness"
    # Machine
    OBSERVED_SPAN = "observed_span"
    EPSILON = "epsilon"
    EPSILON_DECAY_RATE = "epsilon_decay_rate"
    BUFFER_SIZE = "buffer_size"
    BATCH_SIZE = "batch_size"
    LEARNING_RATE = "learning_rate"
    NUM_HIDDEN = "num_hidden"
    WIDTHS = "widths"

    ### Runner
    NUM_EPISODES = "num_episodes"
    REMEMBER_EVERY = "remember_every"
    PHASES = "phases"
    PHASE_NAMES = "phase_names"
    FREQUENT_PROGRESSBAR_UPDATE = "frequent_progressbar_update"

    ### Environment

    ### Simulator
    SUMO_TYPE = "sumo_type"
    ENV_VAR = "env_var"
    SIMULATION_TIMESTEPS = "simulation_timesteps"

    ### Plotter
    COLORS = "colors"
    SMOOTH_BY = "smooth_by"
    DEFAULT_WIDTH = "default_width"
    DEFAULT_HEIGHT = "default_height"
    MULTIMODE_WIDTH = "multimode_width"
    MULTIMODE_HEIGHT = "multimode_height"
    DEFAULT_NUM_COLUMNS = "default_num_columns"

    ### Agent generation
    NUM_AGENTS = "num_agents"
    RATIO_MUTATING = "ratio_mutating"
    AGENT_ATTRIBUTES = "agent_attributes"

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
    ORIGIN = "origin"
    DESTINATION = "destination"
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
    DQN = "dqn"
    Q_LEARNING = "q"
    
    ###########################################################


    ####################### FILE PATHS ########################

    PARAMS_PATH = MODE + "params.json"

    CONNECTION_FILE_PATH = "network_and_config/csomor1.con.xml"
    EDGE_FILE_PATH = "network_and_config/csomor1.edg.xml"
    ROUTE_FILE_PATH = "network_and_config/csomor1.rou.xml"
    ROUTES_XML_SAVE_PATH = "network_and_config/route.rou.xml"
    SUMO_CONFIG_PATH = "network_and_config/csomor1.sumocfg"
    
    PATHS_CSV_SAVE_PATH = "network_and_config/paths.csv"
    AGENTS_DATA_PATH = "network_and_config/agents_data.csv"

    RECORDS_FOLDER = "training_records"
    EPISODES_LOGS_FOLDER = "episodes"
    SIMULATION_LENGTH_LOG_FILE_NAME = "simulation_length.txt"
    LOSSES_LOG_FILE_NAME = "losses.txt"
    PATHS_CSV_FILE_NAME = "paths.csv"
    FREE_FLOW_TIMES_CSV_FILE_NAME = "free_flow_times.csv"

    PLOTS_FOLDER = "plots"
    REWARDS_PLOT_FILE_NAME = "rewards.png"
    TRAVEL_TIMES_PLOT_FILE_NAME = "travel_times.png"
    TT_DIST_PLOT_FILE_NAME = "tt_dist.png"
    FF_TRAVEL_TIME_PLOT_FILE_NAME = "ff_travel_time.png"
    FLOWS_PLOT_FILE_NAME = "flows.png"
    SIMULATION_LENGTH_PLOT_FILE_NAME = "simulation_length.png"
    LOSSES_PLOT_FILE_NAME = "losses.png"
    ACTIONS_PLOT_FILE_NAME = "actions.png"
    ACTIONS_SHIFTS_PLOT_FILE_NAME = "actions_shifts.png"
    MACHINE_AGENTS_EPSILONS_PLOT_FILE_NAME = "epsilons.png"
    
    ###########################################################