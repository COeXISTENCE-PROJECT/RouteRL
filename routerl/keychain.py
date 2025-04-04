class Keychain:

    """
    This is where we store our file paths, parameter access keys and other constants reliably
    When change needed, just fix it here.
    """

    ################ PARAMETER ACCESS KEYS ####################
    
    DEFAULTS_FILE = "defaults.json"
    PLOTTER_CONFIG_FILE = "plotter_config.json"

    AGENTS = "agent_parameters"
    ENVIRONMENT = "environment_parameters"
    SIMULATOR = "simulator_parameters"
    PLOTTER = "plotter_parameters"
    PATH_GEN = "path_generation_parameters"

    ### Agent parameters
    HUMAN_PARAMETERS = "human_parameters"
    MACHINE_PARAMETERS = "machine_parameters"
    # Common
    ACTION_SPACE_SIZE = "action_space_size"
    # Human
    MODEL = "model"
    NOISE_WEIGHT_AGENT = "noise_weight_agent"
    NOISE_WEIGHT_PATH = "noise_weight_path"
    NOISE_WEIGHT_DAY = "noise_weight_day"
    BETA = "beta"
    BETA_K_I_VARIABILITY = "beta_k_i_variability"
    EPSILON_I_VARIABILITY = "epsilon_i_variability"
    EPSILON_K_I_VARIABILITY = "epsilon_k_i_variability"
    EPSILON_K_I_T_VARIABILITY = "epsilon_k_i_t_variability"
    GREEDY = "greedy"
    GAMMA_C = "gamma_c"
    GAMMA_U = "gamma_u"
    REMEMBER = "remember"
    ALPHA_ZERO = "alpha_zero"
    ALPHAS = "alphas"
    # Machine
    BEHAVIOR = "behavior"
    OBSERVED_SPAN = "observed_span"
    OBSERVATION_TYPE = "observation_type"
    PREVIOUS_AGENTS = "previous_agents"
    PREVIOUS_AGENTS_PLUS_START_TIME = "previous_agents_plus_start_time"
    PREVIOUS_AGENTS_PLUS_START_TIME_DETECTOR_DATA = "previous_agents_plus_start_time_detector_data"
    GROUP_VICINITY = "group_vicinity"

    ### Environment
    NUMBER_OF_DAYS = "number_of_days"
    SAVE_EVERY = "save_every"

    ### Simulator
    NETWORK_NAME = "network_name"
    SUMO_TYPE = "sumo_type"
    SIMULATION_TIMESTEPS = "simulation_timesteps"
    STUCK_TIME = "stuck_time"

    ### Plotter
    RECORDS_FOLDER = "records_folder"
    PLOTS_FOLDER = "plots_folder"
    PLOT_CHOICES = "plot_choices"
    PHASES = "phases"
    PHASE_NAMES = "phase_names"
    SMOOTH_BY = "smooth_by"

    ### Agent generation
    NUM_AGENTS = "num_agents"
    NEW_MACHINES_AFTER_MUTATION = "new_machines_after_mutation"

    ### Path generation
    NUMBER_OF_PATHS = "number_of_paths"
    ORIGINS = "origins"
    DESTINATIONS = "destinations"
    WEIGHT = "weight"
    NUM_SAMPLES = "num_samples"
    VISUALIZE_PATHS = "visualize_paths"

    ###########################################################
    
    #################### CONSTANTS ############################

    NOT_AVAILABLE = "N/A"
    
    # Sumo
    ENV_VAR = "SUMO_HOME"
    
    # Common dataframe headers
    AGENT_ID = "id"
    AGENT_ORIGIN = "origin"
    AGENT_DESTINATION = "destination"
    AGENT_START_TIME = "start_time"
    AGENT_KIND = "kind"
    
    ACTION = "action"
    REWARD = "reward"
    COST = "cost"
    COST_TABLE = "cost_table"
    TRAVEL_TIME = "travel_time"
    FREE_FLOW_TIME = "free_flow_time"

    # Agent type encodings
    ALL = "All"
    TYPE_HUMAN = "Human"
    TYPE_MACHINE = "AV"

    # Behavior encodings
    SELFISH = "selfish"
    COMPETITIVE = "competitive"
    COLLABORATIVE = "collaborative"
    COOPERATIVE = "cooperative"
    SOCIAL = "social"
    ALTRUISTIC = "altruistic"
    MALICIOUS = "malicious"
    BEHAVIORS = [SELFISH, COMPETITIVE, COLLABORATIVE, COOPERATIVE, SOCIAL, ALTRUISTIC, MALICIOUS]

    # Model encodings
    GAWRON = "gawron"
    WEIGHTED = "weighted"
    RANDOM = "random"
    GENERAL_MODEL = "general_model"
    AON = "aon"
    HUMAN_MODELS = [GAWRON, WEIGHTED, RANDOM, AON]
    
    # Network names
    ARTERIAL = "arterial"
    COLOGNE = "cologne"
    CSOMOR = "csomor"
    GRID = "grid"
    INGOLSTADT = "ingolstadt"
    NGUYEN = "nguyen"
    ORTUZAR = "ortuzar"
    TWO_ROUTE_YIELD = "two_route_yield"
    MANHATTAN = "manhattan"
    NETWORK_NAMES = [ARTERIAL, COLOGNE, CSOMOR, GRID, INGOLSTADT, NGUYEN, ORTUZAR, TWO_ROUTE_YIELD, MANHATTAN]
    
    # Plot choice modes
    PLOT_NONE = "none"
    PLOT_ALL = "all"
    PLOT_BASIC = "basic"
    
    ###########################################################

    ####################### FILE PATHS ########################
    
    PATHS_CSV_FILE_NAME = "paths.csv"
    AGENTS_CSV_FILE_NAME = "agents.csv"
    
    NETWORK_FOLDER = "../networks/$net$/"
    CONNECTION_FILE_PATH = "../networks/$net$/$net$.con.xml"
    EDGE_FILE_PATH = "../networks/$net$/$net$.edg.xml"
    NOD_FILE_PATH = "../networks/$net$/$net$.nod.xml"
    SUMO_CONFIG_PATH = "../networks/$net$/$net$.sumocfg"
    ROU_FILE_PATH = "../networks/$net$/$net$.rou.xml"
    SUMO_FCD = "../networks/$net$/fcd.xml"
    ROUTE_XML_PATH = "../networks/$net$/route.rou.xml"
    DETECTORS_XML_PATH = "../networks/$net$/det.add.xml"
    DETECTORS_CSV_PATH = "../networks/$net$/detectors.csv"
    
    DEFAULT_ODS_PATH = "../networks/default_ods.json"
    
    EPISODES_LOGS_FOLDER = "episodes"
    DETECTOR_LOGS_FOLDER = "detector"
    SUMO_LOGS_FOLDER = "SUMO_output"
    DETECTOR_STOPPED_VEHICLES = "detectors_stopped_vehicles"
    LOSSES_LOG_FILE_NAME = "losses.txt"

    REWARDS_PLOT_FILE_NAME = "rewards.png"
    TRAVEL_TIMES_PLOT_FILE_NAME = "travel_times.png"
    TT_DIST_PLOT_FILE_NAME = "tt_dist.png"
    SIM_LENGTH_PLOT_FILE_NAME = "simulation_length.png"
    LOSSES_PLOT_FILE_NAME = "losses.png"
    ACTIONS_PLOT_FILE_NAME = "actions.png"
    ACTIONS_SHIFTS_PLOT_FILE_NAME = "actions_shifts.png"
    
    ###########################################################

    ###################### ZENODO DATA ########################

    ZENODO_RECORD_ID = 14866276

    ###########################################################