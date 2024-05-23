class Keychain:

    """
    This is where we store our file paths, parameter access keys and other constants reliably
    When change needed, just fix it here! Avoid hardcoding...
    """

    ################## RELATIVE FILE PATHS ####################

    PARAMS_PATH = "params.json"

    RECORDS_FOLDER = "training_records"
    EPISODES_LOGS_FOLDER = "episodes"
    AGENTS_LOGS_FOLDER = "agents"

    AGENTS_DATA_FILE_NAME = "agents_data.csv"
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
    
    

    
    ################ PARAMETER ACCESS KEYS ####################

    AGENTS_GENERATION_PARAMETERS = "agent_generation_parameters"
    TRAINING_PARAMETERS = "training_parameters"
    ENVIRONMENT_PARAMETERS = "environment_parameters"
    SIMULATION_PARAMETERS = "simulation_parameters"
    PATH_GENERATION_PARAMETERS = "path_generation_parameters"

    # Agent generation
    AGENTS_DATA_PATH = "agents_data_path"
    NUM_AGENTS = "num_agents"
    RATIO_MUTATING = "ratio_mutating"

    ACTION_SPACE_SIZE = "action_space_size"
    HUMAN_PARAMETERS = "human_parameters"
    MACHINE_PARAMETERS = "machine_parameters"

    MODEL = "model"
    APPEARANCE_PHASE = "appearance_phase"
    LEARNING_PHASES = "learning_phases"
    BEHAVIOR = "behavior"
    ALPHA = "alpha"
    BETA = "beta"
    BETA_RANDOMNESS = "beta_randomness"
    EPSILON = "epsilon"
    EPSILON_DECAY_RATE = "epsilon_decay_rate"
    BUFFER_SIZE = "buffer_size"
    BATCH_SIZE = "batch_size"
    LEARNING_RATE = "learning_rate"
    NUM_HIDDEN = "num_hidden"
    WIDTHS = "widths"

    # Training
    NUM_EPISODES = "num_episodes"
    REMEMBER_EVERY = "remember_every"
    PHASES = "phases"
    MUTATION_TIME = "mutation_time"
    SECOND_MUTATION_TIME = "second_mutation_time"
    FREQUENT_PROGRESSBAR_UPDATE = "frequent_progressbar_update"

    # Environment

    # Simulation
    SUMO_TYPE = "sumo_type"
    SUMO_CONFIG_PATH = "sumo_config_path"
    SIMULATION_TIMESTEPS = "simulation_timesteps"

    # Path generation
    CONNECTION_FILE_PATH = "connection_file_path"
    EDGE_FILE_PATH = "edge_file_path" 
    ROUTE_FILE_PATH = "route_file_path"
    ROUTES_XML_SAVE_PATH = "routes_xml_save_path"
    PATHS_CSV_SAVE_PATH = "paths_csv_save_path"
    PATHS_SAVE_PATH = "paths_save_path"
    NUMBER_OF_PATHS = "number_of_paths"
    ORIGINS = "origins"
    DESTINATIONS = "destinations"
    WEIGHT = "weight"

    # Recorder
    RECORDER_PARAMETERS = "recorder_parameters"

    # Plotter
    PLOTTER_PARAMETERS = "plotter_parameters"

    ###########################################################
    
    

    
    ####################### ELSE ##############################

    SMALL_BUT_NOT_ZERO = 1e-14
    NOT_AVAILABLE = "N/A"

    SUMO_HOME = "SUMO_HOME"

    AGENT_ATTRIBUTES = "agent_attributes"
    
    # Common dataframe column headers
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
    HUMANS = "humans"
    MACHINES = "machines"
    ALL = "all"
    PATH_INDEX = "path_index"
    FREE_FLOW_TIME = "free_flow_time"
    LAST_SIM_DURATION = "last_sim_duration"
    ORIGIN = "origin"
    DESTINATION = "destination"
    PATH = "path"

    # Agent type encodings
    TYPE_HUMAN = "human"
    TYPE_MACHINE = "machine"

    # Behavior encodings
    SELFISH = "selfish"
    DISRUPTIVE = "disruptive"
    SOCIAL = "social"
    ALTURISTIC = "alturistic"

    # Model encodings
    GAWRON = "gawron"
    DQN = "dqn"
    Q = "q"
    
    ###########################################################