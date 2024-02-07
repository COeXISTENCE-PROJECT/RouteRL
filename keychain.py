class Keychain:

    """
    This is where we store our file paths, parameter access keys and other constants reliably
    When change needed, just fix it here! Avoid hardcoding...
    """

    ################## RELATIVE FILE PATHS ####################

    PARAMS_PATH = "params.json"


    RECORDS_PATH = "training_records"
    REWARDS_LOGS_PATH = "rewards"
    ACTIONS_LOGS_PATH = "actions"
    Q_TABLES_LOG_PATH = "q_tables"
    ONE_AGENT_EXPERIENCE_LOG_PATH = "one_reward.csv"

    PLOTS_LOG_PATH = "plots"
    REWARDS_PLOT_FILE_NAME = "rewards.png"
    ONE_AGENT_PLOT_FILE_NAME = "one_agent.png"
    
    ###########################################################
    
    

    
    ################ PARAMETER ACCESS KEYS ####################

    AGENTS_GENERATION_PARAMETERS = "agent_generation_parameters"
    TRAINING_PARAMETERS = "training_parameters"
    SIMULATION_PARAMETERS = "simulation_parameters"

    # Agent generation
    AGENTS_DATA_PATH = "agents_data_path"
    NUM_AGENTS = "num_agents"
    SIMULATION_TIMESTEPS = "simulation_timesteps"

    ACTION_SPACE_SIZE = "action_space_size"
    MACHINE_AGENT_PARAMETERS = "machine_agent_parameters"
    HUMAN_AGENT_PARAMETERS = "human_agent_parameters"

    MIN_ALPHA = "min_alpha"
    MAX_ALPHA = "max_alpha"
    MIN_EPSILON = "min_epsilon"
    MAX_EPSILON = "max_epsilon"
    MIN_EPS_DECAY = "min_eps_decay"
    MAX_EPS_DECAY = "max_eps_decay"
    GAMMA = "gamma"

    BETA = "beta"
    ALPHA = "alpha"

    # Training
    NUM_EPISODES = "num_episodes"
    LOG_EVERY = "log_every"

    # Simulation
    SUMO_TYPE = "sumo_type"
    SUMO_CONFIG_PATH = "sumo_config_path"
    CONNECTION_FILE_PATH = "connection_file_path"
    EDGE_FILE_PATH = "edge_file_path" 
    ROUTE_FILE_PATH = "route_file_path"
    ROUTES_XML_SAVE_PATH = "routes_xml_save_path"
    
    PATHS_SAVE_PATH = "paths_save_path"
    NUMBER_OF_PATHS = "number_of_paths"

    ORIGINS = "origins"
    DESTINATIONS = "destinations"

    # Recorder
    RECORDER_PARAMETERS = "recorder_parameters"
    REMEMBER_EVERY = "remember_every"
    RECORDER_MODE = "mode"
    TRACK_HUMAN = "track_human"
    TRACK_MACHINE = "track_machine"

    ###########################################################
    
    

    
    ####################### ELSE ##############################

    SMALL_BUT_NOT_ZERO = 1e-14

    SUMO_HOME = "SUMO_HOME"
    
    # Agent attribute df column headers
    AGENT_ID = "id"
    AGENT_ORIGIN = "origin"
    AGENT_DESTINATION = "destination"
    AGENT_START_TIME = "start_time"
    AGENT_TYPE = "agent_type"
    AGENT_ATTRIBUTES = "agent_attributes"

    # Joint action df column headers
    ACTION = "action"

    # Joint rewards df column headers
    REWARD = "reward"

    # Q-Table log df headers
    Q_TABLE = "q_table"
    EPSILON = "epsilon"

    # SUMO df headers
    TRAVEL_TIMES = "travel_times"

    # Agent type encodings
    TYPE_HUMAN = "h"
    TYPE_MACHINE = "m"

    # Recorder modes
    PLOT_ONLY = "plot_only"
    SAVE_ONLY = "save_only"
    PLOT_AND_SAVE = "plot_and_save"

    ###########################################################