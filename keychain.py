class Keychain:

    """
    This is where we store our file paths, parameter access keys and other constants reliably
    When change needed, just fix it here! Avoid hardcoding...
    """

    ################## RELATIVE FILE PATHS ####################

    PARAMS_PATH = "params.json"

    AGENTS_DATA_PATH = "agents_data.csv"

    RECORDS_PATH = "training_records"
    REWARDS_LOGS_PATH = "rewards"
    ACTIONS_LOGS_PATH = "actions"
    Q_TABLES_LOG_PATH = "q_tables"
    
    ###########################################################
    
    

    
    ################ PARAMETER ACCESS KEYS ####################

    AGENTS_GENERATION_PARAMETERS = "agent_generation_parameters"
    TRAINING_PARAMETERS = "training_parameters"

    SIMULATION_TIMESTEPS = "simulation_timesteps"
    AGENT_START_INTERVALS = "agent_start_intervals"
    ACTION_SPACE_SIZE = "action_space_size"
    AGENT_LEARNING_PARAMETERS = "agent_learning_parameters"
    MACHINE_AGENT_PARAMETERS = "machine_agent_parameters"
    HUMAN_AGENT_PARAMETERS = "human_agent_parameters"

    MIN_ALPHA = "min_alpha"
    MAX_ALPHA = "max_alpha"
    MIN_EPSILON = "min_epsilon"
    MAX_EPSILON = "max_epsilon"
    MIN_EPS_DECAY = "min_eps_decay"
    MAX_EPS_DECAY = "max_eps_decay"

    BETA = "beta"

    # Training
    NUM_EPISODES = "num_episodes"

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
    AGENT_ATTRIBUTES = [AGENT_ID, AGENT_ORIGIN, AGENT_DESTINATION, AGENT_START_TIME, AGENT_TYPE]

    # Joint action df column headers
    ACTION = "action"

    # Joint rewards df column headers
    REWARD = "reward"

    # Q-Table log df headers
    Q_TABLE = "q_table"
    EPSILON = "epsilon"

    # Agent type encodings
    TYPE_HUMAN = "h"
    TYPE_MACHINE = "m"

    ###########################################################