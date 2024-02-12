from environment import TrafficEnvironment
from keychain import Keychain as kc
from services import Trainer

from utilities import create_agent_objects
from utilities import confirm_env_variable
from utilities import get_params



def main(params):
    env = TrafficEnvironment(params[kc.ENVIRONMENT_PARAMETERS], params[kc.SIMULATION_PARAMETERS]) 
    agents = create_agent_objects(params[kc.AGENTS_GENERATION_PARAMETERS], env.calculate_free_flow_times())
    trainer = Trainer(params[kc.TRAINING_PARAMETERS])
    agents = trainer.train(env, agents)



if __name__ == "__main__":
    confirm_env_variable(kc.SUMO_HOME, append="tools")
    params = get_params(kc.PARAMS_PATH)
    main(params)