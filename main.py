from environment import TrafficEnvironment
from keychain import Keychain as kc
from services import Trainer
from services import create_agent_objects
from services import confirm_env_variable
from services import get_json

confirm_env_variable(kc.SUMO_HOME, append="tools")
params = get_json(kc.PARAMS_PATH)


def main():

    agents = create_agent_objects(params[kc.AGENTS_GENERATION_PARAMETERS])
    env = TrafficEnvironment(agents) # pass some params for the simulation
    
    trainer = Trainer(params[kc.TRAINING_PARAMETERS])
    agents = trainer.train(env, agents)


main()