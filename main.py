from environment import TrafficEnvironment
from keychain import Keychain as kc
from services import Trainer
from services import create_agent_objects
from services import get_json
import os
import sys

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")



def main():

    params = get_json(kc.PARAMS_PATH)

    agents = create_agent_objects(params[kc.AGENTS_GENERATION_PARAMETERS])
    env = TrafficEnvironment(agents) # pass some params for the simulation
    
    trainer = Trainer(params[kc.TRAINING_PARAMETERS])
    agents = trainer.train(env, agents)




main()