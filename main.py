from environment import TrafficEnvironment
from keychain import Keychain as kc
from services.create_agents import create_agent_objects
from services.trainer import Trainer
from services.utils import get_json

def main():

    params = get_json(kc.PARAMS_PATH)

    env = TrafficEnvironment() # pass some params for the simulation
    agents = create_agent_objects(params[kc.AGENTS_GENERATION_PARAMETERS])
    
    trainer = Trainer(params[kc.TRAINING_PARAMETERS])
    agents = trainer.train(env, agents)



main()