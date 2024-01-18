from environment import TrafficEnvironment
from services.create_agents import create_agent_objects
from services.trainer import Trainer
from services.utils import get_json


params_file_path = "params.json"


def main():

    params = get_json(params_file_path)

    env = TrafficEnvironment() # pass some params
    agents = create_agent_objects(params["agent_generation_parameters"])
    
    trainer = Trainer(params["training_parameters"])
    agents = trainer.train(env, agents)



main()