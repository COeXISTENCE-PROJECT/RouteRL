from create_agents import create_agent_objects
from environment import TrafficEnvironment
from trainer import Trainer
from utils import get_json


params_file_path = "params.json"


def main():

    params = get_json(params_file_path)

    agents = create_agent_objects(params["agent_generation_parameters"])
    env = TrafficEnvironment(agents)
    
    trainer = Trainer(params["training_parameters"])
    agents = trainer.train(env, agents)



main()