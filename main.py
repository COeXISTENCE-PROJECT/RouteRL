from environment import TrafficEnvironment
from keychain import Keychain as kc
from services import Trainer
from services import create_agent_objects
from services import confirm_env_variable
from services import get_params


confirm_env_variable(kc.SUMO_HOME, append="tools")

params = get_params(kc.PARAMS_PATH)


def main():


    env = TrafficEnvironment(params[kc.SIMULATION_PARAMETERS])
    agents = create_agent_objects(params[kc.AGENTS_GENERATION_PARAMETERS], env.calculate_free_flow_times())

    trainer = Trainer(params[kc.TRAINING_PARAMETERS])
    agents = trainer.train(env, agents)


main() # put in the if __name__ == "__main__":