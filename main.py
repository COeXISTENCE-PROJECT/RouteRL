from environment import TrafficEnvironment
from keychain import Keychain as kc
from services import Trainer
from services import create_agent_objects
from services import confirm_env_variable
from services import get_params
from sumo_controller import SumoController


confirm_env_variable(kc.SUMO_HOME, append="tools")
params = get_params(kc.PARAMS_PATH)


def main():

    env = TrafficEnvironment(params[kc.SIMULATION_PARAMETERS],params[kc.AGENTS_GENERATION_PARAMETERS][kc.AGENTS_DATA_PATH]) 
    agents = create_agent_objects(params[kc.AGENTS_GENERATION_PARAMETERS], env.calculate_free_flow_times())

    sumo_ctrl=SumoController(params)
    sumo_ctrl.sumo_start()

    trainer = Trainer(params[kc.TRAINING_PARAMETERS])
    agents = trainer.train(env, agents)

    sumo_ctrl.sumo_stop()


if __name__ == "__main__":
    main()