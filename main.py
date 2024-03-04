from environment import TrafficEnvironment
from keychain import Keychain as kc
from services import Trainer

from utilities import confirm_env_variable
from utilities import get_params


def main(params):
    env = TrafficEnvironment(params[kc.ENVIRONMENT_PARAMETERS], params[kc.SIMULATION_PARAMETERS], params[kc.AGENTS_GENERATION_PARAMETERS]) 
    trainer = Trainer(params[kc.TRAINING_PARAMETERS])
    trainer.train(env, env.agents)


if __name__ == "__main__":
    confirm_env_variable(kc.SUMO_HOME, append="tools")
    params = get_params(kc.PARAMS_PATH)
    main(params)