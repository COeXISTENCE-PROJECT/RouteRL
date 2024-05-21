from environment import TrafficEnvironment
from keychain import Keychain as kc
from services import Trainer

from utilities import get_params
from utilities import set_seeds
from utilities import check_device
from utilities import create_agent_objects


def main(params):
    env = TrafficEnvironment(params[kc.ENVIRONMENT_PARAMETERS], params[kc.SIMULATION_PARAMETERS])
    agents = create_agent_objects(params[kc.AGENTS_GENERATION_PARAMETERS], env.calculate_free_flow_times())
    trainer = Trainer(params[kc.TRAINING_PARAMETERS])
    trainer.train(env, agents)
    trainer.show_training_results()


if __name__ == "__main__":
    check_device()
    set_seeds()
    params = get_params(kc.PARAMS_PATH)
    main(params)