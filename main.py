from components import SumoSimulator
from components import TrafficEnvironment
from keychain import Keychain as kc
from services import plotter
from services import runner

from create_agents import create_agent_objects
from utilities import check_device
from utilities import get_params
from utilities import set_seeds


def main(params):
    simulator = SumoSimulator(params[kc.SIMULATOR])
    env = TrafficEnvironment(params[kc.ENVIRONMENT], simulator)
    agents = create_agent_objects(params[kc.AGENTS], env.get_free_flow_times())
    runner(env, agents, params[kc.RUNNER])
    plotter(params[kc.PLOTTER])


if __name__ == "__main__":
    check_device()
    set_seeds()
    params = get_params(kc.PARAMS_PATH)
    main(params)