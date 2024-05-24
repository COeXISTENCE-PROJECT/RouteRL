from components import TrafficEnvironment
from components import SumoSimulator
from keychain import Keychain as kc
from services import Plotter
from services import Runner

from create_agents import create_agent_objects
from utilities import get_params
from utilities import set_seeds
from utilities import check_device


def main(params):
    simulator = SumoSimulator(params[kc.SIMULATOR])
    env = TrafficEnvironment(params[kc.ENVIRONMENT], simulator)
    agents = create_agent_objects(params[kc.AGENTS], env.get_free_flow_times())
    
    runner = Runner(params[kc.RUNNER])
    runner.run(env, agents)
    Plotter(params[kc.PLOTTER])


if __name__ == "__main__":
    check_device()
    set_seeds()
    params = get_params(kc.PARAMS_PATH)
    main(params)