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
    env = TrafficEnvironment(params[kc.RUNNER], params[kc.ENVIRONMENT], params[kc.SIMULATOR], params[kc.AGENT_GEN], params[kc.AGENTS]) 

    env.start()
    env.reset()
    env.stop()
    
    #runner(env, agents, params[kc.RUNNER])
    #plotter(params[kc.PLOTTER])


if __name__ == "__main__":
    check_device()
    set_seeds()
    params = get_params(kc.PARAMS_PATH)
    main(params)