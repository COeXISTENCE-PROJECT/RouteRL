from environment import SumoSimulator
from environment import TrafficEnvironment
from keychain import Keychain as kc
from services import plotter
from services import runner
from tqdm import tqdm

from create_agents import create_agent_objects
from utilities import check_device
from utilities import get_params
from utilities import set_seeds

from pettingzoo.test import api_test


def main(params):
    env = TrafficEnvironment(params[kc.RUNNER], params[kc.ENVIRONMENT], params[kc.SIMULATOR], params[kc.AGENT_GEN], params[kc.AGENTS]) 

    env.start()
    env.reset()

    max_iter = 35200 #because 352 machine agents * 100 episodes (days)
    for agent in tqdm(env.agent_iter(max_iter=max_iter), total=max_iter):
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            # this is where you would insert your policy
            action = env.action_space(agent).sample()

        env.step(action)

    env.stop()

    plotter(params[kc.PLOTTER])


if __name__ == "__main__":
    check_device()
    set_seeds()
    params = get_params(kc.PARAMS_PATH)
    main(params)