import matplotlib.pyplot as plt
import os
import pandas as pd
import sys
from keychain import Keychain as kc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from environment.environment import TrafficEnvironment
from services.plotter import plotter
from utilities import get_params

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

##################### Environment Creation #####################
params = get_params(kc.PARAMS_PATH)

env = TrafficEnvironment(params[kc.RUNNER], params[kc.ENVIRONMENT], params[kc.SIMULATOR], params[kc.AGENT_GEN], params[kc.AGENTS], params[kc.PHASE])

env.start()

##################### Human Learning #####################
num_episodes = 200

for episode in range(num_episodes):
    env.step()

env.stop()

plotter(params[kc.PLOTTER])