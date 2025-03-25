"""
Import all the necessary modules for the routerl package.

Some example imports:
    from routerl import Keychain
    from routerl import TrafficEnvironment
    from routerl import SumoSimulator
    from routerl import get_learning_model
    from routerl import plotter
    from routerl import Recorder
    
    from routerl.learning import Gawron
    from routerl.environment import HumanAgent
    from routerl.environment import PreviousAgentStart
"""

__version__ = "1.0.0"

from .keychain import Keychain

from .environment import (
    generate_agents,
    generate_agent_data,
    SumoSimulator,
    TrafficEnvironment,
    MachineAgent
)

from .human_learning import (
    get_learning_model
)

from .services import (
    plotter,
    Plotter,
    Recorder
)