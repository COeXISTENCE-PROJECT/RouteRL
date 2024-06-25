import traci
import numpy as np

def sumo_start(config,seed): # RK: why sumo_start - maybe just .start()?
    sumo_binary = 'sumo'
    sumo_cmd = [sumo_binary,"--seed", seed , "-c", config] # RK: This string will go longer in future, with possible other sys args, so this shall be populated from config.
    traci.start(sumo_cmd)

def sumo_stop(): # RK: why sumo_stop - maybe just .stop()?
    traci.close()

def sumo_reset(config): # RK: why sumo_reset - maybe just .reset()?
    traci.load(['-c', config])

def run_simulation_iteration(config,route):

    seed = str(np.random.randint(1,high=10000))

    sumo_start(config,seed)
    time_start = traci.simulation.getTime()

    traci.vehicle.add('1',route)
    traci.vehicle.setImperfection('1',0.5)
    arrive = False

    while arrive==False:
        if not traci.simulation.getArrivedIDList():
            arrive=False
            traci.simulationStep()
        else:
            arrive=True
            time_end = traci.simulation.getTime()

    sumo_stop()

    TT=(time_end-time_start)

    return TT


def simulator():
    travel_time = []

    for i in range(10):
        TT = run_simulation_iteration("Network_and_config/grid4x4.sumocfg",'0_0_0')
        travel_time.append(TT)

    return travel_time

adat = simulator()
print(adat)