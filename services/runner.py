import time
import threading

from collections import Counter
from copy import deepcopy as dc

from .recorder import Recorder
from components import BaseAgent
from components import TrafficEnvironment
from keychain import Keychain as kc
from utilities import show_progress_bar

class Runner:

    """
    Run episodes, train agents, save data
    """

    def __init__(self, params):
        self.num_episodes = params[kc.NUM_EPISODES]
        self.phases = params[kc.PHASES]
        self.phase_names = params[kc.PHASE_NAMES]
        self.frequent_progressbar = params[kc.FREQUENT_PROGRESSBAR_UPDATE]
        self.remember_every = params[kc.REMEMBER_EVERY]
        
        self.remember_episodes = [ep for ep in range(self.remember_every, self.num_episodes+1, self.remember_every)]
        self.remember_episodes += [1, self.num_episodes] + [ep-1 for ep in self.phases] + [ep for ep in self.phases]
        self.remember_episodes = set(self.remember_episodes)
        self.recorder = Recorder()
        
        self.curr_phase = -1



    def run(self, env: TrafficEnvironment, agents: list[BaseAgent]):
        env.start()
        agents_by_start_time = {time: [a for a in agents if a.start_time == time] for time in range(max([agent.start_time for agent in agents])+1)}
        agents_by_id = {agent.id: agent for agent in agents}

        print(f"\n[INFO] Training is starting with {self.num_episodes} episodes.")
        training_start_time = time.time()
        
        # Until we simulate num_episode episodes
        for episode in range(1, self.num_episodes+1):

            # If new phase, apply it
            if episode in self.phases:
                while threading.active_count() > 1: time.sleep(1e-2)
                self.curr_phase += 1
                self._realize_phase(episode, agents)
                agents_by_start_time = {time: [a for a in agents if a.start_time == time] for time in range(max([agent.start_time for agent in agents])+1)}
                agents_by_id = {agent.id: agent for agent in agents}
                phase_start_time = time.time()

            # For each agent, get observation, act, collect travel times
            ep_observations = list()
            env.reset()
            while len(ep_observations) < len(agents):
                timestep, obs = env.get_observation()
                step_actions = [(agent, agent.act(obs)) for agent in agents_by_start_time.get(timestep, list())]
                ep_observations.extend(env.step(step_actions))
                
            # For each agent, learn
            for obs in ep_observations:
                agent = agents_by_id[obs[kc.AGENT_ID]]
                agent.learn(obs[kc.ACTION], ep_observations)
                
            # Record data
            recording_task = threading.Thread(target=self._record, args=(episode, ep_observations, phase_start_time, agents))
            recording_task.start()
            
        while threading.active_count() > 1: time.sleep(1e-2)
        self._show_training_time(training_start_time)
        env.stop()
        self._save_losses(agents)
    
    

    def _record(self, episode, ep_observations, start_time, agents):
        dc_episode, dc_ep_observations, dc_start_time, dc_agents = dc(episode), dc(ep_observations), dc(start_time), dc(agents)
        rewards = [{kc.AGENT_ID: agent.id, kc.REWARD: agent.last_reward} for agent in dc_agents]
        if (dc_episode in self.remember_episodes):
            self.recorder.record(dc_episode, dc_ep_observations, rewards)
        elif not self.frequent_progressbar:
            return
        msg = f"{self.phase_names[self.curr_phase]} {self.curr_phase+1}/{len(self.phases)}"
        curr_progress = dc_episode-self.phases[self.curr_phase]+1
        target = (self.phases[self.curr_phase+1]) if ((self.curr_phase+1) < len(self.phases)) else self.num_episodes+1
        target -= self.phases[self.curr_phase]
        show_progress_bar(msg, dc_start_time, curr_progress, target)



    def _realize_phase(self, episode, agents):
        for idx, agent in enumerate(agents):
            if getattr(agent, 'mutate_phase', None) == self.curr_phase:
                agents[idx] = agent.mutate()
            agents[idx].is_learning = self.curr_phase
            
        counts = Counter([agent.kind for agent in agents])
        info_text = "[INFO]"
        info_text +=f" {kc.TYPE_HUMAN}: {counts[kc.TYPE_HUMAN]} " if counts[kc.TYPE_HUMAN] else ""
        info_text +=f" {kc.TYPE_MACHINE}: {counts[kc.TYPE_MACHINE]} " if counts[kc.TYPE_MACHINE] else ""
        learning_situation = [agent.is_learning == True for agent in agents]
        
        print(f"\n[INFO] Phase {self.curr_phase+1} ({self.phase_names[self.curr_phase]}) has started at episode {episode}!")
        print(info_text)
        print(f"[INFO] Number of learning agents: {sum(learning_situation)}")



    def _show_training_time(self, start_time):
        now = time.time()
        elapsed = now - start_time
        training_time = time.strftime("%H hours, %M minutes, %S seconds", time.gmtime(elapsed))
        sec_ep = "{:.2f}".format(elapsed/self.num_episodes)
        print(f"\n[COMPLETE] Training completed in: {training_time} ({sec_ep} s/e)")



    def _save_losses(self, agents):
        self.recorder.save_losses(agents)
      

#############################################
        
        
def runner(env, agents, params):
    runner = Runner(params)
    runner.run(env, agents)
    return runner