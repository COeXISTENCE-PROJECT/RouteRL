import concurrent.futures
import pandas as pd
import time

from collections import Counter

from keychain import Keychain as kc
from services import Plotter
from services import Recorder
from utilities import show_progress_bar

import matplotlib.pyplot as plt
class Trainer:

    """
    Class to train agents
    """

    def __init__(self, params):
        self.num_episodes = params[kc.NUM_EPISODES]
        self.phases = [1] + params[kc.PHASES]

        self.frequent_progressbar = params[kc.FREQUENT_PROGRESSBAR_UPDATE]
        self.remember_every = params[kc.REMEMBER_EVERY]
        self.remember_episodes = [ep for ep in range(self.remember_every, self.num_episodes+1, self.remember_every)]
        self.remember_episodes += [1, self.num_episodes] + [ep-1 for ep in self.phases] + [ep for ep in self.phases]
        self.remember_episodes = set(self.remember_episodes)

        self.recorder = Recorder(params[kc.RECORDER_PARAMETERS])
        self.plotter = Plotter(self.phases, self.recorder, params[kc.PLOTTER_PARAMETERS])


    # Training loop
    def train(self, env, agents):
        env.start()
        agents = sorted(agents, key=lambda x: x.start_time)

        print(f"\n[INFO] Training is starting with {self.num_episodes} episodes.")
        training_start_time = time.time()
        curr_phase = -1
        # Until we simulate num_episode episodes
        for episode in range(1, self.num_episodes+1):

            if episode in self.phases:
                curr_phase += 1
                print(f"\n[INFO] Phase {curr_phase} started at episode {episode}!")
                agents = self.mutate_agents(episode, curr_phase, agents)

            env.reset()
            self.submit_actions(env, agents)
            observation_df, info = env.step()
            self.teach_agents(agents, env.joint_action, observation_df)

            self.record(episode, training_start_time, env.joint_action, observation_df, agents, info[kc.LAST_SIM_DURATION])
            if self.frequent_progressbar: show_progress_bar("TRAINING", training_start_time, episode, self.num_episodes)

        self.show_training_time(training_start_time)
        env.stop()
        self.save_losses(agents)



    def submit_actions(self, env, agents):
        for agent in agents:
            observation = env.get_observation(agent.kind, agent.origin, agent.destination)
            action = agent.act(observation)
            env.register_action(agent, action)


    def teach_agents(self, agents, joint_action_df, observation_df):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.learn_agent, agent, joint_action_df, observation_df) for agent in agents]
            concurrent.futures.wait(futures)
        

    def learn_agent(self, agent, joint_action_df, observation_df):
        action = joint_action_df.loc[joint_action_df[kc.AGENT_ID] == agent.id, kc.ACTION].item()
        reward = observation_df.loc[observation_df[kc.AGENT_ID] == agent.id, kc.REWARD].item()
        agent.learn(action, reward)
    

    def record(self, episode, start_time, joint_action_df, joint_observation_df, agents, last_sim_duration):
        if (episode in self.remember_episodes):
            self.recorder.remember_all(episode, joint_action_df, joint_observation_df, agents, last_sim_duration)
            show_progress_bar("TRAINING", start_time, episode, self.num_episodes)


    def mutate_agents(self, episode, curr_phase, agents):
        anyone_mutated = False
        for idx, agent in enumerate(agents):
            if getattr(agent, 'mutate_phase', None) == curr_phase:
                agents[idx] = agent.mutate()
                anyone_mutated = True
        if anyone_mutated:
            counts = Counter([agent.kind for agent in agents])
            info_text = "[INFO] Some humans mutated: "
            info_text +=f" Humans: {counts[kc.TYPE_HUMAN]} " if counts[kc.TYPE_HUMAN] else ""
            info_text +=f" Machines: {counts[kc.TYPE_MACHINE]} " if counts[kc.TYPE_MACHINE] else ""
            info_text +=f" Disruptive Machines: {counts[kc.TYPE_MACHINE_2]}" if counts[kc.TYPE_MACHINE_2] else ""
            print(info_text)
        return agents


    def show_training_time(self, start_time):
        now = time.time()
        elapsed = now - start_time
        training_time = time.strftime("%H hours, %M minutes, %S seconds", time.gmtime(elapsed))
        sec_ep = "{:.2f}".format(elapsed/self.num_episodes)
        print(f"\n[COMPLETE] Training completed in: {training_time} ({sec_ep} s/e)")


    def show_training_results(self):
        self.plotter.visualize_all(self.recorder.saved_episodes)


    def save_losses(self, agents):
        self.recorder.save_losses(agents)