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
        self.phases = params[kc.PHASES]
        self.phase_names = params[kc.PHASE_NAMES]

        self.frequent_progressbar = params[kc.FREQUENT_PROGRESSBAR_UPDATE]
        self.remember_every = params[kc.REMEMBER_EVERY]
        self.remember_episodes = [ep for ep in range(self.remember_every, self.num_episodes+1, self.remember_every)]
        self.remember_episodes += [1, self.num_episodes] + [ep-1 for ep in self.phases] + [ep for ep in self.phases]
        self.remember_episodes = set(self.remember_episodes)

        self.recorder = Recorder()
        self.plotter = Plotter(self.phases, self.phase_names)


    # Training loop
    def train(self, env, agents):
        env.start()
        agents = sorted(agents, key=lambda x: x.start_time)

        print(f"\n[INFO] Training is starting with {self.num_episodes} episodes.")
        training_start_time, phase_start_time = time.time(), time.time()
        curr_phase = -1
        # Until we simulate num_episode episodes
        for episode in range(1, self.num_episodes+1):

            if episode in self.phases:
                curr_phase += 1
                print(f"\n[INFO] Phase {curr_phase+1} ({self.phase_names[curr_phase]}) has started at episode {episode}!")
                agents = self.realize_phase(curr_phase, agents)
                phase_start_time = time.time()

            env.reset()
            self.submit_actions(env, agents)
            observation_df, info = env.step()
            self.teach_agents(agents, env.joint_action, observation_df)

            self.record(episode, phase_start_time, curr_phase, env.joint_action, observation_df, self.get_rewards(agents), agents, info[kc.LAST_SIM_DURATION])

        self.show_training_time(training_start_time)
        env.stop()
        self.save_losses(agents)



    def submit_actions(self, env, agents):
        for agent in agents:
            observation = env.get_observation()
            action = agent.act(observation)
            env.register_action(agent, action)


    def teach_agents(self, agents, joint_action_df, observation_df):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            teach_tasks = [executor.submit(self.learn_agent, agent, joint_action_df, observation_df) for agent in agents]
            concurrent.futures.wait(teach_tasks)
        

    def learn_agent(self, agent, joint_action_df, observation_df):
        action = joint_action_df.loc[joint_action_df[kc.AGENT_ID] == agent.id, kc.ACTION].item()
        agent.learn(action, observation_df)


    def get_rewards(self, agents):
        rewards_df = pd.DataFrame(columns=[kc.AGENT_ID, kc.REWARD])
        for agent in agents:
            reward = agent.last_reward
            rewards_df.loc[len(rewards_df.index)] = [agent.id, reward]
        return rewards_df
    

    def record(self, episode, start_time, curr_phase, joint_action_df, joint_observation_df, rewards_df, agents, last_sim_duration):
        if (episode in self.remember_episodes):
            self.recorder.remember_all(episode, joint_action_df, joint_observation_df, rewards_df, agents, last_sim_duration)
        elif not self.frequent_progressbar:
            return
        msg = f"{self.phase_names[curr_phase]} {curr_phase+1}/{len(self.phases)}"
        curr_progress = episode-self.phases[curr_phase]
        target = (self.phases[curr_phase+1]) if ((curr_phase+1) < len(self.phases)) else self.num_episodes
        target -= self.phases[curr_phase]+1
        show_progress_bar(msg, start_time, curr_progress, target)


    def realize_phase(self, curr_phase, agents):
        for idx, agent in enumerate(agents):
            if getattr(agent, 'mutate_phase', None) == curr_phase:
                agents[idx] = agent.mutate()
        for agent in agents:
            agent.is_learning = curr_phase
        counts = Counter([agent.kind for agent in agents])
        info_text = "[INFO]"
        info_text +=f" Humans: {counts[kc.TYPE_HUMAN]} " if counts[kc.TYPE_HUMAN] else ""
        info_text +=f" Machines: {counts[kc.TYPE_MACHINE]} " if counts[kc.TYPE_MACHINE] else ""
        print(info_text)
        learning_situation = [agent.is_learning for agent in agents]
        print(f"[INFO] Number of learning agents: {sum(learning_situation)}")
        return agents


    def show_training_time(self, start_time):
        now = time.time()
        elapsed = now - start_time
        training_time = time.strftime("%H hours, %M minutes, %S seconds", time.gmtime(elapsed))
        sec_ep = "{:.2f}".format(elapsed/self.num_episodes)
        print(f"\n[COMPLETE] Training completed in: {training_time} ({sec_ep} s/e)")


    def show_training_results(self):
        self.plotter.visualize_all()


    def save_losses(self, agents):
        self.recorder.save_losses(agents)