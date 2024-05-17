import concurrent.futures
import pandas as pd
import time

from collections import Counter

from keychain import Keychain as kc
from services import Plotter
from services import Recorder
from utilities import show_progress_bar


class Trainer:

    """
    Class to train agents
    """

    def __init__(self, params):
        self.num_episodes = params[kc.NUM_EPISODES]
        self.mutation_time, self.second_mutation_time  = params[kc.MUTATION_TIME], params[kc.SECOND_MUTATION_TIME]
        self.mutation_times = [self.mutation_time, self.second_mutation_time]

        self.frequent_progressbar = params[kc.FREQUENT_PROGRESSBAR_UPDATE]
        self.remember_every = params[kc.REMEMBER_EVERY]
        self.remember_also = {1, self.num_episodes, self.mutation_time-1, self.mutation_time, self.second_mutation_time-1, self.second_mutation_time}

        self.recorder = Recorder(params[kc.RECORDER_PARAMETERS])
        self.plotter = Plotter(self.mutation_time, self.second_mutation_time, self.recorder, params[kc.PLOTTER_PARAMETERS])



    # Training loop
    def train(self, env, agents):
        env.start()
        agents = sorted(agents, key=lambda x: x.start_time)

        print(f"\n[INFO] Training is starting with {self.num_episodes} episodes.")
        training_start_time = time.time()

        # Until we simulate num_episode episodes
        for episode in range(1, self.num_episodes+1):

            if episode in self.mutation_times:
                agents = self.mutate_agents(episode, agents)
            env.reset()
            done = False

            # Until the episode concludes
            while not done:
                self.submit_actions(env, agents)
                observation, info = env.step()
                self.teach_agents(agents, env.joint_action, observation)
                done = True     # can condition this

            self.record(episode, training_start_time, env.joint_action, observation, agents, info[kc.LAST_SIM_DURATION])
            if self.frequent_progressbar: show_progress_bar("TRAINING", training_start_time, episode, self.num_episodes)

        self.show_training_time(training_start_time)
        env.stop()



    def submit_actions(self, env, agents):
        for agent in agents:
            observation = env.get_observation(agent.kind, agent.origin, agent.destination)
            action = agent.act(observation)
            env.register_action(agent, action)


    def teach_agents(self, agents, joint_action_df, observation):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.learn_agent, agent, joint_action_df, observation) for agent in agents]
            concurrent.futures.wait(futures)
        

    def learn_agent(self, agent, joint_action_df, observation):
        action = joint_action_df.loc[joint_action_df[kc.AGENT_ID] == agent.id, kc.ACTION].item()
        agent.learn(action, observation)
    

    def record(self, episode, start_time, joint_action_df, joint_reward_df, agents, last_sim_duration):
        if (not (episode % self.remember_every)) or (episode in self.remember_also):
            self.recorder.remember_all(episode, joint_action_df, joint_reward_df, agents, last_sim_duration)
            show_progress_bar("TRAINING", start_time, episode, self.num_episodes)


    def mutate_agents(self, episode, agents):
        mutate_to = kc.TYPE_MACHINE if episode == self.mutation_time else kc.TYPE_MACHINE_2
        for idx, agent in enumerate(agents):
            if getattr(agent, 'mutate_type', None) == mutate_to:
                agents[idx] = agent.mutate()
        counts = Counter([agent.kind for agent in agents])
        mutated_to = "machines" if episode == self.mutation_time else "malicious machines"
        print(f"\n[INFO] Some humans mutated to {mutated_to} at episode {episode}!")
        print(f"[INFO] Number of humans: {counts[kc.TYPE_HUMAN]}, machines: {counts[kc.TYPE_MACHINE]}, malicious machines: {counts[kc.TYPE_MACHINE_2]}")
        return agents


    def show_training_time(self, start_time):
        now = time.time()
        training_time = time.strftime("%H hours, %M minutes, %S seconds", time.gmtime(now - start_time))
        print(f"\n[COMPLETE] Training completed in: {training_time}")


    def show_training_results(self):
        self.plotter.visualize_all(self.recorder.saved_episodes)