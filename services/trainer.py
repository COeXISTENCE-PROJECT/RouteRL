import concurrent.futures
import pandas as pd
import time

from keychain import Keychain as kc
from services import Plotter
from services import Recorder
from utilities import show_progress_bar


class Trainer:

    """
    Class to train agents
    """

    def __init__(self, params):
        self.recorder_params = params[kc.RECORDER_PARAMETERS]
        self.plotter_params = params[kc.PLOTTER_PARAMETERS]

        self.num_episodes = params[kc.NUM_EPISODES]
        self.mutation_time = params[kc.MUTATION_TIME]

        self.remember_every = params[kc.REMEMBER_EVERY]
        self.remember_also = {1, self.num_episodes, self.mutation_time-1, self.mutation_time}


    def train(self, env, agents):
        # Create record & plot objects
        self.recorder = Recorder(self.recorder_params)
        self.plotter = Plotter(self.mutation_time, self.recorder.episodes_folder, self.recorder.agents_folder, self.recorder.sim_length_file_path, self.plotter_params)
        env.start()

        agents = self.sort_agents(agents)

        print(f"\n[INFO] Training is starting with {self.num_episodes} episodes.")
        start_time = time.time()

        for ep in range(1, self.num_episodes+1):    # Until we simulate num_episode episodes

            if ep == self.mutation_time:    agents = self.mutate_agents(agents)
            env.reset()
            done = False

            while not done:     # Until the episode concludes
                for agent in agents:
                    observation = env.get_observation(agent.origin, agent.destination)
                    action = agent.act(observation)
                    env.register_action(agent, action)
                observation, terminated, truncated, info = env.step()
                self.teach_agents(agents, env.joint_action, observation)
                done = all(terminated.values())

            self.record(ep, env.joint_action, observation[[kc.AGENT_ID, kc.TRAVEL_TIME]], agents, info[kc.LAST_SIM_DURATION])
            show_progress_bar("TRAINING", start_time, ep+1, self.num_episodes)

        self.show_training_time(start_time)
        env.stop()
        self.plotter.visualize_all(self.recorder.saved_episodes)


    def sort_agents(self, agents):
        return sorted(agents, key=lambda x: x.start_time)


    def teach_agents(self, agents, joint_action_df, observation):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.learn_agent, agent, joint_action_df, observation) for agent in agents]
            concurrent.futures.wait(futures)
        

    def learn_agent(self, agent, joint_action_df, observation):
        action = joint_action_df.loc[joint_action_df[kc.AGENT_ID] == agent.id, kc.ACTION].item()
        agent.learn(action, observation)
    

    def record(self, episode, joint_action_df, joint_reward_df, agents, last_sim_duration):
        if (not (episode % self.remember_every)) or (episode in self.remember_also):
            self.recorder.remember_all(episode, joint_action_df, joint_reward_df, agents, last_sim_duration)


    def mutate_agents(self, agents):
        for idx, agent in enumerate(agents):
            if agent.mutate_to is not None:
                new_agent = agent.mutate()
                agents[idx] = new_agent
        return agents


    def show_training_time(self, start_time):
        now = time.time()
        training_time = time.strftime("%H hours, %M minutes, %S seconds", time.gmtime(now - start_time))
        print(f"\n[COMPLETE] Training completed in: {training_time}")