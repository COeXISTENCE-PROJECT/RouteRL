import concurrent.futures
import pandas as pd
import time

from keychain import Keychain as kc
from recorder import Recorder
from services.utils import show_progress_bar


class Trainer:

    def __init__(self, params):
        self.num_episodes = params[kc.NUM_EPISODES]
        self.recorder_params = params[kc.RECORDER_PARAMETERS]


    def train(self, env, agents):
        start_time = time.time()
        self.recorder = Recorder(agents, self.recorder_params)

        for ep in range(self.num_episodes):    # Until we simulate num_episode episodes
            state = env.reset()
            done = False
            while not done:     # Until episode concludes
                joint_action = {kc.AGENT_ID : list(), kc.ACTION : list(), kc.AGENT_ORIGIN : list(), 
                                kc.AGENT_DESTINATION : list(), kc.AGENT_START_TIME : list()}

                for agent in agents:    # Every agent picks action
                    action = agent.act(state)
                    joint_action = self.add_action_to_joint_action(agent, action, joint_action)
                
                joint_action_df = pd.DataFrame(joint_action)
                joint_reward_df, next_state, done = env.step(joint_action_df)

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(self.learn_agent, agent, joint_action_df, joint_reward_df, state, next_state) for agent in agents]
                    concurrent.futures.wait(futures)

                state = next_state

            self.recorder.remember_all(ep, joint_action_df, joint_reward_df, agents)
            show_progress_bar("TRAINING", start_time, ep+1, self.num_episodes)

        self.recorder.remember_all(ep, joint_action_df, joint_reward_df, agents)
        print("\n[COMPLETE] Training completed in: %s" % (time.strftime("%H hours, %M minutes, %S seconds", time.gmtime(time.time() - start_time))))
        
        self.recorder.rewind()
        
        return agents
    

    def learn_agent(self, agent, joint_action_df, joint_reward_df, state, next_state):
        action = joint_action_df.loc[joint_action_df[kc.AGENT_ID] == agent.id, kc.ACTION].item()
        reward = joint_reward_df.loc[joint_action_df[kc.AGENT_ID] == agent.id, kc.REWARD].item()
        agent.learn(action, reward, state, next_state)
    

    def add_action_to_joint_action(self, agent, action, joint_action):  # Add individual action to joint action
        joint_action[kc.AGENT_ID].append(agent.id)
        joint_action[kc.AGENT_ORIGIN].append(agent.origin)
        joint_action[kc.AGENT_DESTINATION].append(agent.destination)
        joint_action[kc.AGENT_START_TIME].append(agent.start_time)
        joint_action[kc.ACTION].append(action)
        return joint_action