import pandas as pd
import time

from keychain import Keychain as kc
from services.utils import make_dir, show_progress, show_progress_bar


class Trainer:

    def __init__(self, params):

        self.num_episodes = params[kc.NUM_EPISODES]


    def train(self, env, agents):
        state = env.reset()
        start_time = time.time()

        for ep in range(self.num_episodes):    # Until we simulate num_episode episodes
            done = False
            while not done:     # Until episode concludes
                joint_action = {kc.AGENT_ID : list(), kc.ACTION : list(), kc.AGENT_ORIGIN : list(), 
                                kc.AGENT_DESTINATION : list(), kc.AGENT_START_TIME : list()}

                for agent in agents:    # Every agent picks action
                    action = agent.act(state)
                    joint_action = self.add_action_to_joint_action(agent, action, joint_action)
                
                joint_action_df = pd.DataFrame(joint_action)
                joint_reward_df, next_state, done = env.step(joint_action_df)
                
                for agent in agents:    # Every agent learns from received rewards
                    action = joint_action_df.loc[joint_action_df[kc.AGENT_ID] == agent.id, kc.ACTION]
                    reward = joint_reward_df.loc[joint_action_df[kc.AGENT_ID] == agent.id, kc.REWARD]
                    agent.learn(action, reward, state, next_state)
                
                ########## Save training records
                q_tab_df = pd.DataFrame({kc.AGENT_ID: [a.id for a in agents], kc.EPSILON: [f'%.2f' % (a.epsilon) for a in agents], 
                                       kc.Q_TABLE: [f"%.2f  %.2f  %.2f" % (a.q_table[0], a.q_table[1], a.q_table[2]) for a in agents]})

                joint_reward_df.to_csv(make_dir(kc.RECORDS_PATH, kc.REWARDS_LOGS_PATH, f"rewards_ep%d.csv" % (ep)), index = False)
                joint_action_df.to_csv(make_dir(kc.RECORDS_PATH, kc.ACTIONS_LOGS_PATH, f"actions_ep%d.csv" % (ep)), index = False)
                q_tab_df.to_csv(make_dir(kc.RECORDS_PATH, kc.Q_TABLES_LOG_PATH, f"q_tables_ep%d.csv" % (ep)), index = False)
                ##########

                del joint_action, joint_reward_df, joint_action_df, q_tab_df
                state = next_state
            
            state = env.reset()
            show_progress("Training", start_time, ep+1, self.num_episodes, end_line='\n')

        print("\n[COMPLETE] Training completed in: %s" % (time.strftime("%H hours, %M minutes, %S seconds", time.gmtime(time.time() - start_time))))

        return agents
    

    def add_action_to_joint_action(self, agent, action, joint_action):
        """
        Add individual action to joint action
        """
        joint_action[kc.AGENT_ID].append(agent.id)
        joint_action[kc.AGENT_ORIGIN].append(agent.origin)
        joint_action[kc.AGENT_DESTINATION].append(agent.destination)
        joint_action[kc.AGENT_START_TIME].append(agent.start_time)
        joint_action[kc.ACTION].append(action)
        return joint_action