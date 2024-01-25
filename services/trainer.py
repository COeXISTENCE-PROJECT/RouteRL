import concurrent.futures
import pandas as pd
import time

from keychain import Keychain as kc
from services.utils import make_dir, show_progress, show_progress_bar


class Trainer:

    def __init__(self, params):

        self.num_episodes = params[kc.NUM_EPISODES]
        self.log_every = params[kc.LOG_EVERY]

    def learn_agent(agent, joint_action_df, joint_reward_df, state, next_state):
        action = joint_action_df.loc[joint_action_df[kc.AGENT_ID] == agent.id, kc.ACTION]
        reward = joint_reward_df.loc[joint_action_df[kc.AGENT_ID] == agent.id, kc.REWARD]
        agent.learn(action, reward, state, next_state)


    def train(self, env, agents):
        start_time = time.time()

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

                # Parallelized version
                par_start_time = time.time()

                # Assuming `agents` is a list of Agent objects
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(self.learn_agent, agent, joint_action_df, joint_reward_df, state, next_state) for agent in agents]

                    # Wait for all futures to complete
                    concurrent.futures.wait(futures)

                parallel_time = time.time() - par_start_time


                """# Original sequential version
                par_start_time = time.time()
                
                for agent in agents:    # Every agent learns from received rewards
                    action = joint_action_df.loc[joint_action_df[kc.AGENT_ID] == agent.id, kc.ACTION].iloc[0]
                    reward = joint_reward_df.loc[joint_action_df[kc.AGENT_ID] == agent.id, kc.REWARD].iloc[0]
                    agent.learn(action, reward, state, next_state)

                sequential_time = time.time() - par_start_time

                print(f"Sequential Time: {sequential_time} seconds")
                print(f"Parallel Time: {parallel_time} seconds")   """   

                
                if not (ep % self.log_every):
                ########## Save training records
                    q_tab_df = pd.DataFrame({kc.AGENT_ID: [a.id for a in agents], kc.EPSILON: [f'%.2f' % (getattr(a, 'epsilon', -1)) for a in agents], 
                                        kc.Q_TABLE: [f"%.2f  %.2f  %.2f" % (getattr(a, 'q-table[0]', 1), getattr(a, 'q-table[1]', 1), getattr(a, 'q-table[2]', 1)) for a in agents]})

                    joint_reward_df.to_csv(make_dir(kc.RECORDS_PATH, kc.REWARDS_LOGS_PATH, f"rewards_ep%d.csv" % (ep)), index = False)
                    joint_action_df.to_csv(make_dir(kc.RECORDS_PATH, kc.ACTIONS_LOGS_PATH, f"actions_ep%d.csv" % (ep)), index = False)
                    q_tab_df.to_csv(make_dir(kc.RECORDS_PATH, kc.Q_TABLES_LOG_PATH, f"q_tables_ep%d.csv" % (ep)), index = False)
                ##########

                state = next_state

            show_progress("TRAINING", start_time, ep+1, self.num_episodes, end_line='\n')

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