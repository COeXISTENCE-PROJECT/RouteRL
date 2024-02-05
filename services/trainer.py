import concurrent.futures
import pandas as pd
import time

from keychain import Keychain as kc
from services.utils import make_dir
from services.utils import show_progress, show_progress_bar

############ zoltan's request [1/4]
import random
from agent import HumanAgent
from services.utils import list_to_string, df_to_prettytable
############


class Trainer:

    def __init__(self, params):
        self.num_episodes = params[kc.NUM_EPISODES]
        self.log_every = params[kc.LOG_EVERY]


    def train(self, env, agents):

        start_time = time.time()

        ############ zoltan's request [2/4]
        one_human_cost_log = {key:list() for key in ['episode', 'agent_id', 'action', 'reward', 'cost_table']}
        human_to_watch = random.choice(agents)
        while not isinstance(human_to_watch, HumanAgent):
            human_to_watch = random.choice(agents) 
        ############

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

                if not (ep % self.log_every):
                ########## Save training records
                    joint_reward_df.to_csv(make_dir(kc.RECORDS_PATH, kc.REWARDS_LOGS_PATH, f"rewards_ep%d.csv" % (ep)), index = False)
                    joint_action_df.to_csv(make_dir(kc.RECORDS_PATH, kc.ACTIONS_LOGS_PATH, f"actions_ep%d.csv" % (ep)), index = False)
                ##########

                ############ zoltan's request [3/4]
                reward = joint_reward_df.loc[joint_action_df[kc.AGENT_ID] == human_to_watch.id, kc.REWARD].item()
                action = joint_action_df.loc[joint_action_df[kc.AGENT_ID] == human_to_watch.id, kc.ACTION].item()
                one_human_cost_log['episode'].append(ep)
                one_human_cost_log['agent_id'].append(human_to_watch.id)
                one_human_cost_log['action'].append(action)
                one_human_cost_log['reward'].append("%.3f" % reward)
                one_human_cost_log['cost_table'].append(list_to_string(human_to_watch.cost))
                ############

                state = next_state

            show_progress_bar("TRAINING", start_time, ep+1, self.num_episodes, end_line='\n')

        print("\n[COMPLETE] Training completed in: %s" % (time.strftime("%H hours, %M minutes, %S seconds", time.gmtime(time.time() - start_time))))

        ############ zoltan's request [4/4]
        one_human_cost_log_df = pd.DataFrame(one_human_cost_log)
        one_human_cost_log_df.to_csv('one_reward.csv',index=False)
        df_to_prettytable(one_human_cost_log_df, "HUMAN #{human_to_watch.id} EXPERIENCE")
        ############
        env.plot_rewards()
        
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