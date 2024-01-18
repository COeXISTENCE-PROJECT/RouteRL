import pandas as pd
import time


class Trainer:

    def __init__(self, params):

        self.num_episodes = params["num_episodes"]
        paths = params["log_paths"]
        self.reward_logs_path = paths["reward_logs_path"]
        self.action_logs_path = paths["action_logs_path"]
        self.q_table_logs_path = paths["q_table_logs_path"]


    def train(self, env, agents):
        state = env.reset()
        start_time = time.time()

        for episode in range(self.num_episodes):    # Until we simulate num_episode episodes
            done = False
            while not done:     # Until episode concludes
                joint_action = {'id' : list(), 'action' : list(), 'origin' : list(), 'destination' : list(), 'start_time' : list()}

                for agent in agents:    # Every agent picks action
                    action = agent.pick_action(state)
                    joint_action = self.add_action_to_joint_action(agent, action, joint_action)
                
                joint_action_df = pd.DataFrame(joint_action)
                joint_reward_df, next_state, done = env.step(joint_action_df)
                
                for agent in agents:    # Every agent learns from received rewards
                    action = joint_action_df.loc[joint_action_df['id'] == agent.id, 'action']
                    reward = joint_reward_df.loc[joint_action_df['id'] == agent.id, 'reward']
                    agent.learn(action, reward, state, next_state)
                
                ########## Save training records
                joint_action_df['epsilon'] = [f'%.2f' % (a.epsilon) for a in agents]    # Just curiosity - how do epsilons decay?
                q_tables_df = pd.DataFrame({'id': [a.id for a in agents], 'q-table': [f"%.2f  %.2f  %.2f" % (a.q_table[0], a.q_table[1], a.q_table[2]) for a in agents]})

                joint_reward_df.to_csv(self.reward_logs_path + f"_%d.csv" % (episode), index = False)
                joint_action_df.to_csv(self.action_logs_path + f"_%d.csv" % (episode), index = False)
                q_tables_df.to_csv(self.q_table_logs_path + f"q_tables_%d.csv" % (episode), index = False)
                ##########

                del joint_action, joint_reward_df, joint_action_df, q_tables_df
                state = next_state
            
            state = env.reset()
            self.plot_progress(start_time, episode+1, self.num_episodes)

        print('\n[COMPLETE] Training completed!')
        return agents
    

    def add_action_to_joint_action(self, agent, action, joint_action):
        """
        Add individual action to joint action
        """
        joint_action['id'].append(agent.id)
        joint_action['origin'].append(agent.origin)
        joint_action['destination'].append(agent.destination)
        joint_action['start_time'].append(agent.start_time)
        joint_action['action'].append(action)
        return joint_action
    

    def plot_progress(self, start_time, progress, target):
        """
        Don't worry about it, just printing progress bar with ETA
        """
        bar_length = 50
        progress_fraction = progress / target
        filled_length = int(bar_length * progress_fraction)
        bar = 'X' * filled_length + '-' * (bar_length - filled_length)
        elapsed_time = time.time() - start_time
        remaining_time = ((elapsed_time / progress_fraction) - elapsed_time) if progress_fraction else 0
        print(f'\rProgress: |%s| %.2f%%, ETA: %.2f seconds' % (bar, progress_fraction * 100, remaining_time), end='')