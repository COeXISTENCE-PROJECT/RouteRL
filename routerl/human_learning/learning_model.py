import numpy as np
from collections import deque

from abc import ABC, abstractmethod

from routerl.keychain import Keychain as kc


class BaseLearningModel(ABC):
    """
    This is an abstract base class for the learning models used to model human learning and decision-making.\n
    Users can create their own learning models by inheriting from this class.
    """

    def __init__(self):
        pass

    @abstractmethod
    def act(self, state) -> None:
        """Method to select an action based on the current state and cost.

        Returns:
            None
        """
        pass

    @abstractmethod
    def learn(self, state, action, reward) -> None:
        """Method to learn the model based on the current state and cost.

        Arguments:
            state (Any): The current state of the environment.
            action (Any): The action to take.
            reward (Any): The reward received from the environment.
        Returns:
            None
        """

        pass


class GeneralModel(BaseLearningModel):
    """
    This is the general model of human route-choice behaviour which can accomodate several classic methods.
    It follows the two-step structure of `learn` and `act`.
    The new experience after travelling is used to update expected costs in the `learn`.
    The learned experiences is used to make routing decision (action) in `act`.

    * The variability (non-determinism) can be controlled at several levels:
        - ``epsilon_i_variability``, ``epsilon_k_i_variability``, ``epsilon_k_i_t_variability`` :
                variance of normal distribution from which error terms are drawn.
        - ``beta_k_i_variability`` : 
                variance of normal distribution for which ``$beta_k_i$`` is drawn (computed in utility).
        - ``noise_weight_agent``, ``noise_weight_path``, ``noise_weight_day`` : 
                relative weights for the error term composition.

    * The parameters of the model are:
        - ``beta``                  :   
                **negative value** - multiplier of reward (travel time) used in utility, determines sensitivity
        - ``greedy``                :   
                1 - exploration_rate , probability with which the choice are rational (argmax(U)) and not probabilistic (random)
        - ``gamma_u``, ``gamma_c``  :   
                bounded rationality components. Expressed as relative increase in costs/utilities below which user do not change behaviour (do not notice it)
        - ``alpha_zero``            :   
                weight with which the recent experience is carried forward in learning
        - ``remember``              :   
                number of days remembered to learn
        - ``alphas``                :   
                vector of weights for historically recorded reward in weighted average !!needs to be size of 'remember'!!
    
    """

    def __init__(self, params, initial_knowledge):
        """
        params is the dictionary, exactly like the one used in RouteRL
        
        initial knowledge shall come from SUMO - free flow travel time (or utilities)
        """
        super().__init__()
        self.costs = np.array(initial_knowledge, dtype=float) # cost matrix
        self.action_space = len(self.costs) # number of paths
        self.mean_time = np.mean(self.costs) # mean free flow travel time over paths - possibly useful to normalize errors
        
        # weights of respective componenets of error term (should sum to one)
        self.noise_weight_agent = params[kc.NOISE_WEIGHT_AGENT] # drawn once per agent
        self.noise_weight_path = params[kc.NOISE_WEIGHT_PATH] # drawn once per agent per path
        self.noise_weight_day = params[kc.NOISE_WEIGHT_DAY] # drawn daily per agent per path
        assert self.noise_weight_agent + self.noise_weight_path + self.noise_weight_day  == 1 , "Relative weights in error terms do not sum up to 1."

        # \beta_{t,k,i} agent-specific travel time multiplier per path
        self.beta_k_i = params[kc.BETA]*np.random.normal(1, params[kc.BETA_K_I_VARIABILITY], size = self.action_space) # used to compute Utility in `act`
        self.random_term_agent = np.random.normal(0, params[kc.EPSILON_I_VARIABILITY]) # \vareps_{i}
        self.random_term_path = np.random.normal(0, params[kc.EPSILON_K_I_VARIABILITY], size = self.action_space) # \vareps_{k,i}
        self.random_term_day = params[kc.EPSILON_K_I_T_VARIABILITY] # \vareps_{k,i,t}

        self.greedy = params[kc.GREEDY] # probability that agent will make a greedy choice (and not random exploration)

        self.gamma_c = params[kc.GAMMA_C] # bounded rationality on costs (relative)
        self.gamma_u = params[kc.GAMMA_U] # bounded rationality on utilities (relative)
        self.remember = int(params[kc.REMEMBER]) # window for averaging out

        self.alpha_zero = params[kc.ALPHA_ZERO] # weight of recent experience in learning

        self.alphas = list(params[kc.ALPHAS]) # weights for the memory in respective days
        assert len(self.alphas) == self.remember, "weights of history $\alpha_i in 'alphas' and 'remember do not match"
        assert abs(self.alpha_zero + sum(self.alphas) - 1) < 0.01 , "weights for weighted average do not sum up to 1"
        
        self.memory = [deque([cost],maxlen=self.remember) for cost in self.costs] # memories of experiences per path 
        self.first_day = True

    def learn(self, state, action, reward):
        """
        update 'costs' of 'action' after receiving a 'reward'
        """
        self.memory[action].append(reward) #add recent reward to memory (of rewards)

        log = {'action': action, 
               'reward':reward,
               'costs':self.costs,
               'gamma_c': self.gamma_c,
               'alpha_zero': self.alpha_zero,
               'alphas': self.alphas}
        print('learning prior:' + str(log))
      
        if abs(self.costs[action]-reward)/self.costs[action]>=self.gamma_c: #learn only if relative difference in rewards is above gamma_c
            weight_normalization_factor = 1/(self.alpha_zero+ sum([self.alphas[i] for i,j in enumerate(self.memory[action])])) # needed to make sure weights are always summed to 1
            self.costs[action] = weight_normalization_factor * self.alpha_zero* self.costs[action] #experience weights
            self.costs[action] += sum([weight_normalization_factor * self.alphas[i]*self.memory[action][i] for i,j in enumerate(self.memory[action])]) # weighted average of historical rewards

            log = {'action': action, 
               'reward':reward,
               'costs':self.costs,
               'gamma_c': self.gamma_c,
               'alpha_zero': self.alpha_zero,
               'alphas': self.alphas,
               'weight_normalization_factor' : weight_normalization_factor}
        else:
            print("I don't learn")

            log = {'action': action, 
               'reward':reward,
               'costs':self.costs,
               'gamma_c': self.gamma_c,
               'alpha_zero': self.alpha_zero,
               'alphas': self.alphas}
        
        print('learning after:' + str(log))

    def act(self, state):  
        """
        select path from action space based on learned expected costs"""
        # for each path you multiply the expected costs with path-specific beta (drawn at init) and add the noise (computed from 3 components in `get_noises`)
        utilities = [self.beta_k_i[i] * (self.costs[i] + self.mean_time * noise) for i, noise in enumerate(self.get_noises())]
        log = {'utilities': utilities, 
               'self.beta_k_i':self.beta_k_i,
               'costs':self.costs}
        print('I act based on those utilities:' + str(log))
        
        if self.first_day or abs(self.last_action["utility"] - utilities[self.last_action['action']])/self.last_action["utility"] >= self.gamma_u: #bounded rationality
            print("I act")
            if np.random.random() < self.greedy:
                action = int(np.argmax(utilities)) # greedy choice
            else:
                 print("I act random")
                 action = np.random.choice(self.action_space)  # random choice
        else:
            print("I do not act")
            action = self.last_action['action']    
        self.first_day = False
        self.last_action = {"action": action, "utility": utilities[action]}
        return action       
    
    def get_noises(self):
        """"
        compute random term for the utility, composed of 3 parts - two drawn at init and one here, inside

        returns vector of errors
        """
        daily_noise = np.random.normal(0,self.random_term_day, size= self.action_space)
        noise = [self.noise_weight_agent * self.random_term_agent + 
                self.noise_weight_path * self.random_term_path[k] + 
                self.noise_weight_day * daily_noise[k]
                    for k,_ in enumerate(self.costs)]
        print("this is my noise: ")
        print(noise)

        return [self.noise_weight_agent * self.random_term_agent + 
                self.noise_weight_path * self.random_term_path[k] + 
                self.noise_weight_day * daily_noise[k]
                    for k,_ in enumerate(self.costs)]
    

"""
Library of specific models implementations
"""
class ProbabilisticModel(GeneralModel):
    """
    Purely random choice decision model to be used as a baseline. It randomly selects an action from the action space.\n
    
    Args:
        params (dict): A dictionary containing model parameters.
        initial_knowledge (list or array): Initial knowledge of cost expectations.
    """
    def __init__(self, params, initial_knowledge):
        params[kc.GREEDY] = 0
        super().__init__(params, initial_knowledge)

class GawronModel(GeneralModel):
    """
    The Gawron learning model. This model is based on: `Gawron (1998) <https://kups.ub.uni-koeln.de/9257/>`_\n
    In summary, it iteratively shifts the cost expectations towards the received reward.\n
    For decision-making, calculates action utilities based on the ``beta`` parameter and cost expectations, and selects the action with the lowest utility.
    
    Args:
        params (dict): A dictionary containing model parameters.
        initial_knowledge (list or array): Initial knowledge of cost expectations.
    """
    # classic 0.8 - 0.2 exponential smoothing/markov/gawron
    def __init__(self, params, initial_knowledge):
        params[kc.REMEMBER] = 1
        params[kc.ALPHA_ZERO] = 0.2
        params[kc.ALPHAS] = [0.8]
        params[kc.GREEDY] = 1
        super().__init__(params, initial_knowledge)

class WeightedModel(GeneralModel):
    """
    Weighted Average learning model. Theory based on: `Cascetta (2009) <https://link.springer.com/book/10.1007/978-0-387-75857-2/>`_.\n
    In summary, the model uses the reward and a 5-day weighted average of the past cost expectations with decreasing weights to update the current cost expectation.\n
    For decision-making, calculates action utilities based on the ``beta`` parameter and cost expectations, and selects the action with the lowest utility.
    

    Args:
        params (dict): A dictionary containing model parameters.
        initial_knowledge (list or array): Initial knowledge of cost expectations.
    """
    # 
    def __init__(self, params, initial_knowledge):
        params[kc.REMEMBER] = 5
        params[kc.ALPHA_ZERO] = 0
        params[kc.ALPHAS] = [1/(n+2)/1.45 for n in range(5)]
        params[kc.GREEDY] = 1
        super().__init__(params, initial_knowledge)


class RandomModel(GeneralModel):
    """
    Purely random choice decision model to be used as a baseline. It randomly selects an action from the action space.\n

    Args:
        params (dict): A dictionary containing model parameters.
        initial_knowledge (list or array): Initial knowledge of cost expectations.
    """

    def __init__(self, params, initial_knowledge):
        params[kc.GREEDY] = 0
        params[kc.REMEMBER] = 0
        params[kc.NOISE_WEIGHT_DAY] = 0
        params[kc.NOISE_WEIGHT_PATH] = 0
        params[kc.NOISE_WEIGHT_AGENT] = 0
        super().__init__(params, initial_knowledge)


class AONModel(GeneralModel):
    """
    Activate on nodes decision model. This model always selects the path with the lowest initial cost,
    meaning it behaves as a shortest-path router based on the given
    initial knowledge.

    Args:
        params (dict): A dictionary containing model parameters.
        initial_knowledge (list or array): Initial knowledge of cost expectations.
    """

    def __init__(self, params, initial_knowledge):
        params[kc.GREEDY] = 1
        params[kc.REMEMBER] = 0
        params[kc.NOISE_WEIGHT_DAY] = 0
        params[kc.NOISE_WEIGHT_PATH] = 0
        params[kc.NOISE_WEIGHT_AGENT] = 0
        params[kc.BETA] = -1
        params[kc.BETA_K_I_VARIABILITY] = 0
        super().__init__(params, initial_knowledge)
