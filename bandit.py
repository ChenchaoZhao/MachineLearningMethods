import numpy as np

class NArmedBandit(object):
    def __init__(self, n, reward_distribution_parameters_, decision='greedy', reward_distribution='normal', eps=None, beta=None, optimistic_init = 0):
        if isinstance(n, int) and n > 1:
            self.n_arms = n
        else:
            raise ValueError('incorrect n_arms')
            
        self.decision = decision
        if decision == 'eps-greedy':
            if eps and eps > 0 and eps < 1:
                self.eps = eps
            else:
                raise ValueError('incorrect eps value')
        elif decision == 'softmax':
            if beta:
                self.beta = beta # inverse temperature
            else:
                self.beta = 1.0
        
        self.reward_distribution = reward_distribution
        
        if self.reward_distribution == 'normal':
            self.reward_value_ = reward_distribution_parameters_[0]
            self.reward_variance_ = reward_distribution_parameters_[1]
        else:
            raise ValueError('Unknown distribution {}'.format(self.reward_distribution))
            
        self.total_rewards_ = dict().fromkeys(range(self.n_arms), optimistic_init) # per action
        self.plays_per_action = dict().fromkeys(range(self.n_arms), 0)
        self.clock = 0
        self.reward_history_ = []
        self.action_history_ = []
        
    def __reward(self, action):
        if self.reward_distribution == 'normal':
            reward = np.random.normal(self.reward_value_[action], scale=np.sqrt(self.reward_variance_[action]))
            self.reward_history_.append(reward)
            return reward
    def Q_t_(self, action = None):
        if action:
            if action in range(self.n_arms):
                return self.total_rewards_[a]/self.plays_per_action[a] if self.plays_per_action[a] > 0 else 0
            else:
                raise ValueError('action is not in range')
        else:
            mean_ = []
            for a in range(self.n_arms):
                new = self.total_rewards_[a]/self.plays_per_action[a] if self.plays_per_action[a] > 0 else 0
                mean_.append(new)
            return mean_
    def softmax_(self, action = None):
        Z_ = np.exp(np.array(self.Q_t_())*self.beta)
        if action:
            if action in range(self.n_arms):
                return Z_[action]/Z_.sum()
            else:
                raise ValueError('action is not in range')
        else:
            return Z_/Z_.sum()
    
    def __random_play(self):
        action = np.random.choice(range(self.n_arms)) # uniformly select an action
        self.action_history_.append(action)
        self.clock += 1
        reward = self.__reward(action)
        self.total_rewards_[action] += reward
        self.plays_per_action[action] += 1
    
    def __greedy_play(self):
        mean_empirical_reward_ = np.mean(self.Q_t_())
        max_empirical_reward_ = np.max(self.Q_t_())
        if max_empirical_reward_ > mean_empirical_reward_:
            action = np.argmax(self.Q_t_())
        else:
            action = np.random.choice(range(self.n_arms)) # uniformly select an action
            
        self.action_history_.append(action)
        self.clock += 1
        reward = self.__reward(action)
        self.total_rewards_[action] += reward
        self.plays_per_action[action] += 1
        
    def __softmax_play(self):
        action = np.random.choice(range(self.n_arms), p=self.softmax_())
        self.action_history_.append(action)
        self.clock += 1
        reward = self.__reward(action)
        self.total_rewards_[action] += reward
        self.plays_per_action[action] += 1
    
    def play(self):
        if self.decision == 'greedy':
            self.__greedy_play()
        elif self.decision == 'eps-greedy':
            isGreedy = np.random.choice([True, False], p=[1-self.eps, self.eps])
            if isGreedy:
                self.__greedy_play()
            else:
                self.__random_play()
        elif self.decision == 'softmax':
            self.__softmax_play()
                
    def __len__(self):
        return self.clock
    
    def __str__(self):
        return '{} {}-armed bandit'.format(self.decision, self.n_arms)