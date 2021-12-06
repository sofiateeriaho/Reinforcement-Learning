import numpy as np
import random
import matplotlib.pyplot as plt
import math

class Action:
    def __init__(self, id, T):
        self.id = id
        # current reward return average
        self.mean = 0
        # number of trials
        self.T = 0
        # probability of success (for Bernoulli)
        self.p = 0
        self.rewards = [0] * T

    # initial reward average
    def start_reward(self, reward):
        self.mean = reward

    # return reward from bernoulli distribution
    def choose_action(self):
        reward = self.rewards[self.T]
        return reward

    # update the action-value estimate
    def update(self, x):
        self.T += 1
        # action-value function
        self.mean = (1 - 1.0 / self.T) * self.mean + 1.0 / self.T * x


def generate_rewards(K, T):
    # probability of winning for each bandit
    p = np.random.rand(K)
    rewards = np.random.rand(T, K) < np.tile(p, (T, 1))

    return p, rewards

def initialize_actions(k, T):
    actions = []
    p, rewards = generate_rewards(k, T)

    # initialize set of actions by determining their probability and rewards for each time step
    for a in range(1, k + 1):
        action_rewards = np.empty(T)
        actions.append(Action(a, T))
        actions[a-1].p = p[a-1]
        for j in range(T):
            action_rewards[j] = rewards[j][a-1]

        actions[a-1].rewards = action_rewards

    return actions

def e_greedy(k, eps, T):

    actions = initialize_actions(k, T)

    # choose a percentage of indexes given epsilon
    random_list = []
    nr = int(eps * float(T))
    for i in range(nr):
        random_list.append(random.randint(0, T))

    # keep track of chosen actions and their rewards
    chosen_actions = []
    chosen_rewards = np.empty(T)
    total = 0

    for i in range(T):

        if i in random_list:
            # explore
            j = np.random.choice(k)
        else:
            # exploit
            j = np.argmax([a.mean for a in actions])

        x = actions[j].choose_action()
        actions[j].update(x)
        chosen_rewards[i] = x
        chosen_actions.append(actions[j].id)
        total += x

    avg = np.cumsum(chosen_rewards) / (np.arange(T) + 1)

    return chosen_rewards

def optimistic_initial_values(eps, start, k, T):

    # initialize actions
    actions = initialize_actions(k, T)

    for a in range(len(actions)):
        # set initial reward
        actions[a].start_reward(start)

    # choose a percentage of indexes given epsilon
    random_list = []
    nr = int(eps * float(T))
    for i in range(nr):
        random_list.append(random.randint(0, T))

    # keep track of chosen actions and their rewards
    chosen_actions = []
    chosen_rewards = np.empty(T)

    for i in range(T):

        if i in random_list:
            # explore
            j = np.random.choice(k)
        else:
            # exploit
            j = np.argmax([a.mean for a in actions])

        x = actions[j].choose_action()
        actions[j].update(x)
        chosen_rewards[i] = x
        chosen_actions.append(actions[j].id)

    return chosen_rewards

def upper_conf_bound(k, T):
    # initialize actions
    actions = initialize_actions(k, T)

    # keep track of chosen actions and their rewards
    chosen_actions = []
    chosen_rewards = np.empty(T)
    sums_of_reward = [0] * k

    for n in range(0, T):
        arm = 0
        max_upper_bound = 0
        for i in range(0, k):
            if (actions[i].T > 0):
                average_reward = sums_of_reward[i] / actions[i].T
                c = math.sqrt(2 * math.log(n + 1) / actions[i].T)
                upper_bound = average_reward + c
            else:
                # change this
                upper_bound = 1e400

            if upper_bound > max_upper_bound:
                max_upper_bound = upper_bound
                arm = i

        reward = actions[arm].choose_action()
        actions[arm].update(reward)
        chosen_rewards[n] = reward
        sums_of_reward[arm] += reward

        chosen_actions.append(actions[arm].id)

    return chosen_rewards

# return sum of array values at each index
def sum_arrays(arr1, arr2, T):

    total = np.empty(T)

    for i in range(T):
        total[i] = arr1[i] + arr2[i]

    return total

# function to run N amount of experiments with time step T and total actions k
def run_experiments(N, T, k, method):

    # specify parameters
    eps = 0.1
    initial_reward = 1

    # store sum of rewards at each time step T
    total_rewards = np.empty(T)
    # store average rewards at each time step T
    avg_rewards = np.empty(T)

    for _ in range(N):
        chosen_rewards = np.empty(T)
        if method == 1:
            chosen_rewards = e_greedy(k, 0, T)
        elif method == 2:
            chosen_rewards = e_greedy(k, eps, T)
        elif method == 3:
            chosen_rewards = optimistic_initial_values(eps, initial_reward, k, T)
        elif method == 4:
            chosen_rewards = upper_conf_bound(k, T)

        # sum up rewards at each time step
        total_rewards = sum_arrays(total_rewards, chosen_rewards, T)

    # take average of rewards at each time step
    for i in range(T):
        avg_rewards[i] = total_rewards[i] / N

    return avg_rewards

if __name__ == '__main__':

    # actions/arms
    k = 5
    # time steps per experiment
    T = 100
    # number of experiments
    N = 300

    #algorithms = [greedy, e_greedy, optimistic_initial_values, upper_conf_bound]

    # ctr, rewards = bernoulli(k, T)
    # print(ctr)
    # print(rewards)

    # store avg rewards after running experiment N times
    avg_rewards = np.empty(T)

    for i in range(1, 5):
        avg_rewards = run_experiments(N, T, k, i)
        if i == 1:
            plt.plot(avg_rewards, label="method = greedy")
            avg_rewards = np.empty(T)
        elif i == 2:
            plt.plot(avg_rewards, label="method = epsilon-greedy")
            avg_rewards = np.empty(T)
        elif i == 3:
            #print(avg_rewards)
            plt.plot(avg_rewards, label="method = optimistic initial values")
            avg_rewards = np.empty(T)
        elif i == 4:
            #print(avg_rewards)
            plt.plot(avg_rewards, label="method = UCB")
            avg_rewards = np.empty(T)

    plt.title('Average reward value per time step (Bernoulli)')
    plt.ylabel('Average reward value')
    plt.xlabel('Time steps')

    plt.legend()
    axes = plt.gca()
    axes.yaxis.grid()
    plt.show()