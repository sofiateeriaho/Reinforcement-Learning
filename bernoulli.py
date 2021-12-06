# references: https://www.geeksforgeeks.org/epsilon-greedy-algorithm-in-reinforcement-learning/
# http://ethen8181.github.io/machine-learning/bandits/multi_armed_bandits.html
# https://medium.com/analytics-vidhya/multi-armed-bandit-analysis-of-epsilon-greedy-algorithm-8057d7087423
# https://github.com/kfoofw/bandit_simulations/blob/master/python/multiarmed_bandits/analysis/ts.md
# https://www.analyticsvidhya.com/blog/2018/09/reinforcement-multi-armed-bandit-scratch-python/

import numpy as np
import random
import matplotlib.pyplot as plt
import math
from collections import Counter

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

    def set_prob(self, p):
        self.p = p

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
        actions[a-1].set_prob(p)
        for j in range(T):
            action_rewards[j] = rewards[j][a-1]

        actions[a-1].rewards = action_rewards

    return actions

def get_best(actions):
    max = 0
    chosen = -1
    for idx in range(len(actions)):
        if actions[idx].p > max:
            max = actions[idx].p
            chosen = actions[idx].id

    return chosen

def e_greedy(k, T, eps, *args):

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

    best_action = get_best(actions)

    return chosen_rewards, best_action, chosen_actions

def greedy(k, T, *args):
    rewards, best_action, chosen_actions = e_greedy(k, T, 0)
    return rewards, best_action, chosen_actions

def optimistic_initial_values(k, T, eps, start, *args):

    # initialize actions
    actions = initialize_actions(k, T)

    # set an initial reward for each action
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

    best_action = get_best(actions)

    return chosen_rewards, best_action, chosen_actions

def softmax(k, T, *args):

    # initialize actions
    actions = initialize_actions(k, T)

    chosen_actions = []
    chosen_rewards = np.empty(T)

    # annealing (adaptive learning rate)
    tau = 1 / np.log(T + 0.000001)

    probs_n = np.exp([a.mean for a in actions] / tau)
    probs_d = probs_n.sum()
    probs = probs_n / probs_d

    cum_prob = 0.
    z = np.random.rand()

    i = 0
    for i in range(T):
        for idx, prob in enumerate(probs):
            cum_prob += prob
            if cum_prob > z:
                j = idx
            #return len(probs) - 1

        x = actions[j].choose_action()
        actions[j].update(x)
        chosen_rewards[i] = x
        chosen_actions.append(actions[j].id)

    best_action = get_best(actions)

    return chosen_rewards, best_action, chosen_actions

def upper_conf_bound(k, T, *args):
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

        ['{:.4f}'.format(u) for u in chosen_rewards]

    best_action = get_best(actions)

    return chosen_rewards, best_action, chosen_actions

# return sum of array values at each index
def sum_arrays(arr1, arr2, T):

    total = np.empty(T)

    for i in range(T):
        total[i] = arr1[i] + arr2[i]

    return total

# calculate the percentage an action is taken per time step T
def action_percentage(N, T, arr, best):

    percentages = np.empty(T)

    print(best)

    for i in range(T):
        sum = 0
        for j in range(N):
            #print(arr[j][i])
            if arr[j][i] == best[j]:
                sum += 1
        p = (sum * 100)/N
        #print(sum)
        percentages[i] = p

    #print(percentages)
    return percentages

# function to run N amount of experiments with time step T and total actions k
def run_experiments(N, T, k, algorithm):

    # specify parameters
    eps = 0.1
    initial_reward = 1

    # store sum of rewards at each time step T
    total_rewards = np.empty(T)

    # store average rewards at each time step T
    average_rewards = np.empty(T)

    percentage_array = []
    best_actions = []

    # run N amount of experiments for each algorithm
    for y in range(N):
        chosen_rewards, best, chosen_actions = algorithm(k, T, eps, initial_reward)

        percentage_array.append(chosen_actions)

        best_actions.append(best)

        # sum up rewards at each time step
        total_rewards = sum_arrays(total_rewards, chosen_rewards, T)

    # take average of rewards at each time step
    for i in range(T):
        average_rewards[i] = (total_rewards[i] / N)

    all_percentages = action_percentage(N, T, percentage_array, best_actions)

    return average_rewards, all_percentages

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plot_averages(N, T, k):

    algorithms = [greedy, e_greedy, softmax, optimistic_initial_values, upper_conf_bound]
    algorithm_names = ["greedy", "epsilon-greedy", "softmax", "optimistic initial values", "UCB"]

    n_bins = T

    percentages_total = []

    for idx, algo in enumerate(algorithms):
        avg_rewards, percentages = run_experiments(N, T, k, algo)

        ['{:.4f}'.format(i) for i in avg_rewards]
        #print(avg_rewards)

        # poly = np.polyfit(range(0,T), avg_rewards, 13)
        # poly_y = np.poly1d(poly)(range(0,T))

        #print(percentages)
        #plt.hist(percentages, label=algorithm_names[idx])

        # plt.plot(poly_y, 'gray', markersize=14)
        # plt.plot(avg_rewards, label=algorithm_names[idx], alpha=0.6)
        #plt.plot(smooth(avg_rewards,5),label=algorithm_names[idx])



    # plt.title('Average reward value per time step (Bernoulli)')
    # plt.ylabel('Average reward value')
    # plt.xlabel('Time steps')
    # plt.legend()
    # axes = plt.gca()
    # axes.yaxis.grid()
    # plt.show()


    plt.hist(percentages_total, density=True, histtype='bar', label=algorithm_names)

    plt.title('Percentage of best action selection per time step')
    plt.xlabel('Time step (T)')
    plt.ylabel('Percentage of times selected')
    axes = plt.gca()
    axes.yaxis.grid()
    #plt.show()

if __name__ == '__main__':

    # actions/arms
    k = 5
    # time steps per experiment
    T = 10
    # number of experiments
    N = 100

    #print(softmax(k, T))

    plot_averages(N, T, k)
