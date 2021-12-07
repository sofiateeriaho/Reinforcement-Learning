import numpy as np
import random
import matplotlib.pyplot as plt
import math

class Action:
    def __init__(self, id, value):
        self.id = id
        # true reward average
        self.true = value
        # current reward return average
        self.estimate = 0
        # number of trials
        self.T = 0
        # standard devation
        self.std = 1

    # initial reward average
    def start_reward(self, reward):
        self.estimate = reward

    # return reward from standard normal distribution
    def choose_action(self):
        reward = np.random.normal(self.true, self.std)
        return np.round(reward, 1)

    # update the action-value estimate
    def update(self, x):
        self.T += 1
        self.estimate = (1 - 1.0 / self.T) * self.estimate + 1.0 / self.T * x

def initialize_actions(k):
    actions = []
    for a in range(1, k + 1):
        actions.append(Action(a, np.random.randn()))
    return actions

def best_actions(actions):
    max = 0
    best = 0
    for i in range(len(actions)):
        if actions[i].true > max:
            max = actions[i].true
            best = actions[i].id
    return best

def get_percentage(actions, chosen_actions):
    best = best_actions(actions)
    sum = 0
    for i in range(T):
        if chosen_actions[i] == best:
            sum += 1
    percentage = (sum * 100) / T
    return percentage

def greedy(k, T, *args):
    # initialize actions
    actions = initialize_actions(k)

    # keep track of chosen actions and their rewards
    chosen_actions = []
    chosen_rewards = np.empty(T)

    for i in range(T):
        # only exploit
        j = np.argmax([a.estimate for a in actions])

        x = actions[j].choose_action()
        actions[j].update(x)
        chosen_rewards[i] = x
        chosen_actions.append(actions[j].id)

    percentage = get_percentage(actions, chosen_actions)

    return chosen_rewards, percentage

def e_greedy(k, T, eps, *args):

    actions = initialize_actions(k)

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
            j = np.argmax([a.estimate for a in actions])

        x = actions[j].choose_action()
        actions[j].update(x)
        chosen_rewards[i] = x
        chosen_actions.append(actions[j].id)

    percentage = get_percentage(actions, chosen_actions)

    return chosen_rewards, percentage

def optimistic_initial_values(k, T, eps, start, *args):

    # initialize actions
    actions = initialize_actions(k)

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
            j = np.argmax([a.estimate for a in actions])

        x = actions[j].choose_action()
        actions[j].update(x)
        chosen_rewards[i] = x
        chosen_actions.append(actions[j].id)

    percentage = get_percentage(actions, chosen_actions)

    return chosen_rewards, percentage

# referenced from http://ethen8181.github.io/machine-learning/bandits/multi_armed_bandits.html
def softmax(k, T, *args):

    # initialize actions
    actions = initialize_actions(k)

    chosen_actions = []
    chosen_rewards = np.empty(T)

    # annealing (adaptive learning rate)
    tau = 1 / np.log(T + 0.000001)

    probs_n = np.exp([a.estimate for a in actions] / tau)
    probs_d = probs_n.sum()
    probs = probs_n / probs_d

    cum_prob = 0.
    z = np.random.rand()

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

    percentage = get_percentage(actions, chosen_actions)

    return chosen_rewards, percentage

def upper_conf_bound(k, T, *args):
    # initialize actions
    actions = initialize_actions(k)

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
                upper_bound = 1e100

            if upper_bound > max_upper_bound:
                max_upper_bound = upper_bound
                arm = i

        reward = actions[arm].choose_action()
        actions[arm].update(reward)
        chosen_rewards[n] = reward
        sums_of_reward[arm] += reward

        chosen_actions.append(actions[arm].id)

        ['{:.4f}'.format(u) for u in chosen_rewards]

    percentage = get_percentage(actions, chosen_actions)

    return chosen_rewards, percentage

# return sum of array values at each index
def sum_arrays(arr1, arr2, T):

    total = np.empty(T)

    for i in range(T):
        total[i] = arr1[i] + arr2[i]

    return total

# function to run N amount of experiments with time step T and total actions k
def run_experiments(N, T, k, algorithm):

    # specify parameters
    eps = 0.1
    initial_reward = 3

    # store sum of rewards at each time step T
    total_rewards = np.empty(T)
    # store average rewards at each time step T
    average_rewards = np.empty(T)
    total_percentage = 0

    # run N amount of experiments for each algorithm
    for y in range(N):
        chosen_rewards, percentage = algorithm(k, T, eps, initial_reward)
        # sum up all percentages for given method
        total_percentage += percentage
        # sum up rewards at each time step
        total_rewards = sum_arrays(total_rewards, chosen_rewards, T)

    # take average of rewards at each time step
    for i in range(T):
        average_rewards[i] = (total_rewards[i] / N)

    avg_percentage = total_percentage/N

    return average_rewards, avg_percentage

def plot_averages(N, T, k):

    algorithms = [greedy, e_greedy, softmax, optimistic_initial_values, upper_conf_bound]
    algorithm_names = ["greedy", "epsilon-greedy", "softmax", "optimistic initial values", "UCB"]

    # run experiments of each algorithm
    for idx, algo in enumerate(algorithms):
        avg_rewards, percentages = run_experiments(N, T, k, algo)

        ['{:.4f}'.format(i) for i in avg_rewards]

        print(algorithm_names[idx], "=", np.round(percentages, 3))

        plt.plot(avg_rewards, label=algorithm_names[idx])

    plt.title('Average reward value per time step (Gaussian)')
    plt.ylabel('Average reward value')
    plt.xlabel('Time steps')
    plt.legend()
    axes = plt.gca()
    axes.yaxis.grid()
    plt.show()

if __name__ == '__main__':

    # actions/arms
    k = 10
    # time steps per experiment
    T = 100
    # number of experiments
    N = 300

    plot_averages(N, T, k)