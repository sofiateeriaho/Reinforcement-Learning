# Reinforcement Learning - Assignment 1

import numpy as np
import random
import matplotlib.pyplot as plt
import math

class Action:
    def __init__(self, id):
        self.id = id
        # current reward return average
        self.mean = 0
        # number of trials
        self.N = 0
        # probability of success
        self.p = 0
        self.std = 1

    # initial reward average
    def start_reward(self, reward):
        self.mean = reward

    # return reward from standard normal distribution
    # return chosen action
    def choose_action(self):
        ## this one has more control over variance and std
        reward = np.random.normal(self.mean, self.std)
        return np.round(reward, 1)

        # randn gives a distribution from some standardized normal distribution (mean 0 and variance 1)
        # reward = np.random.randn() + self.id
        # return reward

    def choose_action_bernoulli(self):
        if random.random() > self.mean:
            return 0.0
        else:
            return 1.0
        # return np.random.binomial(1, self.p)
        #return np.random.binomial(1, self.p, size=1)[0]

    # update the action-value estimate
    def update(self, x):
        self.N += 1
        # action-value function
        self.mean = (1 - 1.0 / self.N) * self.mean + 1.0 / self.N * x

    def update_upper_bound(self):
        i = math.sqrt(3 / 2 * math.log(n + 1) / numbers_of_selections[i])
        upper_bound = average_reward + i
        return upper_bound

# Plot chosen actions and average trend of rewards over iterations(N)
def plot_avg(actions, chosen_rewards, title, k, N):

    avg = np.cumsum(chosen_rewards) / (np.arange(N) + 1)

    # plot moving average
    plt.plot(avg)
    plt.title(title)
    plt.ylabel('Average reward value')
    plt.xlabel('Iterations')
    axes = plt.gca()
    axes.yaxis.grid()
    plt.show()

    # plot histogram of chosen actions
    bins = np.arange(k + 2) - 0.5
    plt.hist(actions, bins, edgecolor='black')
    plt.xticks(range(k + 1))
    plt.xlim([0, k + 1])

    plt.title('Histogram of actions selections')
    plt.xlabel('Actions')
    plt.ylabel('Number of times selected')
    axes = plt.gca()
    axes.yaxis.grid()
    plt.show()

def e_greedy(k, eps, N):

    # initialize actions
    actions = []
    for a in range(1, k+1):
        actions.append(Action(a))

    # choose a percentage of indexes given epsilon
    random_list = []
    nr = int(eps * float(N))
    for i in range(nr):
        random_list.append(random.randint(0, N))

    # keep track of chosen actions and their rewards
    chosen_actions = []
    chosen_rewards = np.empty(N)

    for i in range(N):

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

    plot_avg(chosen_actions, chosen_rewards, "Epsilon-greedy Method", k, N)

def greedy(k, N):

    # initialize actions
    actions = []
    for a in range(1, k + 1):
        actions.append(Action(a))

    # keep track of chosen actions and their rewards
    chosen_actions = []
    chosen_rewards = np.empty(N)

    for i in range(N):
        # only exploit
        j = np.argmax([a.mean for a in actions])

        x = actions[j].choose_action()
        actions[j].update(x)
        chosen_rewards[i] = x
        chosen_actions.append(actions[j].id)

    plot_avg(chosen_actions, chosen_rewards, "Greedy Method", k, N)

def optimistic_initial_values(eps, start, k, N):

    # initialize actions
    actions = []
    for a in range(1, k + 1):
        actions.append(Action(a))
        # set initial reward
        Action(a).start_reward(start)

    # choose a percentage of indexes given epsilon
    random_list = []
    nr = int(eps * float(N))
    for i in range(nr):
        random_list.append(random.randint(0, N))

    # keep track of chosen actions and their rewards
    chosen_actions = []
    chosen_rewards = np.empty(N)

    for i in range(N):

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

    plot_avg(chosen_actions, chosen_rewards, "Optimistic Initial Values Method", k, N)

def upper_conf_bound(k, N):

    # initialize actions
    actions = []
    for a in range(1, k + 1):
        actions.append(Action(a))

    # keep track of chosen actions and their rewards
    chosen_actions = []
    chosen_rewards = np.empty(N)
    sums_of_reward = [0] * k
    total_reward = 0

    for n in range(0, N):
        arm = 0
        max_upper_bound = 0
        for i in range(0, k):
            if (actions[i].N > 0):
                average_reward = sums_of_reward[i] / actions[i].N
                c = math.sqrt(2 * math.log(n + 1) / actions[i].N)
                upper_bound = average_reward + c
            else:
                upper_bound = 1e400

            if upper_bound > max_upper_bound:
                max_upper_bound = upper_bound
                arm = i

        reward = actions[arm].choose_action()
        actions[arm].update(reward)
        chosen_rewards[n] = reward
        sums_of_reward[arm] += reward
        total_reward += reward
        chosen_actions.append(actions[arm].id)

    plot_avg(chosen_actions, chosen_rewards, "Upper-Confidence Bound Method", k, N)

def print_methods():
    print(f'Choose a method: \n 1 - Greedy \n 2 - Epsilon-greedy '
          f'\n 3 - Optimistic initial values \n 4 - Softmax policy '
          f'\n 5 - Upper-Confidence Bound \n 6 - Action Preferences')  # Press ⌘F8 to toggle the breakpoint.

def user_interact():
    # ask user for method
    m = int(input('Option: - '))

    if m == 1:
        print('Please specify the following parameters')
        k = int(input('k: - '))
        N = int(input('N: - '))
        greedy(k, N)
    elif m == 2:
        print('Please specify the following parameters')
        eps = float(input('epsilon: - '))
        k = int(input('k: - '))
        N = int(input('N: - '))
        e_greedy(k, eps, N)
    elif m == 3:
        print('Please specify the following parameters')
        eps = float(input('epsilon: - '))
        start = int(input('initial reward: - '))
        k = int(input('k: - '))
        N = int(input('N: - '))
        optimistic_initial_values(eps, start, k, N)

    elif m == 4:
        print('Not implemented')
        # Softmax policy

    elif m == 5:
        print('Please specify the following parameters')
        k = int(input('k: - '))
        N = int(input('N: - '))
        upper_conf_bound(k, N)

    elif m == 6:
        print('Not implemented')
        # Action preferences

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    print_methods()
    user_interact()

