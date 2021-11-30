# Reinforcement Learning - Assignment 1

import numpy as np
import random
import matplotlib.pyplot as plt

class Action:
    def __init__(self, id):
        self.id = id
        # current reward return average
        self.mean = 0
        # number of trials
        self.N = 0

    # initial reward average
    def start_reward(self, reward):
        self.mean = reward

    # return chosen action
    def choose_action(self):
        return np.random.randn() + self.id

    # update the action-value estimate
    def update(self, x):
        self.N += 1
        # action-value function
        self.mean = (1 - 1.0 / self.N) * self.mean + 1.0 / self.N * x

    def update_upper_bound(self):
        i = math.sqrt(3 / 2 * math.log(n + 1) / numbers_of_selections[i])
        upper_bound = average_reward + i
        return upper_bound

# Plot chosen rewards and avg trend of rewards over iterations(N)
def plot_avg(rewards, averages, title):

    plt.hist(rewards)
    plt.title('Histogram of actions selections')
    plt.xlabel('Actions')
    plt.ylabel('Number of times selected')
    plt.show()

    # plot moving average
    plt.plot(averages)

    # for c in range(1,k+1):
    #     plt.plot(np.ones(N) * c)

    plt.xscale('log')
    plt.ylabel('Reward value')
    plt.xlabel('Iterations')
    plt.title(title)
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

    # keep track of chosen action to calculate total reward
    chosen_actions = np.empty(N)

    for i in range(N):

        if i in random_list:
            # explore
            j = np.random.choice(k)
        else:
            # exploit
            j = np.argmax([a.mean for a in actions])

        x = actions[j].choose_action()
        actions[j].update(x)
        chosen_actions[i] = x

    avg = np.cumsum(chosen_actions) / (np.arange(N) + 1)
    plot_avg(chosen_actions, avg, "Epsilon-greedy Method")

def greedy(k, N):

    # initialize actions
    actions = []
    for a in range(1, k + 1):
        actions.append(Action(a))

    # keep track of chosen action to calculate total reward
    chosen_actions = np.empty(N)

    for i in range(N):
        # only exploit
        j = np.argmax([a.mean for a in actions])

        x = actions[j].choose_action()
        actions[j].update(x)
        chosen_actions[i] = x

    avg = np.cumsum(chosen_actions) / (np.arange(N) + 1)
    plot_avg(chosen_actions, avg, "Greedy Method")

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

    # keep track of chosen action to calculate total reward
    chosen_actions = np.empty(N)

    for i in range(N):

        if i in random_list:
            # explore
            j = np.random.choice(k)
        else:
            # exploit
            j = np.argmax([a.mean for a in actions])

        x = actions[j].choose_action()
        actions[j].update(x)
        chosen_actions[i] = x

    avg = np.cumsum(chosen_actions) / (np.arange(N) + 1)
    plot_avg(chosen_actions, avg, "Optimistic Initial Values Method")

def upper_conf_bound(eps, conf, k, N):

    # initialize actions
    actions = []
    for a in range(1, k + 1):
        actions.append(Action(a))

    # choose a percentage of indexes given epsilon
    random_list = []
    nr = int(eps * float(N))
    for i in range(nr):
        random_list.append(random.randint(0, N))

    # keep track of chosen action to calculate total reward
    chosen_actions = np.empty(N)

    # implement rest
    print("Not implemented yet")

def print_methods():
    print(f'Choose a method: \n 1 - Greedy \n 2 - Epsilon-greedy '
          f'\n 3 - Optimistic initial values \n 4 - Softmax policy '
          f'\n 5 - Upper-Confidence Bound \n 6 - Action Preferences')  # Press ⌘F8 to toggle the breakpoint.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    print_methods()

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
        #Softmax policy

    elif m == 5:
        print('Please specify the following parameters')
        eps = float(input('epsilon: - '))
        conf = int(input('confidence bound: - '))
        k = int(input('k: - '))
        N = int(input('N: - '))
        upper_conf_bound(eps, conf, k, N)

    elif m == 6:
        print('Not implemented')
        #Action preferences