# import numpy as np
# import matplotlib.pyplot as plt

# This is a sample Python script.
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import random
import matplotlib.pyplot as plt

class Action:
    def __init__(self, id):
        self.id = id
        self.mean = 0
        self.N = 0

    # return chosen action
    def choose_action(self):
        return np.random.randn() + self.id

    # update the action-value estimate
    def update(self, x):
        self.N += 1
        # action-value function
        self.mean = (1 - 1.0 / self.N) * self.mean + 1.0 / self.N * x

# Plot trend of rewards over iterations(N)
def plot_avg(rewards, title):

    # plot moving average
    plt.plot(rewards)

    # for c in range(1,k+1):
    #     plt.plot(np.ones(N) * c)

    plt.xscale('log')
    plt.ylabel('Reward value')
    plt.xlabel('Iterations')
    plt.title(title)
    plt.show()

    # for b in actions:
    #     print(b.mean)

def e_greedy(k, eps, N):

    # initialize actions
    actions = []
    for a in range(1, k+1):
        actions.append(Action(a))

    # for a in range(k):
    #     print(actions[a].id)

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
    plot_avg(avg, "Epsilon-greedy Method")

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
    plot_avg(avg, "Greedy Method")

def optimistic_initial_values(k, N, start):

    # initialize actions
    actions = []
    for a in range(1, k + 1):
        actions.append(Action(a))

    print('idk yet')
    # # keep track of chosen action to calculate total reward
    # chosen_actions = np.empty(N)
    #
    # for i in range(N):
    #
    #     j = np.random.choice(k)
    #
    #     if
    #
    #         x = actions[j].choose_action()
    #         actions[j].update(x)
    #         chosen_actions[i] = x

        #print(j)
    #     if i in random_list:
    #         # explore
    #         j = np.random.choice(k)
    #     else:
    #         # exploit
    #         j = np.argmax([a.mean for a in actions])
    #
    #     x = actions[j].choose_action()
    #     actions[j].update(x)
    #     chosen_actions[i] = x
    #
    # avg = np.cumsum(chosen_actions) / (np.arange(N) + 1)
    # plot_avg(avg, "Optimistic Initial Values Method")

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
        optimistic_initial_values(3, 100, 2)

    #e_greedy(4, 0.01, 1000)

    # dispatch = {'1': greedy}
    #
    # m = input('Option: - ')
    #
    # dispatch[m]()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
