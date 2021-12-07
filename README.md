# Reinforcement Learning Assignment 1

**Sofia Teeriaho(s3887626), Mohamed Gamil(s3897605)**

The file gaussian.py represents our solution for the Gaussian bandit. 

The file bernoulli.py is for running multiple armed bandit for the Bernoulli distribution.

When running the files, both produce the average reward value per time step plot and also print the percentages the best 
action (per experiment) is chosen given each method.

One thing to point out is that, we ran into an error concerning floating point numbers which results in the plot producing
one or more single horizontal lines. For reference, the plots provided in our report are the correct ones. If the plot doesn't look
similar to the ones in the report then we recommend running the code a second or third time.

In addition, for the bernoulli distribution the average rewards (avg_rewards) exceed a reward value of 1 in the y-axis of the plot 
if this line `print(avg_rewards)` is not included in front of `plt.plot(avg_rewards, label=algorithm_names[idx])`. 
