import numpy as np
import matplotlib.pyplot as plt

# count the number of occurences of each item in the given observation.
num_apple = 6
num_banana = 4

# simulates the probability of apple using numpy, ranging from 0 to 1.
p_apple = np.arange(0, 1.00001, 0.00001)
p_banana = 1 - p_apple

# calculate the probability that the given sequence occurs (i.e. 6 apples and 4 bananas)
p_seq = (p_apple**num_apple) * (p_banana**num_banana)

# Plot the probability of the observations as a function of p_apple
plt.plot(p_apple, p_seq)
plt.title("Probability of Observation as a Function of P_Apple")
plt.xlabel("P_Apple")
plt.ylabel("P(Observations)")
plt.show()
