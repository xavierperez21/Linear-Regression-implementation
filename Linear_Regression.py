import numpy as np  # numpy is useful for mathematical calculations
import scipy as sc  # Expands funcionality of numpy.

import matplotlib.pyplot as plt  # graphic visualization

# For this project we're going to use the _"Boston House prices"_ where **we will try to predict the price of a house according to the number of rooms per dwelling.** We use just one parameter to visualize it in a 2D plane and we could see our model.

from sklearn.datasets import load_boston  # Import the dataset

# We load the dataset
boston = load_boston()

print(boston.DESCR)


# The "Least Squares" formula is the following one: $\beta = (X^{T}X)^{⁻1}X^{T}Y$

# X will be the column of the average number of rooms
X = np.array(boston.data[:, 5])

# Y will be the median value of houses in each town (in 1000's)
Y = np.array(boston.target)

# Plotting the data
plt.scatter(X, Y, alpha=0.3)


# Adding column of ones for the independent term
X = np.array([np.ones(506), X]).T


# We apply the least squares formula. "@" is for matrix multiplication.
B = np.linalg.inv(X.T @ X) @ X.T @ Y
print(B)

# Plotting our model (line) using the...
# ...obtained parameters in the "B" array
plt.plot([4, 9], [B[0] + 4*B[1], B[0] + 9*B[1]], c="red")
plt.show()
