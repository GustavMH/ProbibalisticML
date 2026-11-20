import numpy as np
import matplotlib.pyplot as plt

# Random seed for reproducibility
np.random.seed(42)

# Function for plotting in 3D
def plot(x, x_label, y, y_label, z, z_label, name):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot_trisurf(x, y, z, cmap="viridis")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    plt.savefig(f"{name}.png")


# Task A

# Parameters
theta = np.array([-1, 1])
N = 20

# Distribution of the inputs.
input_mean = np.array([0, 0])
input_covariance = np.identity(2)
X = np.random.multivariate_normal(
    mean=input_mean,
    cov=input_covariance,
    size=N
)

# Distribution of the labels.
label_mean = X @ theta
label_variance = 0.1
y = np.random.normal(loc=label_mean, scale=np.sqrt(label_variance), size=N)

if X.shape == (N, 2) and y.shape == (N,) and np.isfinite(X).all() and np.isfinite(y).all(): # Check if the data is valid
    print("Task A")
    print("Generated data successfully.")
else:
    print("Failed to generate data.")


# Task B

# Prior distribution over theta.
prior_mean = np.array([0, 0])
prior_covariance = np.identity(2)

# Posterior distribution over theta. Formulas are in the lecture notes in section 4.2
posterior_mean = prior_covariance @ X.T @ np.linalg.inv(label_variance * np.identity(N) + X @ prior_covariance @ X.T) @ y
posterior_covariance = prior_covariance - prior_covariance @ X.T @ np.linalg.inv(label_variance * np.identity(N) + X @ prior_covariance @ X.T) @ X @ prior_covariance

# Printing the posterior mean and covariance.
print("")
print("Task B - Posterior distribution over the parameters:")
print("Posterior mean:")
print(posterior_mean)
print("Posterior covariance:")
print(posterior_covariance)

# Parameter pairs.
values = np.linspace(-3, 3, 200)
theta_1, theta_2 = np.array([(a, b) for a in values for b in values]).T

# Posterior PDF for the plot
difference = np.stack([theta_1, theta_2], axis=-1) - posterior_mean
exponent = -0.5 * np.sum((difference @ np.linalg.inv(posterior_covariance)) * difference, axis=-1)
density = np.exp(exponent) / (2 * np.pi * np.sqrt(np.linalg.det(posterior_covariance)))

# Plotting the posterior density.
name = "posterior_density"
plot(theta_1, r"$\theta_1$", theta_2, r"$\theta_2$", density, "Posterior density", name)
print(f"Posterior density plot saved as '{name}.png'.")




# Task C - Predictive postrior distribution over the labels.

# posterior predictive distribution
posterior_predictive_mean = X @ posterior_mean
posterior_predictive_variance = label_variance + np.diag(X @ posterior_covariance @ X.T) # Note we store the points as row vectors, so the transposing of X is switched

print("")
print("Task C - Predictive posterior distribution over the labels:")
print("Posterior predictive mean:")
print(posterior_predictive_mean)
print("Posterior predictive variance:")
print(posterior_predictive_variance)

# Plot the predictive posterior distribution over the labels.
# Input pairs.
values = np.linspace(-3, 3, 200)
x_1, x_2 = np.array([(a, b) for a in values for b in values]).T
X_grid = np.stack([x_1, x_2], axis=-1)
posterior_predictive_variances_plot = label_variance + np.diag(X_grid @ posterior_covariance @ X_grid.T) # Posterior predictive variances for the plot

# Plotting the posterior predictive variance.
name = "posterior_predictive_variance"
plot(x_1, r"$x_1$", x_2, r"$x_2$", posterior_predictive_variances_plot, "Posterior predictive variance", name)
print(f"Posterior predictive variance plot saved as '{name}.png'.")


# TODO: Task D
# Task D 