import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
theta = np.array([-1, 1])
N = 20


# Task A

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


# Parameter pairs.
values = np.linspace(-3, 3, 200)
theta_1, theta_2 = np.array([(a, b) for a in values for b in values]).T

# Posterior PDF
difference = np.stack([theta_1, theta_2], axis=-1) - posterior_mean
exponent = -0.5 * np.sum((difference @ np.linalg.inv(posterior_covariance)) * difference, axis=-1)
density = np.exp(exponent) / (2 * np.pi * np.sqrt(np.linalg.det(posterior_covariance)))


name = "posterior_density"
# Plot.
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot_trisurf(theta_1, theta_2, density, cmap="viridis")
ax.set_xlabel(r"$\theta_1$")
ax.set_ylabel(r"$\theta_2$")
ax.set_zlabel("Posterior density")
plt.savefig(f"{name}.png")

print("")
print("Task B - Posterior distribution over the parameters:")
print("Posterior mean:")
print(posterior_mean)
print("Posterior covariance:")
print(posterior_covariance)
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

# Posterior predictive variance.
posterior_predictive_variances_plot = label_variance + np.sum(
    (X_grid @ posterior_covariance) * X_grid, axis=-1
)

name = "posterior_predictive_variance"
# Plot.
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot_trisurf(x_1, x_2, posterior_predictive_variances_plot, cmap="viridis")
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_zlabel("Posterior predictive variance")
plt.savefig(f"{name}.png")

print(f"Posterior predictive variance plot saved as '{name}.png'.")


# TODO: Task D
# Task D 