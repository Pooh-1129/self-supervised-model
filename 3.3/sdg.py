import torch
import numpy as np
import matplotlib.pyplot as plt

# --- Function Definitions ---
def func_min(x, y):
    """Function to be minimized: f(x, y) = x^2 + y^2"""
    return x**2 + y**2

def func_max(x, y):
    """Function to be maximized: f(x, y) = -x^2 - y^2"""
    return -x**2 - y**2

# --- Helper function to run optimization ---
def run_optimization(start_pos, func, lr=0.1, num_steps=30, momentum=0.0, weight_decay=0.0, maximize=False):
    """
    Runs SGD optimization and returns the trajectory of (x, y).
    """
    params = torch.tensor(start_pos, requires_grad=True, dtype=torch.float32)
    optimizer = torch.optim.SGD([params], lr=lr, momentum=momentum, weight_decay=weight_decay, maximize=maximize)
    path = [params.clone().detach().numpy()]

    for _ in range(num_steps):
        optimizer.zero_grad()
        x, y = params[0], params[1]
        loss = func(x, y)
        loss.backward()
        optimizer.step()
        path.append(params.clone().detach().numpy())

    return np.array(path)

# --- NEW: Plotting function for a grid of trajectories ---
def plot_momentum_grid(paths, labels, title, func, filename):
    """
    Generates a 2x2 grid of contour plots, each with a single trajectory.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(title, fontsize=16)

    x_grid = np.linspace(-3.5, 3.5, 100)
    y_grid = np.linspace(-3.5, 3.5, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = func(X, Y)

    for i, ax in enumerate(axes.flat):
        path = paths[i]
        label = labels[i]

        # Draw contour plot on the subplot
        cp = ax.contour(X, Y, Z, levels=15, cmap='viridis')
        ax.clabel(cp, inline=True, fontsize=8)

        # Plot the single trajectory
        ax.plot(path[:, 0], path[:, 1], 'o-', label=label, markersize=3, linewidth=1.5, color='r')

        ax.set_title(label)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.axis('equal')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle

    # Save the entire figure
    plt.savefig(filename)
    print(f"Plot saved as {filename}")

    plt.show()


# --- Plotting function for a single plot (for part c) ---
def plot_single_trajectory(path, label, title, func, filename):
    """
    Generates and saves a single contour plot with one trajectory.
    """
    plt.figure(figsize=(8, 8))
    x_grid = np.linspace(-3.5, 3.5, 100)
    y_grid = np.linspace(-3.5, 3.5, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = func(X, Y)
    cp = plt.contour(X, Y, Z, levels=15, cmap='viridis')
    plt.clabel(cp, inline=True, fontsize=8)

    plt.plot(path[:, 0], path[:, 1], 'o-', label=label, markersize=3, linewidth=1.5)

    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal')
    plt.savefig(filename)
    print(f"Plot saved as {filename}")
    plt.show()
    

# --- Main Execution ---

start_point = (3.0, 3.0)

# (a) Vary Momentum between 0 and 0.9 (on a grid)
print("\n--- Running experiment (a): Varying momentum ---")
momenta = [0.0, 0.3, 0.6, 0.9]
paths_a = [run_optimization(start_point, func_min, momentum=m) for m in momenta]
labels_a = [f'Momentum = {m}' for m in momenta]
plot_momentum_grid(paths_a, labels_a, 'Effect of Varying Momentum', func_min, 'momentum_effects_grid.png')

# (b) Vary Momentum with Weight Decay (on a grid)
print("\n--- Running experiment (b): Varying momentum with weight decay ---")
weight_decay_val = 0.1
paths_b = [run_optimization(start_point, func_min, momentum=m, weight_decay=weight_decay_val) for m in momenta]
labels_b = [f'Momentum = {m}' for m in momenta]
plot_momentum_grid(paths_b, labels_b, f'Effect of Momentum with Weight Decay = {weight_decay_val}', func_min, 'weight_decay_effects_grid.png')

# (c) Maximize the function (single plot)
print("\n--- Running experiment (c): Maximization ---")
path_c = run_optimization(start_point, func_max, momentum=0.5, maximize=True)
plot_single_trajectory(path_c, 'Maximize = True', 'Maximizing f(x, y) = -x² - y²', func_max, 'maximization_effect.png')