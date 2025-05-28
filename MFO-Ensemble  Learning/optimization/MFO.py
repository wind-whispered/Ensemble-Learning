import numpy as np


def optimize_weights_mfo(objective_func, dim, frozen_indices=None, frozen_values=None,
                         population_size=50, max_iter=200, flame_min=5, b_max=1.5, b_min=0.1):
    moths = np.random.rand(population_size, dim)

    if frozen_indices is not None:
        for i in range(population_size):
            moths[i][frozen_indices] = frozen_values

    best_fitness = np.inf
    best_solution = None

    for t in range(max_iter):
        for i in range(population_size):
            if frozen_indices is not None:
                moths[i][frozen_indices] = frozen_values

        moths = np.clip(moths, 0, 1)
        for i in range(population_size):
            if np.sum(moths[i]) == 0:
                moths[i] = np.ones(dim) / dim

        fitness = np.array([objective_func(m) for m in moths])
        sorted_indices = np.argsort(fitness)
        flame_no = max(round(population_size - t * ((population_size - 1) / max_iter)), flame_min)
        flames = moths[sorted_indices[:flame_no]]

        b_t = b_max - (t / max_iter) * (b_max - b_min)
        alpha_t = 1 - (t / max_iter)

        for i in range(population_size):
            flame_idx = i % flame_no
            distance = np.abs(flames[flame_idx] - moths[i])
            t_param = np.random.uniform(-alpha_t, alpha_t)
            moths[i] = distance * np.exp(b_t * t_param) * np.cos(2 * np.pi * t_param) + flames[flame_idx]

        current_best_fitness = np.min(fitness)
        current_best_solution = moths[np.argmin(fitness)]
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = current_best_solution.copy()

    best_solution = np.clip(best_solution, 0, 1)
    if frozen_indices is not None:
        best_solution[frozen_indices] = frozen_values
    return best_solution / np.sum(best_solution)
