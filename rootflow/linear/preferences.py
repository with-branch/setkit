from typing import List, Tuple

import numpy as np
import statsmodels.api as sm


def maximally_constrained_vector_products(
    initial_vector: np.ndarray,
    constraints: List[Tuple[np.ndarray, int]],
    weights: List[float]
):
    # Return a new vector, such that the maximum number of constraints are true
    # It is NOT guaranteed that all of the contraints are satisfiable.
    # It is also possible/practically certain that the maximum constraints will
    # identify a subspace instead of a point, in which case we would like to select
    # the point which is closest to the current vector from within that subspace.
    constraint_vector, constraint_output = constraints[0]
    # Solving a constraint means that the following will be true
    ((initial_vector * constraint_vector) > threshold) == constraint_output
    return initial_vector


if __name__ == "__main__":
    constraint = (np.array([0.2, 0.4, 0.6, -0.89]), 0)
    constraints = [constraint for _ in range(10)]

    current_embedding = np.array([-0.5, 1.0, 0.3, 0.7])
    next_embedding = maximally_constrained_vector_products(
        current_embedding, constraints
    )