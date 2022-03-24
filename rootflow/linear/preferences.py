from typing import List, Tuple

import numpy as np
from pulp import LpVariable, LpProblem, LpMinimize, lpSum, value

MAX_VALUE_WEIGHT = 10


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
    variables = [ LpVariable((f"value_{i}"), 0, 2*MAX_VALUE_WEIGHT) for i in range(len(constraints[0][0])) ]
    helper_variables = []
    prob = LpProblem("optimize_initial_vector_using_least_squares", LpMinimize)

    prob += lpSum(weights[i]*(abs((MAX_VALUE_WEIGHT + constraints[i][1]) - lpSum((variables[x]*constraints[i][0][x]) for x in range(len(variables))))) for i in range(len(constraints)))
    status = prob.solve()

    optimized_vector = [value(variables[i]) - 10 for i in len(variables)]
    return optimized_vector


if __name__ == "__main__":
    constraint = (np.array([0.2, 0.4, 0.6, -0.89]), 0)
    constraints = [constraint for _ in range(10)]

    current_embedding = np.array([-0.5, 1.0, 0.3, 0.7])
    next_embedding = maximally_constrained_vector_products(
        current_embedding, constraints, [1 for _ in range(len(constraints))]
    )