# Author(s): Wojciech Reise
#
# Copyright (C) 2022 Inria
import platform
import subprocess
import uuid
from warnings import warn
import numpy as np


MAX_N = 1000
MAX_ERROR = 6

default_eta = np.ceil(np.log2(MAX_N))
default_rho = np.ceil(np.log2(MAX_ERROR))


# This function requires `fplll`. See https://github.com/fplll/fplll
def SDA(xs, rho=default_rho, eta=default_eta, gamma=None):
    if gamma is None:
        gamma = np.ceil(np.max(np.log2(xs)))
    if not(hasattr(xs, "shape")):
        xs = np.array(xs)
    t, x_0 = xs.shape[0], xs[0]

    # Initialize the lattice basis
    B = np.diag(np.array([2 ** (rho + 1)] + t * [-xs[0]]))
    B[0, 1:] = xs

    if (eta is not None) & (gamma is not None):
        # Condition for v=(q_0 2**(rho+1), ...) to be the shortest vector in the basis
        margin = np.sqrt((t + 1.) / (2. * np.pi * np.e)) * (2 ** (rho + 1) * x_0 ** t) ** (1 / (t + 1)) - np.sqrt(
            t + 1.) * (2. ** (gamma - eta + rho + 1))
        is_condition_satisfied = margin > 0
        if not is_condition_satisfied:
            warn(f"Condition not satisfied: {margin} !> 0")
    else:
        warn("Not checking if quality guarantee satisfied")

    # Save the basis
    tmp_filename = uuid.uuid4().hex + ".fplll"
    with open(tmp_filename, "w") as f:
        f.write(np.array2string(B.astype(int), threshold=np.inf, max_line_width=np.inf))
        #B.tofile()

    # Run the "shortest-vector-problem" algorithm and retrieve the result
    fplll_svp_command = ["fplll", "-a", "svp", tmp_filename]
    ret = subprocess.run(fplll_svp_command, stdout=subprocess.PIPE)
    vector_as_string = ret.stdout
    w = np.array([int(k) for k in (vector_as_string[1:-2]).split(b' ')])
    remove_file = ["rm", tmp_filename]
    subprocess.run(remove_file)

    # Use the first component of the vector to finish the algo
    w_0 = w[0]
    q_0_hat = w_0 / (2 ** (rho + 1))
    r_0_hat = x_0 % q_0_hat
    p_hat = (x_0 - r_0_hat) / q_0_hat
    p_hat

    return p_hat