# Author(s): Wojciech Reise
#
# Copyright (C) 2021 Inria

import numpy as np
from scipy.signal import chirp, hilbert


def hilbert_phase(signal):
    analytic_signal = hilbert(signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))/(2.*np.pi)
    return instantaneous_phase


def zero_crossings_phase(signal):
    signs = np.sign(signal)
    signs_non_zero = signs[np.nonzero(signs)[0]]
    is_sign_different = np.array([0., *np.abs(np.diff(signs_non_zero)/2.)])
    count_full_turns = np.cumsum(is_sign_different)
    return count_full_turns
