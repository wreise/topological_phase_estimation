# Author(s): Wojciech Reise
#
# Copyright (C) 2021 Inria

import concurrent.futures as cf
from functools import partial
from itertools import product
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

from segmentation.estimation_n import estimate_N_from_tree
from segmentation.homology import signal_to_diagram

from benchmarks import zero_crossings_phase
from f_forms import fs_easy
from process_utils import get_GP

logging.basicConfig(filename=f"experiment.log", filemode='w', level=logging.INFO)


def process_one_sample(ind, args):
    ind_f, (ind_gamma, gamma), (ind_SNR, SNR), (ind_time_scale, time_scale) = args

    signal = fs[ind_f](gamma)
    vanilla_noise = get_GP(t, sigma=1., time_scale=time_scale,
                           n_samples=n_trials).T/SNR
    noisy_signals = signal + vanilla_noise
    diagrams = [signal_to_diagram(sig) for sig in noisy_signals]
    dict_meta = {"future_id": ind, "i_gamma": ind_gamma, "i_snr": ind_SNR, "snr": SNR,
                 "i_time_scale": ind_time_scale, "i_f": ind_f}
    results = [{"method": "homology_tree", "N_hat": estimate_N_from_tree(diag)}
               for diag in diagrams]
    #results.extend([{"method": "clustering_boosting",
    #                 "N_hat": estimate_N_with_boosting(diag, estimator=estimate_N_with_clustering)}
    #                for diag in diagrams])
    #results.extend([{"method": "clustering_boosting",
    #                 "N_hat": estimate_N_with_boosting(diag, estimator=partial(estimate_N_with_clustering, get_gcd=SDA))}
    #                for diag in diagrams])
    results.extend([{"method": "zero-crossings", "N_hat": zero_crossings_phase(signal)[-1]}
                    for signal in noisy_signals])
    for r in results:
        r.update(dict_meta)
    return results


if __name__ == "__main__":

    import warnings
    warnings.filterwarnings('ignore')

    gammas = np.load(f"./../data/gammas.npy")
    fs = fs_easy
    t = np.linspace(0., 1., gammas.shape[1])

    SNRs = np.logspace(np.log10(2), np.log10(20), 10)
    time_scales = np.logspace(-2., np.log10(0.4), 10)

    n_trials = 50

    n_all = len(fs) * gammas.shape[0] * len(SNRs)*len(time_scales)
    iterator = product(range(len(fs)), enumerate(gammas),
                       enumerate(SNRs), enumerate(time_scales))

    results = []

    logging.info("Loaded data")
    with cf.ProcessPoolExecutor(max_workers=11) as executor:
        future_to_ind = {executor.submit(process_one_sample, ind_, args): ind_
                         for ind_, args in enumerate(iterator)}

        for future in tqdm(cf.as_completed(future_to_ind), total=n_all):
            ind = future_to_ind[future]
            try:
                result_N_dict = future.result()
                results.extend(result_N_dict)
            except Exception as exc:
                logging.error(f"{ind} generated an exception: {exc}", exc_info=True)

    results = pd.DataFrame.from_dict(results)
    results.to_pickle(f"./results/experiment.pkl")

