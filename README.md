# Topological Phase Estimation
This repository contains an implementation of the algorithms presented in [Topological Phase Estimation](https://arxiv.org/abs/2205.14390).
It also features the code to generate the data for the experiments.

## Installation
We recommend installing the package with conda, for example in the following way.
```
conda create -n top_phase_estimation python=3.8
conda activate top_phase_estimation
python -m pip install .
```
To run the examples or the experiments, change the last line to `python -m pip install ".[experiments,notebooks]"`. `

## API
We point to the location of functions from the paper.
- calculate the homology by filtering the circle: `segmentation.homology.signal_to_diagram`,
- estimate `N`: `segmentation.estimation_n.estimate_N`,
- estimate N without choosing `tau`: `segmentation.estimation_n.estimation_N_with_tree`,
- segmentation of `[0,1]`: `segmentation.segmentation.segment_signal_with_clustering`.

## Notebooks and experiments
The repository contains the data and code used to produce the experimental results from section 6.2.
