# Large-Scale Nonlinear Granger Causality (lsNGC)

## Overview
Large-Scale Nonlinear Granger Causality (lsNGC) is a computational framework designed for inferring directed dependencies from short multivariate time-series data. It addresses the challenge of deriving causal graphs that represent the underlying generative processes of large-scale multivariate data, particularly when the relationships among the data are nonlinear and the observational time series are brief.

## Getting Started
To quickly experiment with lsNGC, access our interactive Colab notebook:
[Demo_lsNGC.ipynb](https://colab.research.google.com/github/ali-vosoughi/Large-scale-nonlinear-causality/blob/main/Demo_lsNGC.ipynb)

## Background
The task of identifying nonlinear and directed relations among components of complex systems from simultaneous time-series observations is a critical and expanding area of research. lsNGC efficiently identifies causal relations through nonlinear state-space transformations of limited observational data, without relying on explicit a priori assumptions about the functional interdependencies among component time series.

## Key Contributions
- Introduction of the lsNGC framework for identifying large-scale nonlinear Granger causality.
- Implementation of conditional Granger causality analysis between two multivariate time series, taking into account a large number of confounding time series.
- Adaptation for scenarios with a limited number of time-series samples but large spatial resolution.
- Conversion of multivariate time-series data into a graph adjacency matrix to represent causal relationships.

## Workflow
lsNGC leverages theoretical concepts from Granger causality analysis, focusing on the predictability and precedence of time series. It estimates causal relationships by creating a nonlinear transformation of the state-space representation for each time series, facilitating the measurement of its influence on the system. Detailed theoretical concepts are discussed in the supplementary material available [here](https://static-content.springer.com/esm/art%3A10.1038%2Fs41598-021-87316-6/MediaObjects/41598_2021_87316_MOESM1_ESM.pdf).

## Results
The lsNGC approach has been evaluated against several benchmark simulations, demonstrating its performance in comparison to four state-of-the-art methodologies. Additional implementation details and results are provided in the supplementary material.

## Citation
If you utilize this code (in whole or part) for your research, please cite our paper:

Wismüller, A., Dsouza, A.M., Vosoughi, M.A., et al. Large-scale nonlinear Granger causality for inferring directed dependence from short multivariate time-series data. Scientific Reports 11, 7817 (2021). [DOI: 10.1038/s41598-021-87316-6](https://doi.org/10.1038/s41598-021-87316-6)

Public access to the paper is available [here](https://www.nature.com/articles/s41598-021-87316-6).

### BibTeX
```bibtex
@article{wismuller2021large,
  title={Large-scale nonlinear Granger causality for inferring directed dependence from short multivariate time-series data},
  author={Wism{\"u}ller, Axel and Dsouza, Adora M and Vosoughi, M Ali and Abidin, Anas},
  journal={Scientific reports},
  volume={11},
  number={1},
  pages={7817},
  year={2021},
  publisher={Nature Publishing Group UK London}
}
