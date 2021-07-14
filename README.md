# Codes for nonlinear Granger causality (lsNGC)
 Large-scale nonlinear Granger causality for inferring directed dependence from short multivariate time-series data

## Background
Identifying nonlinear and directed relations between components of a complex system, especially from simultaneously observed time series, is an actively growing area of research. By modeling interactions with nonlinear state-space transformations from limited observational data, lsNGC identifies casual relations with no explicit a priori assumptions on functional interdependence between component time series in a computationally efficient manner.

## Contributions
- Here, we introduce large-scale nonlinear Granger causality (lsNGC) which facilitates conditional Granger causality between two multivariate time series conditioned on a large number of confounding time series with a small number of observations.
- The code converts multivariate time-series data to graph adjacency matrix that represents causal relationships. 

## Workflow
Large-scale nonlinear Granger causality adopts theoretical concepts from Granger causality analysis. Granger causality (GC) is based on the concept of time series precedence and predictability; here, the improvement in the prediction quality of a time series in the presence of the past of another time series is quantified. This reveals if the predicted time series was influenced by the time series whose past was used in the prediction, uncovering the causal relationship between the two series6 under investigation. The supplementary material (section 1) details the theoretical concepts involved in Granger causality analysis. LsNGC estimates causal relationships by first creating a nonlinear transformation of the state-space representation of the time series, whose influence on others is to be measured, and another representation of the rest of the time series in the system. 

- Supplementary information regarding the implementation is here: https://static-content.springer.com/esm/art%3A10.1038%2Fs41598-021-87316-6/MediaObjects/41598_2021_87316_MOESM1_ESM.pdf

## Results
- To evaluate the approach, several benchmark simulations are considered and performance is compared to four state-of-the-art approaches.
- Supplementary information regarding the implementation is here: https://static-content.springer.com/esm/art%3A10.1038%2Fs41598-021-87316-6/MediaObjects/41598_2021_87316_MOESM1_ESM.pdf

## Cite
If you use this code (partially or as a whole), please cite the following paper:

- Wismüller, A., Dsouza, A.M., Vosoughi, M.A. et al. Large-scale nonlinear Granger causality for inferring directed dependence from short multivariate time-series data. Sci Rep 11, 7817 (2021). https://doi.org/10.1038/s41598-021-87316-6

- The public access to the paper is available here: https://www.nature.com/articles/s41598-021-87316-6



