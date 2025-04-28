# FMA: A Fairness-constrained Optimal Model Averaging Algorithm with Integrated High-Dimensional Sparsity Learning

# Introduction
FMA is a fairness-constrained model averaging algorithm that identifies the optimal linear combination of a pool of sparse candidate models under given fairness constraints. It is effective for deriving fair generalized linear regression models in high-dimensional settings. A detailed description of the FMA algorithm and its applications is provided in the paper referenced below.

This repository contains a vignette example and an implementation of the FMA algorithm for the logistic regression setting. The current implementation supports three types of unfairness measures — statistical parity, equal opportunity, and equal odds — whose definitions can be found in the referenced paper.

# Instructions
In the FMA algorithm, a fairness-assisted stepwise method is integrated to generate candidate models, with its code located in the `./FairAssisted` folder. The code for the main FMA algorithm can be found in the `./FairConstrained` folder. The FMA function is executed in R.

To implement FMA under the unfairness measures of equal opportunity and equal odds, we rely on the Disciplined Convex-Concave Programming (DCCP) solver, which is available only in Python. Consequently, the `reticulate` package is needed to run the Python script related to the DCCP solver `(./FairConstrained/Python Code/optimeopp.py)` within R.

Below are the package dependencies required for the implementation:

**R**: `CVXR == 1.0.11`, `reticulate == 1.34.0`

**Python**: `numpy == 1.26.0`, `cvxpy == 1.4.1`, `dccp == 1.0.3`

# License
FMA is released under the GPL-3 License (refer to the LICENSE file for details).

# Citing *FMA*
