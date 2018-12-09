# Machine Learning Methods

This repo is a collection of machine learning algorithms developed during my doctoral program as well as my implementations of classic models in my downtime.

## Machine Learning Methods from My Dissertation

A brief introduction to the mathematical context of these machine learning methods can be found in my [final defense slides](Dissertation.pdf). The codes are collected in module `mark_i.py`:

- Hyperspherical heat kernel
  - `class HypersphericalHeatKernel(t, dimension_euclidean, max_itr = 100, eps_tol = 1e-8, normed=True)`
  - callable arguments: `cos` and optional `t`
  - Gegenbauer polynomial: my own implementation using recursion relation and memoization; `scipy`  implementation has numerical instability issues.
    - `class GegenbauerPolynomial(alpha, order)`

- Effective Dissimilarity
  - `class EffectiveDissimilarity(data, metric="euc")`
  - method `get_MDS(tau, max_embedding_dimension=3)`  returns an MDS instance
  - <u>M</u>ulti<u>d</u>imensional <u>S</u>caling (MDS)
    - `class MDS(distance_matrix, max_embedding_dimension=3)`
    - method `get_embedding()`  returns an embedding of input distances in an Euclidean space of maximum dimension `max_embedding_dimension`
- Spectral Clustering
  - Scikit-Learning Spectral Clustering has an serious bug, thus I provided my own implementation
  - `class SpectralClustering(n_clusters, norm_method='row', is_exact=True)`
  - Quantum Transport Clustering
    - An alternative way to perform spectral embedding
    - `class QuantumTransportClustering(n_clusters, Hamiltonian, s=1.0, is_exact=True, n_eigs = None) `