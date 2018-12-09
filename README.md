# Machine Learning Methods

This repo is a collection of machine learning algorithms developed during my doctoral program as well as my implementations of classic models in my downtime.

My deep learning projects can also be found at

- [Neural Camouflage](https://chenchaozhao.github.io/NeuralCamouflage/)
- [Fingerprints of the Invisible Hand](https://chenchaozhao.github.io/FingerprintsOfTheInvisibleHand/)

## Machine Learning Methods from My Dissertation

A brief introduction to the mathematical context of these machine learning methods can be found in my [final defense slides](Dissertation.pdf). The codes are collected in module `mark_i.py`:

- Hyperspherical heat kernel

  - ```python
    class HypersphericalHeatKernel(t, dimension_euclidean, max_itr = 100, eps_tol = 1e-8, normed=True)
    ```

  - callable arguments: `cos` and optional `t`

  - Gegenbauer polynomial: my own implementation using recursion relation and memoization; `scipy`  implementation has numerical instability issues.

    - ```python
      class GegenbauerPolynomial(alpha, order)
      ```

- Effective Dissimilarity

  - ```python
    class EffectiveDissimilarity(data, metric="euc")
    ```

  - method `get_MDS(tau, max_embedding_dimension=3)`  returns an MDS instance

  - <u>M</u>ulti<u>d</u>imensional <u>S</u>caling (MDS)

    - ```python
      class MDS(distance_matrix, max_embedding_dimension=3)
      ```

    - method `get_embedding()`  returns an embedding of input distances in an Euclidean space of maximum dimension `max_embedding_dimension`

- Spectral Clustering

  - Scikit-Learning Spectral Clustering has an serious bug, thus I provided my own implementation

  - ```python
    class SpectralClustering(n_clusters, norm_method='row', is_exact=True)
    ```

  - Quantum Transport Clustering

    - An alternative way to perform spectral embedding

    - ```python
      class QuantumTransportClustering(n_clusters, Hamiltonian, s=1.0, is_exact=True, n_eigs = None)
      ```

## Mixture Models

Dirichlet Process Mixture Model

```python
class DirichletProcessMixtureModel(alpha, n_dim=2, sig2=1, max_itr=50)
```

See [demo notebook](/examples/DPMM_demo.ipynb).

## Markov Models

Hidden Markov Model

```python
class HiddenMarkovModel(params, max_iter=500, eps=1e-6)
```

See [demo notebook](/examples/HMM_demo.ipynb).

## Bandit Models

N-Armed Bandit

```python
class NArmedBandit(n, reward_distribution_parameters_, decision='greedy', reward_distribution='normal', eps=None, beta=None, optimistic_init = 0)
```

See [demo notebook](/examples/Bandit_demo.ipynb).

