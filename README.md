`cbi_partitions` is a Python library that implements Conformalized Bayesian Inference (CBI, introduced by [1]) for clustering problems based on partition-valued MCMC output.

Given MCMC samples and a notion of distance between data partitions, the library implements:
- a point estimate of the data clustering;
- a credible set of partitions with guaranteed posterior coverage constructed using conformal prediction principles, together with a normalized measure of posterior typicality for any given partition (interpretable as a $p$-value and suitable for formal hypothesis testing);
- a density-based clustering approach to explore and summarize the multimodal structure of the posterior distribution over the space of partitions.



## Installation

You can install the library directly from GitHub into any Python environment:

```bash
pip install https://github.com/nbariletto/cbi_partitions/archive/main.zip
```

---

<br>

## Overview

The library consists of three main components:

1. **Internal partition distance computations**, implemented in Numba for efficiency;
2. **`PartitionKDE`**, the standard KDE-based pipeline for CBI;
3. **`PartitionBall`**, a distance-based CBI method yielding metric credible balls.

Partitions are represented as integer-valued arrays of length $n$, where the $i$-th entry denotes the cluster label assigned to observation $i=1,\ldots,n$.

---

## Partition distances (internal)

The library internally supports two distances between partitions:

- **Variation-of-Information (VI) metric** [2];
- **Binder loss** [3].

Distance computation and optional remapping of cluster labels to a compact range are handled internally using Numba-accelerated routines. These operations are implementation details and are **not exposed as part of the public API**. Users interact with distances exclusively through the high-level conformal models described below.

---

## PartitionKDE

### Description

`PartitionKDE` implements the KDE-based CBI pipeline described by [1].

Given a collection of training partitions $\theta_t$, $t=1,\ldots,T$ and a distance function $D$ between partitions, a new partition $\theta$ is scored by averaging an exponential kernel applied to its distances from the training set:

$$
s(\theta) = \frac{1}{T}\sum_{t=1}^T e^{-\gamma \cdot \mathcal D(\theta,\theta_t)}.
$$

The score acts as:
- a proxy for posterior density used for point estimation and posterior multimodality analysis;
- a conformity score to construct a conformal credible set with prescribed posterior coverage;

---

### Constructor

```python
PartitionKDE(
    train_partitions,
    metric='vi',
    gamma=0.5,
    subsample_size=None,
    remap_labels=True
)
```

#### Parameters

- **`train_partitions`**: array-like of shape `(T, n)`  
  Training partitions, typically obtained from MCMC.
- **`metric`**: `'vi'` or `'binder'` (default `'vi'`)  
  Distance used internally by the kernel.
- **`gamma`**: float (default `0.5`)  
  Kernel decay parameter.
- **`subsample_size`**: int or `None` (default `None`)  
  Optional random subsample size for the training set.
- **`remap_labels`**: bool (default `True`)  
  Whether cluster labels are remapped internally.

---

### Methods

#### `score`

```python
score(partitions)
```

Computes the KDE score $s(\cdot)$ for one or more partitions.


- **Parameters**
  - `partitions`: a single partition (array of length `n`) or an array of partitions of shape `(m, n)`

- **Returns**
  - A one-dimensional array of length `m` containing the KDE score of each partition

---

#### `calibrate`

```python
calibrate(calib_partitions)
```

Computes all quantities needed for subsequent conformal credible set constructions and posterior multimodality analysis, using the partitions in `calib_partitions` as a calibration set.

This method must be called before computing p-values or performing multimodality analysis.

- **Parameters**
  - `calib_partitions`: array of calibration partitions of shape `(m, n)`

- **Side effects**
  After calling this method, the following attributes are available:

  - `calib_partitions_`: the calibration partitions
  - `calib_scores_`: KDE scores for all calibration partitions
  - `calib_dist_matrix_`: pairwise distance matrix between calibration partitions
  - `dpc_delta_`: for each calibration partition, the distance to the closest calibration partition with a higher KDE score. By default, if a partition appears more than once in the calibration set, all instances but one will receive a delta value of zero
  - `dpc_gamma_`: product of the KDE score and the corresponding separation value, used to rank candidate modes

---

#### `compute_p_value`

```python
compute_p_value(partitions)
```

Computes the conformal $p$-value for one or more partitions.

Given a partition $\theta$, its KDE score is first computed. The returned value $p(\theta)$ is then obtained as the fraction of calibration partitions whose KDE score is less than or equal to that of $\theta$, with a finite-sample correction. This quantity can be interpreted as a conformal $p$-value under the assumption that the calibration samples and the tested partition are jointly iid from the posterior distribution. Moreover, the set of partitions $\theta$ with $p(\theta)\geq \alpha$ is a set with posterior coverage at least $1-\alpha$ [4].

The method also supports batch evaluation: if multiple partitions are provided, $p$-values are computed independently for each of them in a vectorized and parallelized manner.

- **Parameters**
  - `partitions`: either a single partition (array of length `n`) or an array of partitions of shape `(m, n)`

- **Returns**
  - If a single partition is provided, returns a scalar $p$-value in $(0,1]$
  - If multiple partitions are provided, returns an array of $p$-values of length `m`

---

#### `get_point_estimate`

```python
get_point_estimate(source='calibration')
```

Returns the partition with the highest KDE score within the selected dataset. This can be interpreted as a pseudo–maximum-a-posteriori (pseudo-MAP) estimate.

- **Parameters**
  - `source`: `'train'` or `'calibration'`, indicating in which set of partitions is searched

- **Returns**
  - A single partition (array of length `n`). If multiple partitions attain the maximum KDE score, one of them is returned. In this case, the choice corresponds to the first maximizer encountered and is deterministic given the ordering of the partitions.

---

## Posterior multimodality and density peaks

During calibration, the library also computes quantities used to explore posterior multimodality using density-based clustering ideas, and in particular Density-Peak Clustering [4].

For each calibration partition $\theta$, the following are computed internally:

- its KDE score, interpreted as a proxy for posterior density;
- its distance to the closest calibration partition with a higher KDE score.

Partitions that simultaneously exhibit a high KDE score and a large distance to higher-density partitions can be interpreted as representatives of distinct modes of the posterior distribution over partitions.

---

### `plot_dpc_decision_graph`

```python
plot_dpc_decision_graph(save_path=None)
```

Plots the decision graph displaying KDE score ($s$) versus minimum distance to higher-density calibration partitions ($\delta$).

- **Parameters**
  - `save_path`: optional path where the figure is saved as .png

---

### `get_dpc_modes`

```python
get_dpc_modes(s_thresh, delta_thresh)
```

Identifies calibration partitions corresponding to well-separated, high-density regions of the posterior, based on the user-specified KDE thresholds `s_thresh` (KDE score, $s$) and `delta_thresh` (minimum distance from higher-density partitions, $\delta$). These thresholds are selected by examining the decision graph produced by `plot_dpc_decision_graph`, with the goal of isolating partitions that lie in the top-right portion of the graph, away from the bulk of the other samples.


- **Parameters**
  - `s_thresh`: minimum KDE score threshold
  - `delta_thresh`: minimum separation threshold

- **Returns**
  - An array of indices of calibration partitions satisfying both thresholds, ordered by the product of KDE score and separation


## PartitionBall

### Description

`PartitionBall` implements an alternative CBI procedure, yielding credible sets that coincide with metric balls around a chosen centering point estimate [6]. This class is not central to the proposed CBI methodology and is included primarily for comparison with a previously established method that can be reinterpreted through the lens of CBI.

---

### Constructor

```python
PartitionBall(
    point_estimate_partition,
    metric='vi',
    remap_labels=True
)
```

#### Parameters

- **`point_estimate_partition`**: array-like of shape `n`  
  A partition at which to center the credible set (usually estimated in advance).
- **`metric`**: `'vi'` or `'binder'` (default `'vi'`)  
  Distance used internally by the kernel.
- **`remap_labels`**: bool (default `True`)  
  Whether cluster labels are remapped internally.

---

### Methods

#### `score`

```python
score(partitions)
```

Computes the metric ball score for one or more partitions.

For any partition $\theta$, given a distance $\mathcal D$ between partitions and denoting by $\hat\theta$ the partition corresponding to `point_estimate_partition`, the ball score for $\theta$ is defined as

$$
\tilde s(\theta) = -\mathcal D(\theta,\hat\theta).
$$

- **Parameters**
  - `partitions`: a single partition (array of length `n`) or an array of partitions of shape `(m, n)`

- **Returns**
  - A one-dimensional array of length `m` containing the KDE score of each partition


---

#### `calibrate`

```python
calibrate(calib_partitions)
```

Computes the ball scores for all partitions in `calib_partitions`.

This method must be called before computing p-values.

- **Parameters**
  - `calib_partitions`: array of calibration partitions of shape `(m, n)`

- **Side effects**
  After calling this method, the following attributes are available:

  - `calib_partitions_`: the calibration partitions
  - `calib_scores_`: ball scores for all calibration partitions

---

#### `compute_p_value`

```python
compute_p_value(partitions)
```

Computes the conformal $p$-value for one or more partitions.

Given a partition $\theta$, its ball score $\tilde s(\theta)$ is first computed. The returned value $\tilde p(\theta)$ is then obtained as the fraction of calibration partitions whose ball score is less than or equal $\tilde s(\theta)$, with a finite-sample correction. This quantity can be interpreted as a conformal $p$-value under the assumption that the calibration samples and the tested partition are jointly iid from the posterior distribution. Moreover, the set of partitions $\theta$ with $\tilde p(\theta)\geq \alpha$ is a ball (in the chosen metric) around `point_estimate_partition` with posterior coverage at least $1-\alpha$ [4].

The method also supports batch evaluation: if multiple partitions are provided, $p$-values are computed independently for each of them in a vectorized and parallelized manner.

- **Parameters**
  - `partitions`: either a single partition (array of length `n`) or an array of partitions of shape `(m, n)`

- **Returns**
  - If a single partition is provided, returns a scalar $p$-value in $(0,1]$
  - If multiple partitions are provided, returns an array of $p$-values of length `m`
---

## References

[1] Bariletto, N., Ho, N., & Rinaldo, A. (2025). *Conformalized Bayesian Inference, with Applications to Random Partition Models*. arXiv preprint.

[2] Meilă, M. (2007). Comparing clusterings—an information based distance. Journal of Multivariate Analysis, 98(5), 873-895.

[3] Binder, D. A. (1978). Bayesian cluster analysis. Biometrika, 65(1), 31-38.

[4] Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic learning in a random world. Boston, MA: Springer US.

[5] Rodriguez, A., & Laio, A. (2014). Clustering by fast search and find of density peaks. Science, 344(6191), 1492-1496.

[6] Wade, S., & Ghahramani, Z. (2018). Bayesian cluster analysis: Point estimation and credible balls (with discussion). Bayesian Analysis, 13(2), 559–626.






