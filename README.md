`cbi_partitions` is a Python library that implements Conformalized Bayesian Inference (CBI, introduced by [1]) for clustering problems based on partition-valued MCMC output.

Given MCMC samples and a notion of distance between data partitions, the library implements:
- a point estimate of the data clustering;
- a credible set of partitions with guaranteed posterior coverage constructed using conformal prediction principles, together with a normalized measure of posterior typicality for any given partition (interpretable as a $p$-value and suitable for formal hypothesis testing);
- a density-based clustering approach to explore and summarize the multimodal structure of the posterior distribution over the space of partitions.

---

## Installation

You can install the library directly from GitHub into any Python environment:

```bash
pip install https://github.com/nbariletto/cbi_partitions/archive/main.zip
```

---

<br>

## Overview

The library is organized around three main components:

1. Partition distance functions implemented in Numba (Variation of Information and Binder);
2. `PartitionKDE`, a density-based conformal model on the space of partitions;
3. `PartitionBall`, a distance-based conformal model yielding credible balls around a point estimate.

All partitions are represented as integer-valued arrays of length $n$, where the $i$-th entry denotes the cluster label of observation $i$.

---

## Partition distances

### Variation of Information (VI)

The VI distance between two partitions $\theta_1, \theta_2$ is defined as

$$
\mathrm{VI}(\theta_1, \theta_2)
= H(\theta_1 \mid \theta_2) + H(\theta_2 \mid \theta_1),
$$

where $H(\cdot \mid \cdot)$ denotes conditional entropy.  
The VI distance is a metric and is invariant to label permutations.

### Binder distance

The Binder distance is defined as

$$
1 - \text{Rand Index},
$$

and measures disagreement between pairwise co-clustering decisions.

### Label remapping

For numerical stability and memory efficiency, cluster labels can be internally remapped to a compact set $\{0,\dots,K-1\}$.  
This behavior is controlled by the `remap_labels` argument in all public APIs.

---

## PartitionKDE

### Description

`PartitionKDE` implements a kernel density–based conformal score on the space of partitions.  
Given training partitions $\{\theta_t\}_{t=1}^T$, the score of a partition $\theta$ is

$$
s(\theta)
= \frac{1}{T} \sum_{t=1}^T \exp\bigl(-\gamma\, D(\theta,\theta_t)\bigr),
$$

where $D$ is either the VI or Binder distance and $\gamma > 0$ is a tuning parameter.

The score serves simultaneously as:
- a proxy for posterior density;
- a conformity score for conformalized Bayesian inference;
- the basis for point estimation and multimodality analysis.

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

- **train_partitions** (`array-like`, shape `(T, n)`): training partitions (e.g. MCMC output).
- **metric** (`'vi'` or `'binder'`, default `'vi'`): distance used in the kernel.
- **gamma** (`float`, default `0.5`): bandwidth parameter.
- **subsample_size** (`int` or `None`, default `None`): optional training subsample size.
- **remap_labels** (`bool`, default `True`): whether to remap cluster labels internally.

---

### Methods

#### score

```python
score(partitions)
```

Computes KDE scores for one or more partitions.

- **Parameters**
  - `partitions`: array of shape `(n,)` or `(m, n)`
- **Returns**
  - `numpy.ndarray` of shape `(m,)`

---

#### calibrate

```python
calibrate(calib_partitions)
```

Scores calibration partitions and computes all quantities required for conformal inference and multimodality analysis.

- **Parameters**
  - `calib_partitions`: array of shape `(C, n)`

This method must be called before computing p-values.

---

#### compute_p_value

```python
compute_p_value(partition)
```

Computes the conformal p-value

$$
p(\theta)
= \frac{1 + \#\{c : s(\theta_c) \le s(\theta)\}}{1 + C}.
$$

- **Returns**
  - `float` in $(0,1]$

Higher values indicate greater posterior typicality.

---

#### get_point_estimate

```python
get_point_estimate(source='calibration')
```

Returns the partition with the highest KDE score.

- **source**: `'train'` or `'calibration'`

This estimator can be interpreted as a pseudo-MAP estimate.

---

### Density Peak Clustering (DPC)

For each calibration partition $\theta$, the following are computed:

- $s(\theta)$: normalized density score;
- $\delta(\theta)$: distance to the nearest partition with higher score.

These quantities are used to identify posterior modes.

#### plot_dpc_decision_graph

```python
plot_dpc_decision_graph(save_path=None)
```

Plots the $(s,\delta)$ decision graph.

---

#### get_dpc_modes

```python
get_dpc_modes(s_thresh, delta_thresh)
```

Identifies mode candidates via thresholding.

- **Returns**
  - array of indices sorted by $s(\theta)\,\delta(\theta)$

---

## PartitionBall

### Description

`PartitionBall` implements a distance-based conformal score centered at a fixed point estimate $\hat\theta$.

The nonconformity score is

$$
\tilde s(\theta) = D(\theta, \hat\theta),
$$

yielding conformal credible sets that coincide with metric balls around $\hat\theta$.

---

### Constructor

```python
PartitionBall(
    point_estimate_partition,
    metric='vi',
    remap_labels=True
)
```

---

### Methods

#### score

```python
score(partitions)
```

Computes distances to the center partition.

---

#### calibrate

```python
calibrate(calib_partitions)
```

Computes calibration distances.

---

#### compute_p_value

```python
compute_p_value(partition)
```

Computes the conformal p-value

$$
p(\theta)
= \frac{1 + \#\{c : \tilde s(\theta_c) \ge \tilde s(\theta)\}}{1 + C}.
$$

Lower distances correspond to higher posterior typicality.

---

#### get_point_estimate

```python
get_point_estimate()
```

Returns the center partition.

---

## Notes

- All distance computations are Numba-jitted for efficiency.
- Pairwise calibration distances scale quadratically in the number of calibration samples.
- The library is agnostic to the source of partition samples.

---

## References

[1] Bariletto, N., Ho, N., & Rinaldo, A. (2025). *Conformalized Bayesian Inference, with Applications to Random Partition Models*. arXiv:2511.05746.

[2] Rodriguez, A., & Laio, A. (2014). *Clustering by fast search and find of density peaks*. Science, 344(6191), 1492–1496.

