`cbi_partitions` is a Python library for performing Conformalized Bayesian Inference (CBI, introduced by [1]) for clustering problems based on partition-valued MCMC output.

Given MCMC samples and the Variation-of-Information (VI) metric between data partitions [2], the library implements:
- a point estimate of the data clustering;
- a credible set of partitions with guaranteed posterior coverage constructed using conformal prediction principles [3], together with a normalized measure of posterior typicality for any given partition (interpretable as a $p$-value and suitable for formal hypothesis testing);
- a density-based clustering approach to explore and summarize the multimodal structure of the posterior distribution over the space of partitions.



## Installation

You can install the library directly from GitHub into any Python environment:

```bash
pip install https://github.com/nbariletto/cbi_partitions/archive/main.zip
```


---

<br>

## Tutorial: Analyzing a multimodal posterior distribution over partitions

This tutorial provides a step-by-step reproduction of the experiment described in [1], which consists of a CBI analysis of MCMC samples from a mixture-based random partition model fitted to two-dimensional simulated data. Specifically, we simulate a dataset with an ambiguous clustering structure, draw partitions from the posterior of a Pitman–Yor (PY) Gaussian mixture model, and use `cbi_partitions` to quantify uncertainty and detect posterior multimodality.

### 1. Data simulation and MCMC
We generate a dataset of $n = 100$ observations from a mixture of three Gaussian components. The component-specific means and covariance matrices are chosen to induce overlap between the two leftmost clusters, thereby generating posterior uncertainty. The plot below shows the true data-generating clustering structure and illustrates this ambiguity.


```python
import numpy as np
import matplotlib.pyplot as plt

# --- Simulation Configuration ---
config = {
    'n_obs': 100, 
    'dim': 2, 
    'n_clusters': 3,
    'seed': 12345
}

def simulate_gmm_data(n_obs, dim, n_clusters, seed):
    np.random.seed(seed)
    means = np.array([[-3, -3], [-3, 3], [3, 0]])
    cov = np.eye(dim) * 1.5
    
    true_labels = np.random.randint(0, n_clusters, n_obs)
    X = np.zeros((n_obs, dim))
    
    for i in range(n_obs):
        X[i, :] = np.random.multivariate_normal(means[true_labels[i]], cov)
        
    return X, true_labels

X, true_labels = simulate_gmm_data(**config)

# Visualize Ground Truth
plt.figure(figsize=(3, 2))
plt.scatter(X[:,0], X[:,1], c=true_labels, cmap='brg', edgecolor='k', s=50)
plt.show()
```
<p align="center">
  <img src="images/true_partition.png" alt="Ground Truth">
</p>

We now fit a $PY(0.03, 0.01)$ mean–covariance Gaussian mixture model. The MCMC implementation used here is based on the `pyrichlet` library [4], which we import below and which can be easily installed via `pip`. This choice is purely illustrative: in practice, any partition-valued MCMC output can be used, and one may directly skip to the subsequent CBI-specific steps.

```python
from pyrichlet import mixture_models 

# MCMC parameters
mcmc_config = {
    'n_final_samples': 6000,
    'burn_in': 1000,
    'thinning': 5,
    'alpha': 0.03,
    'py_sigma': 0.01,
}

print("--- Running Pyrichlet MCMC ---")
total_iter = mcmc_config['burn_in'] + (mcmc_config['n_final_samples'] * mcmc_config['thinning'])
dim = config['dim']

# Initialize sampler
mm = mixture_models.PitmanYorMixture(
    alpha=mcmc_config['alpha'], 
    pyd=mcmc_config['py_sigma'],
    mu_prior=X.mean(axis=0), 
    lambda_prior=0.01,
    psi_prior=np.eye(dim) * 1.5, 
    nu_prior=dim + 2,
    rng=config['seed'], 
    total_iter=total_iter,
    burn_in=mcmc_config['burn_in'], 
    subsample_steps=mcmc_config['thinning']
)

# Run sampler
mm.fit_gibbs(y=X, show_progress=True)

# Extract partitions
mcmc_partitions = [samp['d'] for samp in mm.sim_params]
partitions = np.array(mcmc_partitions, dtype=np.int64)
```

Below we plot the posterior distribution of the number of clusters implied by the model. This provides a first glimpse of the clustering uncertainty captured by the Bayesian posterior, as partitions with two, three, and four clusters all receive substantial posterior mass.


```python
from collections import Counter

num_clusters = [len(np.unique(p)) for p in mcmc_partitions]
k_counts = Counter(num_clusters)
k_values_all = sorted(k_counts.keys())
k_probs_all = [k_counts[k] / len(mcmc_partitions) for k in k_values_all]

plt.figure(figsize=(6, 4))
plt.bar(k_values_all, k_probs_all, color='skyblue')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Posterior Probability")
plt.xticks(k_values_all)
plt.grid(axis='y', linestyle='--')
plt.show()
```

<p align="center">
  <img src="images/posterior_k.png" alt="Posterior K Distribution">
</p>

<br>

### 2. CBI - Initialization

To begin, we import `PartitionKDE` and `PartitionBall` from the `cbi_partitions` library.

```python
from cbi_partitions import PartitionKDE, PartitionBall
```

From [1], recall that the core component of CBI is the computation, for all calibration partition samples $\theta$, of the VI-KDE score:

$$
s(\theta) = \frac{1}{T}\sum_{t=1}^T e^{-\gamma \cdot \mathcal D_{VI}(\theta,\theta_t)},
$$

where $\theta_t$, $t=1,\ldots,T$, are training partition samples, $\gamma > 0$ is a tuning parameter, and $\mathcal D_{VI}$ denotes the VI distance between partitions. 

The `PartitionBall` method, instead, allows us to compute an alternative score:

$$
\tilde s(\theta) = - D(\theta, \hat \theta),
$$

where $\hat \theta$ is a point estimate of the data clustering. This choice of score produces a conformal set that coincides with a VI-ball centered at $\hat \theta$, whose size can then be compared to that obtained using the VI-KDE score.

Accordingly, we split the MCMC samples into a *training set* of 5000 partitions to estimate the score, and a *calibration set* of 1000 partitions on which to evaluate $s(\theta)$.


```python
np.random.seed(42)
indices = np.arange(partitions.shape[0])
np.random.shuffle(indices)

split_idx = int(len(partitions) * 5/6)
train_partitions = partitions[indices[:split_idx]]
calib_partitions = partitions[indices[split_idx:]]
```

We are now ready to initialize the model and compute the calibration scores. This can be accomplished in just a few simple lines of code:

```python
# --- Initialize CBI pipeline ---
kde = PartitionKDE(
    train_partitions=train_partitions,
    metric='vi', # default, can select 'binder' as well
    gamma=0.5, # default
    subsample_size=None, # default
    remap_labels=True # default
)

# --- Compute all quantities needed for CBI ---
print("Calibrating KDE model...")
kde.calibrate(calib_partitions)
```

**Note:** When defining `kde`, the only required argument is `train_partitions`; all other parameters have default values (which we use). In particular, the `PartitionKDE` class allows you to:

- choose between the VI and Binder metrics for CBI,
- set the value of $\gamma$ in the definition of the KDE score,
- specify a random training subsample size for computing the calibration scores (`None`, recommended, uses the full training set),
- enable or disable a custom remapping of cluster labels to facilitate VI distance computations.

Additionally, for custom uses such as plotting, the attributes `kde.calib_scores_` and `kde.dpc_delta_` provide access to the computed calibration scores and the separation values $\delta$ required for the multimodality analysis below.

<br>

### 3. CBI – Pseudo-MAP Point Estimate

Using the calibration scores computed above, we obtain the point estimate as the calibration partition with the highest VI-KDE score. This is achieved using the `.get_point_estimate()` method and can be interpreted as a pseudo Maximum-A-Posteriori (MAP) estimate, since the VI-KDE score can be viewed as a proxy for posterior density.

```python
point_est_partition = kde.get_point_estimate()

plt.figure(figsize=(6, 4))
plt.scatter(X[:,0], X[:,1], c=point_est_partition, cmap='brg', edgecolor='k', s=50)
plt.show()
```

<p align="center">
  <img src="images/point_estimate.png" alt="Point estimate">
</p>

Point estimation resolves posterior uncertainty by selecting a two-cluster partition as the point estimate, effectively merging the two leftmost, ambiguously separated data-generating clusters.



<br>

### 4. CBI - Multimodality Analysis
Following [1], we now analyze posterior multimodality using ideas from density-based clustering [5], with the VI-KDE score serving as a proxy for posterior density. Specifically, the calibration step above has already computed (a) the VI-KDE score $s(\theta)$ and (b) the separation parameter $\delta(\theta)$ for each calibration sample $\theta$. Recall from [1] that $\delta(\theta)$ measures the distance between $\theta$ and the closest calibration partition with a higher VI-KDE score. Consequently, we can identify posterior modes by examining the decision graph plotted below and selecting partitions that exhibit unusually large values of both $s(\theta)$ (indicating they lie in high-density regions) and $\delta(\theta)$ (indicating they are well separated from other high-density samples).


```python
kde.plot_dpc_decision_graph()
plt.show()
```

<p align="center">
  <img src="images/dpc_decision_graph.png" alt="DPC decision graph">
</p>

Two distinct modes are clearly identified, corresponding to the points in the north-east corner of the decision graph. Below, we extract the corresponding partitions using visually selected $s$ and $\delta$ thresholds, then display these partitions.



```python
modes_idx = kde.get_dpc_modes(s_thresh=0.75, delta_thresh=0.6)

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
for i in range(2):
    axs[i].scatter(X[:,0], X[:,1], c=calib_partitions[modes_idx[i]], cmap='brg', edgecolor='k', s=50)
plt.show()
```

<p align="center">
  <img src="images/dpc_modes.png" alt="DPC modes">
</p>

This shows that both the KDE point estimate (by construction of the density-based clustering procedure) and a three-cluster partition, closely matching the data-generating clustering, are supported as modes of the posterior distribution.

<br>

### 5. CBI – Hypothesis Testing

Point estimation and multimodality analysis provide a global description of the posterior. We now test whether a specific, user-specified partition $\theta$ is supported as typical under the posterior. Following [1], this is done by computing the conformal $p$-value:

$$
p(\theta) = \frac{1 + \text{number of calibration samples with } s(\cdot)\leq s(\theta)}{1 + \text{number of calibration samples}}.
$$

$p(\theta)$ can be formally interpreted as a $p$-value under the null hypothesis that the calibration samples and $\theta$ are jointly i.i.d. from the posterior, or more generally as a measure of posterior typicality: the higher its value, the better $\theta$ ranks in terms of VI-KDE score among the calibration samples. Importantly, the set of partitions

$$
\theta \quad \text{s.t.} \quad p(\theta) \geq \alpha
$$

is guaranteed to have posterior coverage of at least $1-\alpha$ (see [1] for details).

We test four specific clustering hypotheses at a significance level of $\alpha=0.1$, ensuring that the resulting set has at least 90% posterior coverage:

1. **The true partition;**  
2. **The "collapsed" partition,** obtained merging the two leftmost ground-truth clusters;  
3. **The one-cluster trivial partition,** testing whether full homogeneity among data points is supported by the posterior;  
4. **The 100-cluster trivial partition,** testing whether full heterogeneity among data points is supported by the posterior.



```python
# Collapsed partition (merge true leftmost clusters)
collapsed_labels = true_labels.copy()
collapsed_labels[collapsed_labels == 1] = 0

# One-cluster partition (full homogeneity hypothesis)
one_cluster_partition = np.ones(config['n_obs'])

# 100 cluster partition (full heterogeneity hypothesis)
n_cluster_partition = np.arange(config['n_obs'])

# Compute and display p-values
p_val_true = kde.compute_p_value(true_labels)
p_val_coll = kde.compute_p_value(collapsed_labels)
p_val_one = kde.compute_p_value(one_cluster_partition)
p_val_n = kde.compute_p_value(n_cluster_partition)

print(f"Ground Truth (K=3) p-value:    {p_val_true:.4f}")
print(f"Collapsed (K=2) p-value:       {p_val_coll:.4f}")
print(f"One cluster p-value:           {p_val_one:.4f}")
print(f"100 cluster p-value:           {p_val_n:.4f}")
```

```text
Ground Truth (K=3) p-value:    0.3756
Collapsed (K=2) p-value:       0.6893
One cluster p-value:           0.0010
100 cluster p-value:           0.0010
```

As expected from our previous multimodality analysis, both the true partition and its collapsed version are included in the set as highly typical parameter values under the posterior (they were indeed identified as modes). In contrast, both the fully homogeneous and fully heterogeneous hypotheses are strongly ruled out.

<br>

### 6. Comparison with VI Balls Around a Point Estimate

Finally, we compare the results of the VI-KDE score procedure with the VI ball centered around our point estimate (which essentially coincides with the "collapsed" true partition). The VI ball with posterior coverage $1-\alpha$, originally proposed by [6], is obtained within the CBI framework using the negative VI distance $\tilde s(\theta)$ from the point estimate as a score, and is implemented via the `PartitionBall` class.

In particular, we are interested in comparing the sizes of the two sets. Since we identified two modes, we hypothesize that the posterior places most of its mass near each mode, with little mass ''in between'' or ''far from'' both modes. To build intuition, think of the simpler picture a bimodal distribution on the real line with two relatively well separated modes. 

We therefore test inclusion in both sets for the following partitions:

1. **A partition far from both modes,** obtained by ''moving away'' from the "collapsed" partition in the opposite direction from the true partition;  
2. **A partition "between" the two modes.**

The plot below illustrates these two partitions:

```python
# 2. Partition far from modes (vertical split)
far_labels = (X[:, 0] > 2.5).astype(np.int64)

# 3. Partition between modes (collapsed partition with split top-left corner)
between_labels = collapsed_labels.copy()
mask = (between_labels == 0) & (X[:, 1] >= 3.23)
between_labels[mask] = 5

fig, axs = plt.subplots(1, 2, figsize=(8, 3))
for i in range(2):
    axs[i].scatter(X[:,0], X[:,1], c=[far_labels, between_labels][i], cmap='brg', edgecolor='k', s=50)
    axs[i].set_title(["Far from modes","Between modes"][i])
plt.show()
```

<p align="center">
  <img src="images/test_partitions.png" alt="Test partitions">
</p>

We now examine the results of hypothesis testing using both the VI-KDE and VI-ball scores.

```python
# Run conformal tests with KDE score
p_val_far = kde.compute_p_value(far_labels)
p_val_between = kde.compute_p_value(between_labels)

# Initialize VI-ball procedure
ball_model = PartitionBall(
    point_estimate_partition=point_est_partition,
    metric='vi'
)
ball_model.calibrate(calib_partitions)

# Test 'far-from-modes' and 'between-modes' partitions with ball score
p_val_far_ball = ball_model.compute_p_value(far_labels)
p_val_between_ball = ball_model.compute_p_value(between_labels)

# Results
print(f"Far-from-modes partition p-value (KDE):        {p_val_far:.4f}")
print(f"Between-modes partition p-value (KDE):         {p_val_between:.4f}")
print(f"Far-from-modes partition p-value (Ball):       {p_val_far_ball:.4f}")
print(f"Between-modes partition p-value (Ball):        {p_val_between_ball:.4f}")
```

```text
Far-from-modes partition p-value (KDE):        0.0010
Between-modes partition p-value (KDE):         0.0819
Far-from-modes partition p-value (Ball):       0.1399
Between-modes partition p-value (Ball):        0.2587
```

Both partitions are excluded from the 90% VI-KDE credible set but are included in the 90% VI-ball credible set. This demonstrates that the VI-KDE procedure naturally adapts to the multimodality of the posterior by leveraging a pseudo-density estimate, producing a credible set that "skips" low-density regions both between modes and far from them. In contrast, the VI ball with the same coverage is insensitive to the geometry of the posterior support: its radius expands uniformly in all directions from the point estimate until the target coverage is reached, ultimately including low-density, unrepresentative partitions like those we tested.




<br>

### References

[1] Bariletto, N., Ho, N., & Rinaldo, A. (2025). Conformalized Bayesian Inference, with Applications to Random Partition Models. arXiv preprint arXiv:2511.05746.

[2] Meilă, M. (2007). Comparing clusterings—an information based distance. Journal of Multivariate Analysis, 98(5), 873-895.

[3] Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic learning in a random world. Boston, MA: Springer US.

[4] Selva, F., Fuentes-García, R., & Gil-Leyva, M. F. (2025). pyrichlet: A Python Package for Density Estimation and Clustering Using Gaussian Mixture Models. Journal of Statistical Software, 112, 1-39.

[5] Rodriguez, A., & Laio, A. (2014). Clustering by fast search and find of density peaks. Science, 344(6191), 1492-1496.

[6] Wade, S., & Ghahramani, Z. (2018). Bayesian cluster analysis: Point estimation and credible balls (with discussion). Bayesian Analysis, 13(2), 559–626.





























