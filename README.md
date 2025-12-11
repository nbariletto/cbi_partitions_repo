# cbi_partitions: Conformal Bayesian Inference for Partitions

**cbi_partitions** is a Python library for performing Conformal Bayesian Inference (CBI) on partition-valued data (clustering results). It provides valid uncertainty quantification and hypothesis testing for clustering analysis, specifically addressing challenges with multimodal posteriors.

## Installation

You can install this package directly from GitHub into any Python environment (no Git required):

```bash
pip install https://github.com/nbariletto/cbi_partitions/archive/main.zip
```

*Note: The tutorial below requires the `pyrichlet` library for the MCMC sampling step. If you do not have access to this library, you can substitute the sampling step with any standard Bayesian clustering sampler (e.g., Dirichlet Process Mixture Models).*

---

## Tutorial: Reproducing the Multimodal Experiment

This tutorial provides a step-by-step reproduction of the experiment described in the paper. We simulate a dataset with ambiguous clustering structure, sample the posterior using a Pitman-Yor Process Mixture Model, and use `cbi_partitions` to quantify uncertainty and detect multimodality.

### 1. Data Simulation (Gaussian Mixture)
We generate a dataset of $N=100$ points from a mixture of 3 Gaussian components. The covariance is set to create overlap, inducing posterior uncertainty.

```python
# --- MCMC Parameters ---
mcmc_config = {
    'n_final_samples': 6000,
    'burn_in': 1000,
    'thinning': 5,
    'alpha': 0.03,
    'py_sigma': 0.01,
}

print("--- Running Pyrichlet MCMC ---")
total_iter = mcmc_config['burn_in'] + (mcmc_config['n_final_samples'] * mcmc_config['thinning'])
p_dim = config['p_dim']

# Initialize Sampler
mm = mixture_models.PitmanYorMixture(
    alpha=mcmc_config['alpha'], 
    pyd=mcmc_config['py_sigma'],
    mu_prior=X.mean(axis=0), 
    lambda_prior=0.01,
    psi_prior=np.eye(p_dim) * 1.5, 
    nu_prior=p_dim + 2,
    rng=config['seed'], 
    total_iter=total_iter,
    burn_in=mcmc_config['burn_in'], 
    subsample_steps=mcmc_config['thinning']
)

# Run Sampler
mm.fit_gibbs(y=X, show_progress=False)
print("--- Done ---")

# Extract partitions
mcmc_partitions = [samp['d'] for samp in mm.sim_params]
partitions = np.array(mcmc_partitions, dtype=np.int64)
```

![Ground Truth](images/true_partition.png)

### 2. MCMC Sampling (Pitman-Yor Process)
We use a Pitman-Yor Process Mixture Model to sample from the posterior. The parameters $\alpha=0.03$ and $\sigma=0.01$ are chosen to allow for flexibility in the number of clusters.

```python
# --- MCMC Parameters ---
mcmc_config = {
    'n_final_samples': 6000,
    'burn_in': 1000,
    'thinning': 5,
    'alpha': 0.03,
    'py_sigma': 0.01,
}

print("--- Running Pyrichlet MCMC ---")
total_iter = mcmc_config['burn_in'] + (mcmc_config['n_final_samples'] * mcmc_config['thinning'])
p_dim = config['p_dim']

# Initialize Sampler
mm = mixture_models.PitmanYorMixture(
    alpha=mcmc_config['alpha'], 
    pyd=mcmc_config['py_sigma'],
    mu_prior=X.mean(axis=0), 
    lambda_prior=0.01,
    psi_prior=np.eye(p_dim) * 1.5, 
    nu_prior=p_dim + 2,
    rng=config['seed'], 
    total_iter=total_iter,
    burn_in=mcmc_config['burn_in'], 
    subsample_steps=mcmc_config['thinning']
)

# Run Sampler
mm.fit_gibbs(y=X, show_progress=True)

# Extract Partitions
mcmc_partitions = [samp['d'] for samp in mm.sim_params]
partitions = np.array(mcmc_partitions, dtype=np.int64)
```

Below we plot the posterior distribution of the number of clusters implied by our model.
```
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
![Posterior K Distribution](images/posterior_k.png)

### 3. Conformal Model Initialization
We split the MCMC samples into a **Training Set** (5000 partitions) to estimate the partition density and a **Calibration Set** (1000 partitions) to compute non-conformity scores. We use the **PartitionKDE** model with the Variation of Information (VI) metric.

```python
np.random.seed(config['seed'])
indices = np.arange(partitions.shape[0])
np.random.shuffle(indices)

split_idx = int(len(partitions) * 5/6)
train_partitions = partitions[indices[:split_idx]]
calib_partitions = partitions[indices[split_idx:]]

# --- Initialize KDE Model ---
kde = PartitionKDE(
    train_partitions=train_partitions,
    metric='vi',
    gamma=0.5
)

# --- Compute all quantities needed for CBI ---
print("Calibrating KDE model...")
kde.calibrate(calib_partitions)
```

### 4. Pseudo-MAP point estimate

Given the calibration scores we just computed, we compute the point estimate as the calibration partition with highest pseudo-density score. This is done using the .get_point_estimate() method.

```
point_est_partition = kde.get_point_estimate()

plt.figure(figsize=(6, 4))
plt.scatter(X[:,0], X[:,1], c=point_est_partition, cmap='brg', edgecolor='k', s=50)
plt.show()
```


### 5. Detecting Multimodality (DPC)
We use Density Peak Clustering (DPC) to visualize the posterior landscape and identify distinct modes.

```python
kde.plot_dpc_decision_graph()
plt.show()
```

There are clearly two modes, corresponding to the two points that stand out as having both a large $\delta$ and $s$ value. Below we plot the corresponding partitions.

```
modes_idx = kde.get_dpc_modes(s_thresh=0.75, delta_thresh=0.6)

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
for i in range(2):
    axs[i].scatter(X[:,0], X[:,1], c=calib_partitions[modes_idx[i]], cmap='brg', edgecolor='k', s=50)
plt.show()
```

![DPC Decision Graph](images/dpc_decision_graph.png)

The presence of multiple peaks in the decision graph (high density, high separation) confirms the posterior is multimodal.

![DPC Modes](images/dpc_modes.png)

### 6. Hypothesis Testing
We test three specific clustering hypotheses to see if they are consistent with the data at a significance level of $\alpha=0.1$ (90% confidence).

1.  **Collapsed (K=2):** Merging the two closest ground-truth clusters (0 and 1).
2.  **X1-Split (K=2):** Splitting the data purely based on the X1 coordinate ($X_1 > 2.5$).
3.  **Split-Collapsed (K=3):** A hybrid partition that merges clusters 0 and 1, but splits off a new group based on X2.

```python
ALPHA_CONF = 0.1

# --- Define Test Partitions ---

# 1. Collapsed Partition (Merge clusters 0 and 1)
collapsed_labels = true_labels.copy()
collapsed_labels[collapsed_labels == 1] = 0 

# 2. X1-Split Partition (Vertical split)
x1_split_labels = (X[:, 0] > 2.5).astype(np.int64)

# 3. Split-Collapsed Partition (Merge 0+1, but split top corner)
split_collapsed_labels = collapsed_labels.copy()
mask = (split_collapsed_labels == 0) & (X[:, 1] >= 3.2)
split_collapsed_labels[mask] = 2

# --- Run Conformal Tests ---
p_val_true = kde.compute_p_value(true_labels)
p_val_coll = kde.compute_p_value(collapsed_labels)
p_val_x1 = kde.compute_p_value(x1_split_labels)
p_val_split_coll = kde.compute_p_value(split_collapsed_labels)

# --- Output Results ---
print(f"Ground Truth (K=3) p-value:    {p_val_true:.4f}")
print(f"Collapsed (K=2) p-value:       {p_val_coll:.4f}")
print(f"X1-Split (K=2) p-value:        {p_val_x1:.4f}")
print(f"Split-Collapsed (K=3) p-value: {p_val_split_coll:.4f}")
```

### 6. Comparison with PartitionBall
We also compare the results using `PartitionBall`, a simpler conformal model based on fixed-radius balls around the point estimate (rather than density estimation).

```python
# Initialize and Calibrate
ball_model = PartitionBall(
    point_estimate_partition=point_est_partition,
    metric='vi'
)
ball_model.calibrate(calib_partitions)

# Test 'X1-Split' and 'Split-Collapsed'
p_val_x1_ball = ball_model.compute_p_value(x1_split_labels)
p_val_split_coll_ball = ball_model.compute_p_value(split_collapsed_labels)

print(f"X1-Split Ball p-value:       {p_val_x1_ball:.4f}")
print(f"Split-Collapsed Ball p-value:{p_val_split_coll_ball:.4f}")
```

### Visualizing Results
The figure below visualizes the partitions we tested and their resulting acceptance/rejection status.

![Test Partition Results](images/test_partitions_results.png)




