"""
A Python library for Conformalized Bayesian Inference (CBI) with posterior distributions over data partitions.
Contains:
1. Numba-jitted partition distance functions (VI, Binder).
2. PartitionKDE: Standard pipeline for CBI.
3. PartitionBall: CBI implementation of metric credible balls.
"""

import numpy as np
import time
from numba import njit, prange
import matplotlib.pyplot as plt

# ==============================================
# 1. NUMBA-JITTED DISTANCE HELPERS
# ==============================================

@njit(cache=True)
def _remap_labels(p):
    """
    Remaps arbitrary labels to 0..K-1 range efficiently.
    Useful if labels are large integers which would
    cause memory issues in contingency tables.
    """
    n = p.shape[0]
    if n == 0:
        return p, 0
    
    new_p = np.zeros(n, dtype=np.int64)
    sorted_p = np.sort(p)
    
    # Numba-safe empty array initialization
    unique_vals = np.empty(n, dtype=p.dtype)

    if n > 0:
        unique_vals[0] = sorted_p[0]
        k = 1
        for i in range(1, n):
            if sorted_p[i] != sorted_p[i-1]:
                unique_vals[k] = sorted_p[i]
                k += 1
        unique_vals = unique_vals[:k]
    else:
        unique_vals = np.empty(0, dtype=p.dtype)
        k = 0
        
    for i in range(n):
        idx = np.searchsorted(unique_vals, p[i])
        new_p[i] = idx
        
    return new_p, k

@njit(cache=True)
def _contingency_table(p1, p2):
    """
    Calculates the contingency table (co-occurrence matrix).
    Assumes p1 and p2 are 0-indexed and compact for efficiency.
    """
    k1 = 0
    k2 = 0
    if p1.shape[0] > 0:
        k1 = p1.max() + 1
        k2 = p2.max() + 1
    
    contingency = np.zeros((k1, k2), dtype=np.float64)
    n = p1.shape[0]
    
    for i in range(n):
        contingency[p1[i], p2[i]] += 1
        
    return contingency

@njit(cache=True)
def _vi_dist(p1, p2, remap=True):
    """Computes the Variation-of-Information (VI) distance."""
    if remap:
        p1, _ = _remap_labels(p1)
        p2, _ = _remap_labels(p2)

    n = p1.shape[0]
    if n == 0: return 0.0

    contingency = _contingency_table(p1, p2)
    k1, k2 = contingency.shape
    
    C_dot_j = np.sum(contingency, axis=0) 
    C_i_dot = np.sum(contingency, axis=1) 

    h_x_given_y = 0.0
    h_y_given_x = 0.0
    
    log2_n = np.log(2.0)

    for i in range(k1):
        for j in range(k2):
            C_ij = contingency[i, j]
            if C_ij > 0:
                h_x_given_y -= (C_ij / n) * (np.log(C_ij / C_dot_j[j]) / log2_n)
                h_y_given_x -= (C_ij / n) * (np.log(C_ij / C_i_dot[i]) / log2_n)
                
    return h_x_given_y + h_y_given_x

@njit(cache=True)
def _binder_dist(p1, p2, remap=False):
    """Computes the Binder distance."""
    # Note: Binder technically doesn't need remapping for memory safety 
    # since it compares pairs, but we support the flag for consistency.
    if remap:
        p1, _ = _remap_labels(p1)
        p2, _ = _remap_labels(p2)

    n = p1.shape[0]
    if n <= 1: return 0.0
        
    a = 0.0 
    b = 0.0 

    for i in range(n):
        for j in range(i + 1, n):
            in_same_p1 = (p1[i] == p1[j])
            in_same_p2 = (p2[i] == p2[j])
            
            if in_same_p1 and in_same_p2:
                a += 1
            elif (not in_same_p1) and (not in_same_p2):
                b += 1
                
    total_pairs = (n * (n - 1)) / 2.0
    return 0.0 if total_pairs == 0 else 1.0 - (a + b) / total_pairs

@njit(cache=True)
def _calculate_distance(p1, p2, metric_code, remap):
    """
    Internal dispatcher for distance calculation.
    metric_code: 0 = vi, 1 = binder
    """
    if metric_code == 0: # VI
        return _vi_dist(p1, p2, remap)
    elif metric_code == 1: # Binder
        return _binder_dist(p1, p2, remap)
    return 0.0

@njit(parallel=True, cache=True)
def _compute_pairwise_matrix(partitions, metric_code, remap):
    """Computes the symmetric pairwise distance matrix efficiently."""
    n = partitions.shape[0]
    matrix = np.zeros((n, n), dtype=np.float64)
    for i in prange(n):
        for j in range(i + 1, n):
            d = _calculate_distance(partitions[i], partitions[j], metric_code, remap)
            matrix[i, j] = d
            matrix[j, i] = d
    return matrix

@njit(parallel=True, cache=True)
def _pvals(scores, calib_scores):
    n = scores.shape[0]
    m = calib_scores.shape[0]
    out = np.empty(n, dtype=np.float64)

    for i in prange(n):
        c = 0
        s = scores[i]
        for j in range(m):
            if calib_scores[j] <= s:
                c += 1
        out[i] = (c + 1) / (m + 1)

    return out


# =======================
# 2. KDE CBI PIPELINE
# =======================

class PartitionKDE:
    """Metric-KDE for CBI."""
    
    def __init__(self, train_partitions, metric='vi', gamma=0.5, subsample_size=None, remap_labels=True):
        if metric not in ['vi', 'binder']:
            raise ValueError("Metric must be 'vi' or 'binder'")
        
        self.metric_ = metric
        self.metric_code_ = 0 if metric == 'vi' else 1
        self.gamma_ = gamma
        self.remap_labels_ = remap_labels
        
        # Explicitly convert input list to numpy array
        train_partitions = np.array(train_partitions, dtype=np.int64)

        if subsample_size and subsample_size < len(train_partitions):
            print(f"Subsampling training set to {subsample_size} samples.")
            indices = np.random.choice(len(train_partitions), subsample_size, replace=False)
            self.train_partitions_ = train_partitions[indices].copy()
        else:
            self.train_partitions_ = train_partitions.copy()
            
        self.n_train_ = self.train_partitions_.shape[0]
        self.n_nodes_ = self.train_partitions_.shape[1]
        print(f"PartitionKDE initialized with {self.n_train_} train samples. Remap={self.remap_labels_}")

    @staticmethod
    @njit(parallel=True, cache=True)
    def _score_kde_batch(partitions, train_partitions, metric_code, gamma, remap):
        """Numba-jitted KDE score calculation."""
        n_partitions = partitions.shape[0]
        n_train = train_partitions.shape[0]
        scores = np.zeros(n_partitions, dtype=np.float64)
        
        for i in prange(n_partitions):
            p_new = partitions[i]
            total_kernel = 0.0
            for j in range(n_train):
                p_train = train_partitions[j]
                dist = _calculate_distance(p_new, p_train, metric_code, remap)
                total_kernel += np.exp(-gamma * dist)
            scores[i] = total_kernel / n_train
        return scores

    def score(self, partitions):
        """Computes the KDE score for one or more partitions."""
        partitions = np.array(partitions, dtype=np.int64)
        if partitions.ndim == 1: 
            partitions = partitions.reshape(1, -1)
        return self._score_kde_batch(partitions, self.train_partitions_, self.metric_code_, self.gamma_, self.remap_labels_)

    def calibrate(self, calib_partitions):
        """Scores all calibration partitions and computes DPC variables."""
        self.calib_partitions_ = np.array(calib_partitions, dtype=np.int64)
        self.n_calib_ = self.calib_partitions_.shape[0]
        
        print(f"Scoring {self.n_calib_} calibration samples...")
        self.calib_scores_ = self.score(self.calib_partitions_)

        # Compute DPC variables
        self._compute_dpc_vars()

    def compute_p_value(self, partitions):
        """Computes the conformal p-value for one or more partitions."""
        if not hasattr(self, 'calib_scores_'):
            raise RuntimeError("Must call .calibrate() before .compute_p_value()")
    
        p = np.array(partitions, dtype=np.int64)
    
        if p.ndim == 1:
            s = self.score(p)[0]
            return (np.sum(self.calib_scores_ <= s) + 1) / (self.n_calib_ + 1)
    
        scores = self.score(p)
        return self._pvals(scores, self.calib_scores_)


    def get_point_estimate(self, source='calibration'):
        """
        Finds the partition with the highest KDE score.
        
        Parameters
        ----------
        source : str, 'train' or 'calibration'
            Which dataset to select the point estimate from.
        """
        if source == 'train':
            if not hasattr(self, 'train_scores_'):
                self.train_scores_ = self.score(self.train_partitions_)
            best_idx = np.argmax(self.train_scores_)
            return self.train_partitions_[best_idx]

        elif source == 'calibration':
            if not hasattr(self, 'calib_scores_'):
                raise RuntimeError("Calibration scores not found. You must run .calibrate() first.")
            best_idx = np.argmax(self.calib_scores_)
            return self.calib_partitions_[best_idx]
        
        else:
            raise ValueError("source must be 'train' or 'calibration'")

    def _compute_dpc_vars(self):
            """Computes DPC variables (s and delta, normalized) using efficient JIT compilation."""
            print("Computing DPC variables...")
            
            # 1. Pairwise distances
            self.calib_dist_matrix_ = _compute_pairwise_matrix(
                self.calib_partitions_, self.metric_code_, self.remap_labels_
            )
            
            # 3. Delta (distance to nearest point with higher density)
            self.dpc_delta_ = np.zeros(self.n_calib_)
            sorted_indices = np.argsort(self.calib_scores_)[::-1] 
                    
            for i, idx_i in enumerate(sorted_indices):
                if i == 0:
                    if self.n_calib_ > 0:
                        self.dpc_delta_[idx_i] = self.calib_dist_matrix_[idx_i].max()
                    else:
                        self.dpc_delta_[idx_i] = 1.0
                    continue
                
                denser_indices = sorted_indices[:i]
                dists_to_denser = self.calib_dist_matrix_[idx_i, denser_indices]
                self.dpc_delta_[idx_i] = dists_to_denser.min()
                
            self.dpc_gamma_ = self.calib_scores_ * self.dpc_delta_

    def plot_dpc_decision_graph(self, save_path=None):
        """Plots the DPC decision graph using internal variables."""
        if not hasattr(self, 'calib_scores_'):
            raise RuntimeError("DPC variables not computed. Run .calibrate() first.")
            
        plt.figure(figsize=(4, 3))
        plt.scatter(self.calib_scores_, self.dpc_delta_, alpha=0.6, c='blue', edgecolors='k')
        plt.xlabel(r'Density ($s$)')
        plt.ylabel(r'Dist. to higher density samples ($\delta$)')
        plt.grid(True, linestyle='--', alpha=0.7)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Saved decision graph to {save_path}")
        plt.show()

    def get_dpc_modes(self, s_thresh, delta_thresh):
        """Identifies mode indexes based on s and delta thresholds."""
        if not hasattr(self, 'calib_scores_'):
            raise RuntimeError("DPC variables not computed. Run .calibrate() first.")
            
        candidates = np.where((self.calib_scores_ > s_thresh) & (self.dpc_delta_ > delta_thresh))[0]
        # Sort by gamma = s * delta
        if len(candidates) > 0:
            gammas = self.dpc_gamma_[candidates]
            sorted_idx = np.argsort(gammas)[::-1]
            return candidates[sorted_idx]
        return np.array([], dtype=np.int64)


# ===================================
# 3. DISTANCE-BASED CBI PIPELINE 
# ===================================

class PartitionBall:
    """CBI instantiation of metric ball credible sets."""
    def __init__(self, point_estimate_partition, metric='vi', remap_labels=True):
        if metric not in ['vi', 'binder']: raise ValueError("Metric must be 'vi' or 'binder'")
        self.metric_ = metric
        self.metric_code_ = 0 if metric == 'vi' else 1
        self.remap_labels_ = remap_labels
        
        self.point_estimate_ = np.array(point_estimate_partition, dtype=np.int64)
        self.n_nodes_ = self.point_estimate_.shape[0]
        print(f"PartitionBall initialized with metric: {self.metric_} (Remap={self.remap_labels_})")

    @staticmethod
    @njit(parallel=True, cache=True)
    def _score_distance_batch(partitions, center_partition, metric_code, remap):
        n_partitions = partitions.shape[0]
        scores = np.zeros(n_partitions, dtype=np.float64)
        for i in prange(n_partitions):
            scores[i] = _calculate_distance(partitions[i], center_partition, metric_code, remap)
        return scores

    def score(self, partitions):
        partitions = np.array(partitions, dtype=np.int64)
        if partitions.ndim == 1: partitions = partitions.reshape(1, -1)
        return self._score_distance_batch(partitions, self.point_estimate_, self.metric_code_, self.remap_labels_)

    def calibrate(self, calib_partitions):
        self.calib_partitions_ = np.array(calib_partitions, dtype=np.int64)
        self.n_calib_ = self.calib_partitions_.shape[0]
        print(f"Scoring {self.n_calib_} calibration samples (distance to center)...")
        self.calib_scores_ = self.score(self.calib_partitions_)

    def compute_p_value(self, partition):
        """Computes non-conformity p-value (higher score = worse)."""
        if not hasattr(self, 'calib_scores_'): raise RuntimeError("Must call .calibrate() first")
        new_score = self.score(partition)[0]
        # p-value = Fraction of scores >= new_score (worse or equal)
        p_val = (np.sum(self.calib_scores_ >= new_score) + 1) / (self.n_calib_ + 1)
        return p_val







