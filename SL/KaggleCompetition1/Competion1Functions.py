from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from scipy.stats import gaussian_kde
from collections import defaultdict
from scipy.stats import norm

# Plot KDE for a feature split by class

def plot_kde_feature(X, y, feature_index):
    plt.figure(figsize=(6, 4))
    for cls in np.unique(y):
        sns.kdeplot(X[y == cls, feature_index], label=f"Class {cls}")
    plt.title(f"KDE of Feature {feature_index}")
    plt.legend()
    plt.show()


# Histogram Naive Bayes Classifier for discrete features
# This is a simple implementation of Naive Bayes using histograms for density estimation.

class HistogramNaiveBayes:
    def __init__(self, bins=10, smoothing=1e-3):
        self.bins = bins
        self.smoothing = smoothing
        self.histograms = defaultdict(dict)
        self.class_priors = {}
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        n = len(y)

        for cls in self.classes:
            X_cls = X[y == cls]
            self.class_priors[cls] = len(X_cls) / n

            for i in range(X.shape[1]):
                hist, bin_edges = np.histogram(X_cls[:, i], bins=self.bins, density=False)
                # Add smoothing
                self.histograms[cls][i] = (hist + self.smoothing, bin_edges)

    def predict_proba(self, X):
        results = []
        for x in X:
            log_probs = {}
            for cls in self.classes:
                log_prob = np.log(self.class_priors[cls])
                for i, val in enumerate(x):
                    hist, bin_edges = self.histograms[cls][i]
                    bin_idx = np.digitize(val, bin_edges) - 1
                    bin_idx = np.clip(bin_idx, 0, len(hist) - 1)
                    prob = hist[bin_idx] / np.sum(hist)
                    log_prob += np.log(prob + 1e-9)
                log_probs[cls] = log_prob
            # Normalize via softmax trick
            max_log = max(log_probs.values())
            exp_scores = {k: np.exp(v - max_log) for k, v in log_probs.items()}
            total = sum(exp_scores.values())
            results.append({k: v / total for k, v in exp_scores.items()})
        return results

    def predict(self, X):
        proba = self.predict_proba(X)
        return [max(p, key=p.get) for p in proba]

# Model NonParametricNaiveBayes with tunable KDE bandwidth and option to choose from Gaussian or Epanechnikov kernels.

# Custom Epanechnikov KDE
def epanechnikov_kernel(u):
    return 0.75 * (1 - u**2) * (np.abs(u) <= 1) # Epanechnikov kernel function

class EpanechnikovKDE:
    def __init__(self, data, bandwidth=1.0):
        self.data = np.asarray(data)
        self.bandwidth = bandwidth
        self.n = len(data)

    def evaluate(self, x):
        x = np.atleast_1d(x) # Ensure x is an array
        estimates = np.zeros_like(x, dtype=float) # Initialize estimates
        # Iterate over each point in x
        for i, xi in enumerate(x):  
            u = (xi - self.data) / self.bandwidth # Calculate the scaled distance
            kernels = epanechnikov_kernel(u) # Apply the Epanechnikov kernel
            estimates[i] = np.sum(kernels) # Sum the kernel values
        return estimates / (self.n * self.bandwidth) # Normalize by n and bandwidth
 
class NonParametricNaiveBayes: 
    def __init__(self, bandwidth=1.0, kernel='gaussian'): # Gaussian KDE bandwidth (for later tuning)
        """
        Non-parametric Naive Bayes classifier using Gaussian KDE for density estimation.
        Parameters:
            - bandwidth: smoothing parameter for KDE.
            - kernel: 'gaussian' or 'epanechnikov'.
        """
        self.bandwidth = bandwidth 
        self.kernel = kernel.lower()
        self.kdes = defaultdict(dict) # KDEs for each class and feature
        self.class_priors = {} 
        self.classes = None # Class labels

    def _create_kde(self, data):
        if self.kernel == 'gaussian':
            return gaussian_kde(data, bw_method=self.bandwidth)
        elif self.kernel == 'epanechnikov':
            return EpanechnikovKDE(data, bandwidth=self.bandwidth)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")

    def fit(self, X, y):
        self.classes = np.unique(y) # Unique class labels
        n = len(y) # Total number of samples
        
        for cls in self.classes: # Iterate over each class
            X_cls = X[y == cls] # Samples for the current class
            self.class_priors[cls] = len(X_cls) / n # Class prior probability

            # Fit KDE for each feature in the class
            for feature_idx in range(X.shape[1]): 
                kde = self._create_kde(X_cls[:, feature_idx]) # Apply KDE
                self.kdes[cls][feature_idx] = kde # Store 

    def predict_proba(self, X): # Predict class probabilities for each sample
        probs = [] # Store in list
        for x in X: # Iterate over each sample
            # Calculate log probabilities for each class
            class_scores = {} # Store log probabilities

            # Iterate over each class and calculate the log probability
            for cls in self.classes:
                log_prob = np.log(self.class_priors[cls]) # Start with the log prior

                # Iterate over each feature and calculate the log probability
                for feature_idx, val in enumerate(x):
                    kde = self.kdes[cls][feature_idx] # Get the KDE for the current class and feature
                    p = kde.evaluate(val)[0] + 1e-9 # Avoid log(0) by adding a small value
                    log_prob += np.log(p) # Log probability of the feature given the class
                class_scores[cls] = log_prob # Store the log probability for the current class
            # Softmax trick
            max_log = max(class_scores.values()) # For numerical stability
            exp_scores = {k: np.exp(v - max_log) for k, v in class_scores.items()} # Exponentiate the log probabilities
            total = sum(exp_scores.values()) # Sum of exponentiated scores
            probs.append({k: v / total for k, v in exp_scores.items()}) # Normalize to get probabilities
        return probs

    def predict(self, X):
        proba = self.predict_proba(X) # Get class probabilities
        return [max(p, key=p.get) for p in proba] # Return the class with the highest probability
    
def tune_bandwidth(X, y, bandwidths, kernel='gaussian'):
    """
    Tune the bandwidth hyperparameter for NonParametricNaiveBayes.

    Parameters:
    - X, y: training data and labels
    - bandwidths: iterable of bandwidth values to try
    - kernel: 'gaussian' or 'epanechnikov'
    """

    skf = StratifiedKFold(n_splits=3)
    best_bw = None
    best_acc = 0

    for bw in bandwidths:
        accs = []
        for train_idx, val_idx in skf.split(X, y):
            model = NonParametricNaiveBayes(bandwidth=bw, kernel=kernel)
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[val_idx])
            accs.append(accuracy_score(y[val_idx], preds))
        avg = np.mean(accs)
        print(f"Kernel={kernel:<12} | Bandwidth={bw:.2f} -> CV Accuracy: {avg:.4f}")
        if avg > best_acc:
            best_acc = avg
            best_bw = bw

    print(f"\n✅ Best Bandwidth for {kernel.upper()}: {best_bw}")
    return best_bw

def epanechnikov_kernel(u):
    return 0.75 * (1 - u**2) * (np.abs(u) <= 1)

def gaussian_kernel(u):
    return norm.pdf(u)
