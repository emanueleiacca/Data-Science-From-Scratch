import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic dataset for binary classification
def generate_classification_data(samples=200, seed=42):
    """
    Generate a synthetic dataset for binary classification.
    """
    np.random.seed(seed)
    
    # Class 0: Cluster centered at (2, 2)
    x0 = np.random.randn(samples // 2, 2) + np.array([2, 2])
    y0 = np.zeros(samples // 2)
    
    # Class 1: Cluster centered at (6, 6)
    x1 = np.random.randn(samples // 2, 2) + np.array([6, 6])
    y1 = np.ones(samples // 2)
    
    # Combine both classes
    X = np.vstack((x0, x1))
    y = np.hstack((y0, y1))
    
    return X, y

# --- Sigmoid Function ---
def sigmoid(z):
    """
    Compute the sigmoid function.
    """
    # Clip values to avoid overflow # a prediction of 0 or 1 will cause log(0) or log(1-0) which is undefined
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z)) # formula

# --- Hypothesis Function ---

# Difference with the one implemented in Regression.py:
# Linear Regression use a linear hypothesis function.
# Logistic Regression use a sigmoid function to output probabilities.

def hypothesis(theta, X):
    """
    Logistic regression hypothesis function.
    """
    return sigmoid(np.dot(X, theta)) # formula

# --- Cost Function ---

# Difference with the one implemented in Regression.py:
# Linear Regression use MSE for continuous predictions.
# Logistic Regression use Binary Cross-Entropy (Log-Loss) for binary classification.

def cost_function(theta, X, y):
    """
    Compute the cost for logistic regression.
    """
    m = len(y)  # Number of samples
    h = hypothesis(theta, X)
    
    # Avoid log(0) errors
    h = np.clip(h, 1e-5, 1 - 1e-5)
    
    cost = -(1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost

# --- Gradient Descent ---

# Similar formula, but here, ℎ𝜃(𝑥) represents probabilities (output of the sigmoid function).
# The updates are scaled by the difference between predicted probabilities and actual labels.

def gradient_descent(X, y, theta, learning_rate, iterations):
    """
    Perform gradient descent for logistic regression.
    """
    m = len(y)  # Number of samples
    cost_history = []
    # Parameters for adapting learning rate
    increase_factor = 1.1
    decrease_factor = 0.5
    patience = 3  # Number of iterations to wait before increasing
    good_steps = 0

    for i in range(iterations):
        predictions = hypothesis(theta, X) 
        gradient = (1 / m) * np.dot(X.T, (predictions - y)) # Formula
        
        # Update parameters
        theta -= learning_rate * gradient 
        
        # Calculate cost and store it
        cost = cost_function(theta, X, y)
        cost_history.append(cost)

        # Adapt learning rate
        if i > 0:
            if cost < cost_history[-2]:  # Cost decreased
                good_steps += 1
                if good_steps >= patience:
                    learning_rate *= increase_factor  # Gradually increase learning rate
                    good_steps = 0  # Reset counter
            else:  # Cost increased
                learning_rate *= decrease_factor  # Quickly decrease learning rate
                good_steps = 0  # Reset counter

        """
        Initial settings:
        learning_rate = 0.1

        Iteration 1:
        cost = 0.7

        Iteration 2:
        cost = 0.6

        Iteration 3:
        cost = 0.5

        Iteration 4:
        cost = 0.45
        # Cost decreased again (0.45 < 0.5)
        good_steps = 3  # Now we've reached our patience threshold!
        # Since good_steps (3) >= patience (3), we increase learning rate
        learning_rate = 0.1 * 1.1 = 0.11
        good_steps = 0  # Reset counter after adjusting learning rate

        Iteration 5:
        cost = 0.47
        # Cost increased (0.47 > 0.45)! We took too big a step
        learning_rate = 0.11 * 0.5 = 0.055  # Quickly decrease learning rate
        """
        # Debug every 100 iterations
        if i % 100 == 0:
            print(f"Iteration {i}: Cost={cost:.4f}")
            # Basic learning rate adaptation
            
    return theta, cost_history

# --- Linear Decision Boundary ---
def predict(theta, X):
    """
    Make predictions using the learned parameters.
    """
    probabilities = hypothesis(theta, X)
    return (probabilities >= 0.5).astype(int) # Convert probabilities to binary predictions


def generate_nonlinear_data(n_samples=100, noise=0.1):
    """
    Generate a dataset that's not linearly separable (e.g., circular pattern)
    """
    np.random.seed(42)
    
    # Generate radius and angle for points
    r1 = np.random.normal(2, noise, n_samples//2)
    r2 = np.random.normal(4, noise, n_samples//2)
    theta = np.random.uniform(0, 2*np.pi, n_samples)
    
    # Generate inner circle (class 0)
    X1 = np.column_stack([
        r1 * np.cos(theta[:n_samples//2]),
        r1 * np.sin(theta[:n_samples//2])
    ])
    y1 = np.zeros(n_samples//2)
    
    # Generate outer circle (class 1)
    X2 = np.column_stack([
        r2 * np.cos(theta[n_samples//2:]),
        r2 * np.sin(theta[n_samples//2:])
    ])
    y2 = np.ones(n_samples//2)
    
    # Combine the datasets
    X = np.vstack([X1, X2])
    y = np.hstack([y1, y2])
    
    return X, y

def add_polynomial_features(X, degree=2):
    """
    Transform features into polynomial features up to specified degree.
    For example, if X has features [x₁, x₂] and degree=2, this will return:
    [1, x₁, x₂, x₁², x₁x₂, x₂²]
    """
    n_samples, n_features = X.shape
    combined_features = [np.ones(n_samples)]  # Add bias term
    
    # Add original features
    for i in range(n_features):
        combined_features.append(X[:, i])
    
    # Add polynomial terms
    for d in range(2, degree + 1):
        for i in range(n_features):
            for j in range(i, n_features):
                # If i == j, this adds x_i^d
                # If i != j, this adds x_i * x_j combinations
                term = X[:, i] * X[:, j]
                if i == j:
                    term = X[:, i] ** d
                combined_features.append(term)
    
    return np.column_stack(combined_features)

def plot_nonlinear_decision_boundary(X, y, theta, poly_degree):
    """
    Plot the non-linear decision boundary
    """
    plt.figure(figsize=(10, 8))
    
    # Plot the data points
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label='Class 0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label='Class 1')
    
    # Create a grid of points
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    # Create polynomial features for grid points
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_poly = add_polynomial_features(grid_points, poly_degree)
    
    # Make predictions on grid points
    Z = hypothesis(theta, grid_poly)
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', label='Decision Boundary')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'Non-linear Decision Boundary (Polynomial Degree {poly_degree})')
    plt.legend()
    plt.show()

def generate_bean_data(n_samples=300, noise=0.1):
    """
    Generate a dataset with a bean-shaped pattern.
    The inner class (0) forms a bean shape, surrounded by class 1.
    """
    # Generate basic circular pattern
    np.random.seed(42)
    t = np.random.uniform(0, 2*np.pi, n_samples//2)
    
    # Create bean shape for class 0
    r = 2 + np.sin(t)  # Varying radius creates bean shape
    x1_inner = r * np.cos(t)
    x2_inner = r * np.sin(t)
    x1_inner += np.random.normal(0, noise, len(t))
    x2_inner += np.random.normal(0, noise, len(t))
    y_inner = np.zeros(len(t))
    
    # Create surrounding points for class 1
    t_outer = np.random.uniform(0, 2*np.pi, n_samples//2)
    r_outer = 4 + np.random.normal(0, 0.5, len(t_outer))
    x1_outer = r_outer * np.cos(t_outer)
    x2_outer = r_outer * np.sin(t_outer)
    y_outer = np.ones(len(t_outer))
    
    # Combine datasets
    X = np.vstack([np.column_stack([x1_inner, x2_inner]),
                   np.column_stack([x1_outer, x2_outer])])
    y = np.hstack([y_inner, y_outer])
    
    return X, y

def create_bean_features(X):
    """
    Transform features to create bean-shaped decision boundaries.
    Includes up to cubic terms and interaction terms.
    """
    x1, x2 = X[:, 0], X[:, 1]
    
    # Create array of transformed features
    features = np.column_stack([
        np.ones(len(x1)),      # Bias term
        x1, x2,                # Linear terms
        x1**2, x2**2,         # Quadratic terms
        x1*x2,                # First-order interaction
        x1**3, x2**3,         # Cubic terms
        x1**2*x2, x1*x2**2    # Higher-order interactions
    ])
    
    return features

def plot_bean_decision_boundary(X, y, theta, scaler):
    """
    Plot the data points and the bean-shaped decision boundary.
    """
    plt.figure(figsize=(12, 8))
    
    # Plot the original data points
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], 
               color='blue', label='Class 0', alpha=0.6)
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], 
               color='red', label='Class 1', alpha=0.6)
    
    # Create a grid of points
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    # Transform grid points
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_points_scaled = scaler.transform(grid_points)
    grid_poly = create_bean_features(grid_points_scaled)
    
    # Make predictions on grid points
    Z = sigmoid(np.dot(grid_poly, theta))
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and contours
    plt.contour(xx, yy, Z, levels=[0.5], colors='k', label='Decision Boundary')
    plt.contourf(xx, yy, Z, alpha=0.2, levels=np.linspace(0, 1, 20))
    
    plt.colorbar(label='Probability of Class 1')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Bean-Shaped Decision Boundary')
    plt.legend()
    plt.show()

def calculate_accuracy_from_model(theta, X, y):
    """
    Calculate accuracy of the logistic regression model.
    """
    probabilities = sigmoid(np.dot(X, theta))
    predictions = (probabilities >= 0.5).astype(int)  # Apply threshold
    
    accuracy = np.mean(predictions == y) * 100  # Calculate accuracy as a percentage
    return accuracy

def compute_nll(theta, X, y):
    """Compute the Negative Log-Likelihood."""
    m = len(y)
    predictions = sigmoid(np.dot(X, theta))
    predictions = np.clip(predictions, 1e-15, 1 - 1e-15)  # Prevent log(0)
    nll = -(1 / m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    return nll


def gradient_descent_NLL(X, y, theta, alpha=0.01, iterations=1000):
    """Perform Gradient Descent to minimize NLL."""
    m = len(y)
    cost_history = []

    for i in range(iterations):
        predictions = sigmoid(np.dot(X, theta))
        gradient = (1 / m) * np.dot(X.T, (y - predictions))
        theta += alpha * gradient  # Gradient Descent Update
        
        cost = compute_nll(theta, X, y)
        cost_history.append(cost)
        
        if i % 100 == 0:
            print(f"Iteration {i}: NLL = {cost:.4f}")
    
    return theta, cost_history

def plot_decision_boundary_2d(X, y, theta):
    """
    Plot decision boundary for logistic regression with two features.
    """
    plt.figure(figsize=(12, 8))
    
    # Scatter plot of data points
    plt.scatter(X[y == 0][:, 1], X[y == 0][:, 2], color='blue', label='Class 0')
    plt.scatter(X[y == 1][:, 1], X[y == 1][:, 2], color='red', label='Class 1')
    
    # Plot Decision Boundary
    x1_values = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    x2_values = -(theta[0] + theta[1] * x1_values) / theta[2]
    
    plt.plot(x1_values, x2_values, label='Decision Boundary', color='black')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Logistic Regression Decision Boundary on 2D Linear Data')
    plt.legend()
    plt.show()

# Gradient of NLL
def gradient_nll(theta, X, y):
    predictions = sigmoid(np.dot(X, theta))
    return -(1 / len(y)) * np.dot(X.T, (y - predictions))

# Hessian of NLL
def hessian_nll(theta, X):
    predictions = sigmoid(np.dot(X, theta))
    diag = predictions * (1 - predictions)
    D = np.diag(diag)
    return (1 / len(X)) * np.dot(X.T, np.dot(D, X))

# Newton's Method for Logistic Regression
def newtons_method(X, y, theta, tol=1e-6, max_iter=100):
    for i in range(max_iter):
        grad = gradient_nll(theta, X, y)  # Compute gradient
        hessian = hessian_nll(theta, X)  # Compute Hessian
        
        # Update theta using Newton's Method
        hessian_inv = np.linalg.inv(hessian + np.eye(hessian.shape[0]) * 1e-4)  # Regularize Hessian
        theta_update = np.dot(hessian_inv, grad)
        
        theta -= theta_update  # Newton's Update
        
        # Interpret velocity and acceleration
        velocity = np.linalg.norm(theta_update)
        acceleration = np.linalg.norm(np.dot(hessian, theta_update))
        
        print(f"Iteration {i+1}: Velocity = {velocity:.4f}, Acceleration = {acceleration:.4f}")
        
        if velocity < tol:
            print("Convergence reached.")
            break
    
    return theta

def plot_newton_decision_boundary(X, y, theta):
    plt.figure(figsize=(12, 8))
    plt.scatter(X[:, 1], X[:, 2], c=y, cmap='bwr', edgecolor='k')
    
    x1_values = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    x2_values = -(theta[0] + theta[1] * x1_values) / theta[2]
    
    plt.plot(x1_values, x2_values, color='black', label='Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Logistic Regression Decision Boundary (Newton\'s Method)')
    plt.legend()
    plt.show()

# BFGS Optimization for Logistic Regression
def bfgs_logistic_regression(X, y, theta_init, tol=1e-6, max_iter=100):
    theta = theta_init
    n_features = len(theta)
    
    # Initialize B as Identity Matrix
    B = np.eye(n_features)
    
    for i in range(max_iter):
        grad = gradient_nll(theta, X, y)
        p = -np.dot(np.linalg.inv(B), grad)  # Direction to move (Newton-like step)
        
        # Line Search (simple learning rate tuning)
        alpha = 0.1  # Fixed step size for simplicity
        theta_new = theta + alpha * p
        
        grad_new = gradient_nll(theta_new, X, y)
        s = theta_new - theta
        y_k = grad_new - grad
        
        # BFGS Update Rule
        if np.dot(s, y_k) > 1e-10:  # Ensure positive definiteness
            Bs = np.dot(B, s)
            B += np.outer(y_k, y_k) / np.dot(y_k, s) - np.outer(Bs, Bs) / np.dot(s, Bs)
        
        velocity = np.linalg.norm(s)
        
        print(f"Iteration {i+1}: NLL = {compute_nll(theta, X, y):.4f}, Velocity = {velocity:.4f}")
        
        if velocity < tol:
            print("Convergence achieved!")
            break
        
        theta = theta_new
    
    return theta

import numpy as np

def softmax(logits):
    """
    Compute the softmax probabilities for a set of logits.
    """
    exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    """
    Compute the cross-entropy loss.
    """
    # Ensure both inputs are NumPy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Validate shapes
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true shape {y_true.shape}, y_pred shape {y_pred.shape}")
    
    # Clip predictions for numerical stability
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    
    # Calculate Cross-Entropy Loss
    loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    return loss

def plot_metric_overlay(X, y_true, y_pred, probs, metric='ce_loss'):
    """
    Plot True and Predicted Labels with Metric Overlay (CE Loss, Entropy, KL-Divergence)
    """
    # ✅ Compute the desired metric for each sample
    if metric == 'ce_loss':
        one_hot_true = np.eye(probs.shape[1])[y_true.flatten()]
        sample_values = -np.sum(one_hot_true * np.log(probs + 1e-15), axis=1)
        metric_label = 'Cross-Entropy Loss'

    else:
        raise ValueError("Invalid metric. Choose 'ce_loss'.")
    
    # ✅ Plot configuration
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    ### 📌 Plot 1: True Labels with Metric Overlay
    scatter1 = axs[0].scatter(
        X[:, 0], X[:, 1],
        c=y_true.flatten(),
        cmap='viridis',
        s=sample_values * 300,  # Scale for better visualization
        alpha=0.7,
        edgecolor='k'
    )
    axs[0].set_title(f'True Labels with {metric_label} Overlay')
    axs[0].set_xlabel('Feature 1')
    axs[0].set_ylabel('Feature 2')
    legend1 = axs[0].legend(*scatter1.legend_elements(), title="Class")
    axs[0].add_artist(legend1)
    cbar1 = plt.colorbar(scatter1, ax=axs[0], label=metric_label)

    ### 📌 Plot 2: Predicted Labels with Metric Overlay
    scatter2 = axs[1].scatter(
        X[:, 0], X[:, 1],
        c=y_pred,
        cmap='viridis',
        s=sample_values * 300,  # Scale for better visualization
        alpha=0.7,
        edgecolor='k'
    )
    axs[1].set_title(f'Predicted Labels with {metric_label} Overlay')
    axs[1].set_xlabel('Feature 1')
    axs[1].set_ylabel('Feature 2')
    legend2 = axs[1].legend(*scatter2.legend_elements(), title="Class")
    axs[1].add_artist(legend2)
    cbar2 = plt.colorbar(scatter2, ax=axs[1], label=metric_label)

    ### 📌 Highlight Misclassified Points
    misclassified = y_true.flatten() != y_pred
    axs[1].scatter(
        X[misclassified, 0],
        X[misclassified, 1],
        edgecolor='red',
        facecolor='none',
        s=100,
        label='Misclassified'
    )

    axs[1].legend()
    plt.tight_layout()
    plt.show()

def entropy(p):
    """
    Calculate the entropy of a probability distribution.
    """
    p = np.asarray(p)
    p = np.clip(p, 1e-15, 1 - 1e-15)  # Avoid log(0)
    return -np.sum(p * np.log(p))

def kl_divergence(p, q):
    """
    Compute KL-Divergence between true distribution p and predicted distribution q.
    Both p and q must be probability distributions.
    """
    p = np.clip(p, 1e-15, 1)  # Avoid log(0)
    q = np.clip(q, 1e-15, 1)  # Avoid log(0)
    return np.sum(p * np.log(p / q), axis=1)

def expected_calibration_error_multiclass(y_true, probs, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE) for multi-class classification.
    """
    n_classes = probs.shape[1]  # Number of classes
    ece_per_class = [] # ECE for each class
    
    for k in range(n_classes): # Iterate over each class
        class_probs = probs[:, k] # Predicted probabilities for class k
        class_true = (y_true == k).astype(int) # True labels for class k
        bins = np.linspace(0, 1, n_bins + 1) # Define bin edges
        bin_indices = np.digitize(class_probs, bins) - 1 # Bin assignment
        
        ece = 0.0 # Initialize ECE for class k
        for i in range(n_bins): # Iterate over each bin
            bin_mask = bin_indices == i # Samples in bin i
            if np.sum(bin_mask) > 0: # Non-empty bin
                bin_accuracy = np.mean(class_true[bin_mask]) # Accuracy in bin
                bin_confidence = np.mean(class_probs[bin_mask]) # Confidence in bin
                ece += (np.sum(bin_mask) / len(y_true)) * np.abs(bin_accuracy - bin_confidence) # Weighted difference
        
        ece_per_class.append(ece) # Append ECE for class k
    
    avg_ece = np.mean(ece_per_class) # Average ECE across classes
    return ece_per_class, avg_ece

def plot_calibration_curve_multiclass(y_true, probs, n_bins=10):
    """
    Plot individual calibration curves for each class with improved clarity.
    """
    n_classes = probs.shape[1]
    bins = np.linspace(0, 1, n_bins + 1)
    
    fig, axs = plt.subplots(n_classes, 1, figsize=(12, 5 * n_classes))
    
    for k in range(n_classes):
        ax = axs[k] if n_classes > 1 else axs
        class_probs = probs[:, k]
        class_true = (y_true == k).astype(int)
        bin_indices = np.digitize(class_probs, bins) - 1
        
        accuracies = []
        confidences = []
        bin_centers = []
        
        for i in range(n_bins):
            bin_mask = bin_indices == i
            if np.sum(bin_mask) > 0:
                bin_accuracy = np.mean(class_true[bin_mask])
                bin_confidence = np.mean(class_probs[bin_mask])
                accuracies.append(bin_accuracy)
                confidences.append(bin_confidence)
                bin_centers.append((bins[i] + bins[i + 1]) / 2)
        
        # Plot each class calibration curve
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        ax.plot(bin_centers, accuracies, marker='o', color='b', label='Accuracy')
        ax.bar(bin_centers, accuracies, width=0.1, alpha=0.3, edgecolor='black', label='Bin Accuracy')
        ax.set_title(f'Calibration Curve for Class {k}')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

def generate_nonlinear_squared_data(n_samples=200, noise=0.1):
    """Generate a non-linear dataset with Gaussian noise."""
    np.random.seed(42)
    X = np.linspace(-3, 3, n_samples).reshape(-1, 1)
    y = np.sin(X).flatten() + np.random.normal(0, noise, n_samples)
    return X, y

def add_polynomial_squared_features(X, degree):
    """Expand input features into polynomial features."""
    return np.hstack([X**i for i in range(1, degree+1)])

# Polynomial Regression
def polynomial_regression(X, y, degree):
    """Fit a polynomial regression model."""
    X_poly = add_polynomial_squared_features(X, degree)
    coeffs = np.linalg.pinv(X_poly) @ y  # Solve normal equation
    y_pred = X_poly @ coeffs
    return y_pred, coeffs

# Mean Squared Error Calculation
def mean_squared_error(y_true, y_pred):
    """Calculate mean squared error."""
    return np.mean((y_true - y_pred)**2)

def standardize_data(X):
    """
    Standardize data to have mean=0 and standard deviation=1.
    """
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

def hyperplane(X, y, learning_rate=0.01, n_iterations=1000):
    """
    Train a linear classifier using gradient descent.
    """
    np.random.seed(42)
    n_samples, n_features = X.shape 
    
    # Initialize weights and bias
    W = np.random.randn(n_features) # Random initialization of weights
    b = 0
    
    # Gradient Descent Optimization
    for _ in range(n_iterations): # Epochs
        # Compute predictions 
        linear_output = np.dot(X, W) + b # y_hat = W*X + b
        predictions = np.sign(linear_output) # Predicted class labels
        
        # Identify misclassified points
        errors = (y * linear_output) < 1  # Points violating margin
        
        # Update weights and bias for misclassified points
        W -= learning_rate * (-np.sum((y[errors][:, None] * X[errors]), axis=0) + W) # Derivative of hinge loss w.r.t weights
        b -= learning_rate * (-np.sum(y[errors])) # Derivative of hinge loss w.r.t bias
    
    return W, b

def plot_hyperplane(X, y, W, b):
    """
    Plot the data points, decision boundary, and margins.
    """
    plt.figure(figsize=(12, 8))
    
    # Plot Class 0 and Class 1
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')
    
    # Plot the decision boundary (hyperplane)
    x_values = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
    y_values = -(W[0] * x_values + b) / W[1]
    plt.plot(x_values, y_values, color='black', label='Hyperplane')
    
    # Plot margins (parallel to the hyperplane)
    margin = 1 / np.sqrt(np.sum(W**2))
    y_values_margin_pos = y_values + margin
    y_values_margin_neg = y_values - margin
    plt.plot(x_values, y_values_margin_pos, 'g--', label='Margin')
    plt.plot(x_values, y_values_margin_neg, 'g--')
    
    # Highlight support vectors (points near the margin)
    distances = np.abs(np.dot(X, W) + b) / np.sqrt(np.sum(W**2))
    support_vectors = distances <= (1 + 1e-3)
    plt.scatter(X[support_vectors][:, 0], X[support_vectors][:, 1], 
                s=100, facecolors='none', edgecolors='k', label='Support Vectors')
    
    # Plot formatting
    plt.title('Optimized Hyperplane with Standardized Data and Proper Margins')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

# Just to know, we are gonna implement those directly in the training loop

# Compute Likelihood
def likelihood(probs, y_onehot):
    """
    Compute likelihood for each sample.
    """
    return np.prod(probs ** y_onehot, axis=1)

# Compute Log-Likelihood
def log_likelihood(probs, y_onehot):
    """
    Compute log-likelihood for each sample.
    """
    return np.sum(y_onehot * np.log(probs + 1e-15), axis=1)

# Training Loop for Likelihood Maximization
def train_likelihood(X, y_onehot, W, b, learning_rate=0.01, epochs=100):
    """
    Train a softmax classifier by maximizing likelihood.
    """
    likelihood_history = []
    
    for epoch in range(epochs):
        # Forward Pass
        logits = np.dot(X, W) + b
        probs = softmax(logits)
        
        # Compute Likelihood (Numerically Unstable)
        likelihood = np.prod(np.sum(probs * y_onehot, axis=1))
        likelihood_history.append(likelihood)
        
        # Backpropagation
        grad_logits = (probs - y_onehot) / len(X)
        grad_W = np.dot(X.T, grad_logits)
        grad_b = np.sum(grad_logits, axis=0, keepdims=True)
        
        # Update Parameters
        W += learning_rate * grad_W  # Add gradient to maximize likelihood
        b += learning_rate * grad_b
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Likelihood: {likelihood:.10f}")
    
    return W, b, likelihood_history

# Training Loop for Log-Likelihood Maximization
def train_log_likelihood(X, y_onehot, W, b, learning_rate=0.01, epochs=1000):
    """
    Train a softmax classifier by maximizing log-likelihood.
    """
    log_likelihood_history = []
    
    for epoch in range(epochs):
        # Forward Pass
        logits = np.dot(X, W) + b
        probs = softmax(logits)
        
        # Compute Log-Likelihood
        log_likelihood = np.sum(y_onehot * np.log(probs + 1e-15)) / len(X)
        log_likelihood_history.append(log_likelihood)
        
        # Backpropagation
        grad_logits = (probs - y_onehot) / len(X)
        grad_W = np.dot(X.T, grad_logits)
        grad_b = np.sum(grad_logits, axis=0, keepdims=True)
        
        # Update Parameters
        W += learning_rate * grad_W  # Add gradient to maximize LL
        b += learning_rate * grad_b
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Log-Likelihood: {log_likelihood:.4f}")
    
    return W, b, log_likelihood_history

#  Training Loop for Optimization
def train_negative_log_likelihood_classifier(X, y_onehot, W, b, learning_rate=0.01, epochs=1000):
    """
    Train a softmax classifier using gradient descent.
    """
    loss_history = []
    for epoch in range(epochs):
        # Forward pass
        logits = np.dot(X, W) + b
        probs = softmax(logits)
        
        # Compute Loss
        loss = negative_log_likelihood(probs, y_onehot)
        loss_history.append(loss)
        
        # Backpropagation
        grad_logits = (probs - y_onehot) / len(X)
        grad_W = np.dot(X.T, grad_logits)
        grad_b = np.sum(grad_logits, axis=0, keepdims=True)
        
        # Update Parameters
        W -= learning_rate * grad_W
        b -= learning_rate * grad_b
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    return W, b, loss_history

# Ridge Regression
def ridge_regression(X, y, alpha=1.0):
    # Closed-form solution for Ridge Regression
    I = np.eye(X.shape[1])
    theta = np.linalg.inv(X.T @ X + alpha * I) @ X.T @ y
    return theta

# Classical Ridge Regression
def ridge_regression_intercept(X, y, lambda_reg):
    """Perform Ridge Regression."""
    X = np.c_[np.ones(len(X)), X]  # Add intercept term
    n_features = X.shape[1]
    theta = np.linalg.inv(X.T @ X + lambda_reg * np.eye(n_features)) @ X.T @ y
    return theta

# Bayesian perspective of Ridge Regression
def bayesian_ridge_posterior(X, y, prior_var, noise_var):
    """
    Compute the posterior distribution over parameters.
    """
    X = np.c_[np.ones(len(X)), X] # Add intercept term
    n_features = X.shape[1] 
    
    # Posterior precision matrix (inverse covariance)
    posterior_precision = (1/noise_var) * X.T @ X + (1/prior_var) * np.eye(n_features) 
    posterior_covariance = np.linalg.inv(posterior_precision) 
    
    # Posterior mean
    posterior_mean = (1/noise_var) * posterior_covariance @ X.T @ y
    
    return posterior_mean, posterior_covariance

def bayesian_lasso_posterior(X, y, prior_scale, noise_var):
    """
    Compute Bayesian Lasso Regression posterior.
    """
    n_features = X.shape[1]
    I = np.eye(n_features)
    precision_matrix = (1 / noise_var) * X.T @ X + (1 / prior_scale) * I
    posterior_cov = np.linalg.inv(precision_matrix)
    posterior_mean = (1 / noise_var) * posterior_cov @ X.T @ y
    return posterior_mean, posterior_cov

# Elastic Net Regression
def elastic_net(X, y, alpha1=1.0, alpha2=1.0, n_iter=1000, tol=1e-4):
    theta = np.zeros(X.shape[1])
    for _ in range(n_iter): # Convergence loop
        for j in range(len(theta)): # Update each coefficient
            residual = y - (X @ theta) + X[:, j] * theta[j] # Update residual
            rho = X[:, j].T @ residual # Compute rho 
            theta[j] = (rho - alpha1 * np.sign(rho)) / (X[:, j].T @ X[:, j] + alpha2) # Update theta
        if np.linalg.norm(X @ theta - y) < tol: # Check convergence 
            break
    return theta