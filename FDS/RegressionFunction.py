import numpy as np

# Generate a simple linear dataset
def generate_simple_data(samples=50):
    """
    Generate a clean synthetic dataset for linear regression.
    """
    x = np.linspace(0, 10, samples)  # Uniformly spaced data points
    y = 2 * x + 5 + np.random.normal(0, 1, samples)  # Linear relationship with minimal noise
    return x, y

def hypothesis(theta_0, theta_1, x):
    """
    Hypothesis function for linear regression.
    """
    return theta_0 + theta_1 * x

def cost_function(x, y, theta_0, theta_1):
    """
    Calculate the cost function for linear regression.
    """
    m = len(y)  # Number of samples
    total_error = 0

    for i in range(m):
        total_error += (hypothesis(theta_0, theta_1, x[i]) - y[i]) ** 2 

    return total_error / (2 * m)

def gradient_descent(x, y, theta_0, theta_1, learning_rate, iterations):
    """
    Perform gradient descent to optimize theta_0 and theta_1.
    """
    m = len(y)  # Number of samples
    cost_history = []  # Store cost at each iteration

    for iteration in range(iterations): # Loop through iterations
        sum_error_0 = 0 # Initialize sum of errors for theta_0
        sum_error_1 = 0 # Initialize sum of errors for theta_1

        # Calculate gradients
        for i in range(m): # Loop through samples
            error = hypothesis(theta_0, theta_1, x[i]) - y[i] # Calculate error
            sum_error_0 += error # Update sum of errors for theta_0
            sum_error_1 += error * x[i] # Update sum of errors for theta_1

        # Update parameters
        theta_0 -= (learning_rate / m) * sum_error_0 # Update theta_0
        theta_1 -= (learning_rate / m) * sum_error_1 # Update theta_1

        # Track cost
        cost = cost_function(x, y, theta_0, theta_1) # Calculate cost
        cost_history.append(cost) # Record cost

        # Debugging: Print every 100 iterations
        #if iteration % 100 == 0:
            #print(f"Iteration {iteration}: Cost={cost:.4f}, Theta_0={theta_0:.4f}, Theta_1={theta_1:.4f}")

    return theta_0, theta_1, cost_history # Return optimized parameters and cost history

# R-Squared Calculation
def r_squared(y, y_pred): 
    """
    Calculate the R-Squared metric for linear regression.
    """
    ss_total = np.sum((y - np.mean(y))**2) # Total sum of squares
    ss_residual = np.sum((y - y_pred)**2) # Residual sum of squares
    return 1 - (ss_residual / ss_total) # Calculate R-Squared

def predict(x_new, theta_0, theta_1):
    """
    Predict the target value for a new input x_new using the regression coefficients.
    """
    return theta_0 + theta_1 * x_new # Calculate prediction using linear model


# Batch Gradient Descent Implementation
def batch_gradient_descent(x, y, theta_0, theta_1, learning_rate, iterations):
    """
    Perform Batch Gradient Descent to minimize cost function.
    Updates parameters using all samples per iteration.
    """
    m = len(y)  # Number of samples
    cost_history = []  # To store cost values for each iteration

    print("\n--- Batch Gradient Descent ---")
    for iteration in range(iterations): 
        # Calculate error for all samples
        error = hypothesis(theta_0, theta_1, x) - y 
        
        # Update parameters
        theta_0 -= (learning_rate / m) * np.sum(error)
        theta_1 -= (learning_rate / m) * np.sum(error * x)
        
        # Calculate and record cost
        cost = cost_function(x, y, theta_0, theta_1)
        cost_history.append(cost)
        
        # Debug: Print progress every 100 iterations
        #if iteration % 100 == 0:
            #print(f"Iteration {iteration}: Theta_0={theta_0:.4f}, Theta_1={theta_1:.4f}, Cost={cost:.4f}")
    
    return theta_0, theta_1, cost_history

# Stochastic Gradient Descent Implementation
def stochastic_gradient_descent(x, y, theta_0, theta_1, learning_rate, iterations):
    """
    Perform Stochastic Gradient Descent.
    Updates parameters using one sample at a time.
    """
    m = len(y)  # Number of samples
    cost_history = []  # To store cost values per epoch

    print("\n--- Stochastic Gradient Descent ---")
    for iteration in range(iterations):
        for i in range(m):
            # Calculate error for one sample
            error = hypothesis(theta_0, theta_1, x[i]) - y[i]
            
            # Update parameters using one sample
            theta_0 -= learning_rate * error
            theta_1 -= learning_rate * error * x[i]
        
        # Calculate and record cost after each epoch
        cost = cost_function(x, y, theta_0, theta_1)
        cost_history.append(cost)
        
        # Debug: Print progress every epoch
        #if iteration % 1 == 0:
            #print(f"Epoch {iteration}: Theta_0={theta_0:.4f}, Theta_1={theta_1:.4f}, Cost={cost:.4f}")
    
    return theta_0, theta_1, cost_history

# Mini-Batch Gradient Descent Implementation
def mini_batch_gradient_descent(x, y, theta_0, theta_1, learning_rate, iterations, batch_size):
    """
    Perform Mini-Batch Gradient Descent.
    Updates parameters using a batch of samples.
    """
    m = len(y)  # Number of samples
    cost_history = []  # To store cost values per epoch

    print("\n--- Mini-Batch Gradient Descent ---")
    for iteration in range(iterations):
        indices = np.random.permutation(m)  # Shuffle dataset
        x_shuffled = x[indices] 
        y_shuffled = y[indices] 

        for i in range(0, m, batch_size): # Loop through mini-batches
            x_batch = x_shuffled[i:i + batch_size] # Get mini-batch features
            y_batch = y_shuffled[i:i + batch_size] # Get mini-batch targets
            
            # Calculate error for mini-batch
            error = hypothesis(theta_0, theta_1, x_batch) - y_batch
            
            # Update parameters using mini-batch
            theta_0 -= (learning_rate / batch_size) * np.sum(error)
            theta_1 -= (learning_rate / batch_size) * np.sum(error * x_batch)
        
        # Calculate and record cost after each epoch
        cost = cost_function(x, y, theta_0, theta_1)
        cost_history.append(cost)
        
        # Debug: Print progress every epoch
        #if iteration % 10 == 0:
            #print(f"Epoch {iteration}: Theta_0={theta_0:.4f}, Theta_1={theta_1:.4f}, Cost={cost:.4f}")
    
    return theta_0, theta_1, cost_history

# Normal Equation Function
def normal_equation(X, y):
    """
    Solve the Normal Equation to find optimal theta.
    θ = (X^T X)^-1 X^T y
    """
    # Compute X^T X
    XT_X = np.dot(X.T, X) # np.dot = row per column
    print("\nStep 1: X^T X Computed")
    print(XT_X)
    
    # Compute X^T y
    XT_y = np.dot(X.T, y)
    print("\nStep 2: X^T y Computed")
    print(XT_y)
    
    # Compute (X^T X)^-1
    XT_X_inv = np.linalg.inv(XT_X)
    print("\nStep 3: (X^T X)^-1 Computed")
    print(XT_X_inv)
    
    # Compute Theta
    theta = np.dot(XT_X_inv, XT_y)
    print("\nStep 4: Optimal Theta Computed")
    print(theta)
    
    return theta

def normal_equation_regularized(X, y, lambda_param):
    """
    Solve the Regularized Normal Equation.
    θ = (X^T X + λI)^-1 X^T y
    """
    I = np.eye(X.shape[1]) # np.eye = identity matrix
    I[0, 0] = 0  # Don't regularize the bias term
    
    theta = np.dot(np.linalg.inv(np.dot(X.T, X) + lambda_param * I), np.dot(X.T, y)) # formula
    return theta

# Calculate Conditional Probability for Each Observation
def conditional_probability(y_i, y_pred_i, sigma_squared):
    """
    Calculate the conditional probability of observed y given predicted y.
    """
    # Ensure y_i and y_pred_i are scalars, not arrays
    prob = (1 / np.sqrt(2 * np.pi * sigma_squared)) * np.exp(-(y_i - y_pred_i)**2 / (2 * sigma_squared)) # formula
    return prob # Return conditional probability

# Calculate Overall Likelihood
def likelihood(y, y_pred, sigma_squared):
    """
    Calculate the likelihood of the model given the observations.
    """
    return np.prod((1 / np.sqrt(2 * np.pi * sigma_squared)) * np.exp(-(y - y_pred)**2 / (2 * sigma_squared))) # formula

# Calculate Log-Likelihood
def log_likelihood(y, y_pred, sigma_squared):
    """
    Calculate the log-likelihood of the model given the observations.
    """
    m = len(y)
    return - (m / 2) * np.log(2 * np.pi * sigma_squared) - (1 / (2 * sigma_squared)) * np.sum((y - y_pred)**2) # formula

# Sum of Squared Errors (SSE)
def calculate_sse(y, y_pred):
    """
    Calculate the Sum of Squared Errors (SSE).
    """
    sse = np.sum((y - y_pred)**2) # formula
    return sse

# Pearson's Product-Moment Correlation (PPMC)
def calculate_ppmc(x, y):
    """
    Calculate the Pearson Correlation Coefficient (r).
    """
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
    
    r = numerator / denominator
    return r

def polynomial_features(x, degree):
    """
    Generate polynomial features up to a given degree.
    """
    X_poly = np.vander(x, degree + 1, increasing=True) # It generates a matrix where each column is a power of the input array x
    """
    example: x = np.array([1, 2, 3])
    degree = 3
    output = [[  1   1   1   1]
              [  1   2   4   8]
              [  1   3   9  27]
              [  1   4  16  64]]
              Row 1: [1,1^1,1^2,1^3]
              Row 2: [1,2^1,2^2,2^3]
              Row 3: [1,3^1,3^2,3^3]
              Row 4: [1,4^1,4^2,4^3]
    """
    return X_poly

def fit_polynomial_regression(X, y):
    """
    Fit Polynomial Regression using the Normal Equation.
    """
    # θ = (XᵀX)⁻¹ Xᵀy, where:
    # - `@` is the matrix multiplication operator in Python (equivalent to np.dot for matrices).
    # - XᵀX captures relationships between features through matrix multiplication.
    # - (XᵀX)⁻¹ calculates the inverse of XᵀX, ensuring solvability.
    # - Xᵀy maps the target values (y) onto the feature space of X using matrix multiplication.
    # The result is a vector of coefficients (theta) that minimizes prediction error. 
    theta = np.linalg.inv(X.T @ X) @ X.T @ y 
    return theta

def predict_polynomial(X, theta):
    """
    Predict using Polynomial Regression coefficients.
    """
    return X @ theta 

def ridge_regression(X, y, lambd):
    """
    Perform Ridge Regression with L2 Regularization.
    """
    #   - λ (lambda) controls the strength of the regularization (higher λ shrinks coefficients more).
    # - `X.T @ X`: Computes the correlation between features.
    # - `X.T @ y`: Projects target values (y) onto the feature space defined by X.
    # - `(X.T @ X + λI)` ensures that the matrix is invertible even if features are collinear (avoiding singular matrix issues).
    # - `np.linalg.inv(...)`: Calculates the inverse of the modified matrix.
    I = np.eye(X.shape[1]) # Identity matrix
    theta = np.linalg.inv(X.T @ X + lambd * I) @ X.T @ y 
    return theta

def lasso_regression(X, y, lambd):
    """
    Perform Lasso Regression with L1 Regularization using Normal Equation + Soft Thresholding.
    """

    # - np.ones((X.shape[0], 1)) creates a column of ones (for the intercept term).
    # - np.c_ concatenates it with X as the first column.
    X_bias = np.c_[np.ones((X.shape[0], 1)), X] # Add a bias term to X
    
    # Compute initial coefficients using Ridge-like normal equation
    I = np.eye(X_bias.shape[1])
    I[0, 0] = 0  # Do not regularize the bias term
    
    # Calculate initial coefficients (theta) using a Ridge-like normal equation
    theta = np.linalg.inv(X_bias.T @ X_bias + lambd * I) @ X_bias.T @ y
    
    # Apply soft-thresholding for L1 regularization
    # - Soft Thresholding adjusts each coefficient to account for the L1 penalty (λ).

    # - np.sign(theta[i]) gets the sign of the coefficient (+1 or -1).
    # - max(abs(theta[i]) - lambd, 0) shrinks the coefficient towards zero.
    # - If the absolute value of a coefficient is smaller than λ, it becomes zero.
    for i in range(1, len(theta)):  # Skip bias term
        theta[i] = np.sign(theta[i]) * max(abs(theta[i]) - lambd, 0) 
    
    return theta

# Predictions only fro Lasso Regression
def predict_polynomial_lasso(X, theta):
    """
    Predict values using polynomial regression coefficients.
    """
    X_bias = np.c_[np.ones((X.shape[0], 1)), X] # Add bias (intercept) term (in normal version bias is already part of X. Here we need to add it)
    return X_bias @ theta

def gaussian_kernel(x, x_query, tau):
    """
    Calculate Gaussian weights for LWR.
    """
    distances = (x - x_query)**2  # Calculate squared distances between x and the query point
    weights = np.exp(-distances / (2 * tau**2))  # Apply the Gaussian kernel formula
    return np.diag(weights)  # Convert the weights into a diagonal matrix


def locally_weighted_regression(x, y, x_query, tau):
    """
    Perform Locally Weighted Regression for a single query point.
    """
    X = np.c_[np.ones_like(x), x]  # Add bias term
    X_query = np.array([1, x_query])  # Add bias term for query point

    W = gaussian_kernel(x, x_query, tau)  # Get weights

    # Solve for theta using weighted normal equation
    theta = np.linalg.pinv(X.T @ W @ X) @ X.T @ W @ y
    y_pred = X_query @ theta # # Prediction using the query point and calculated theta
    return y_pred