import numpy as np

def generate_complex_data(n_samples=300, noise=0.3):
    """Generate overlapping clusters with non-linear separation"""
    np.random.seed(42)
    
    # Class 0: Two Gaussian clusters
    n_class0 = n_samples // 2
    X0_1 = np.random.normal([-1, 0], noise, (n_class0 // 2, 2))
    X0_2 = np.random.normal([1, 0], noise, (n_class0 // 2, 2))
    X0 = np.vstack([X0_1, X0_2])
    
    # Class 1: Curved distribution
    n_class1 = n_samples - n_class0
    theta = np.linspace(-np.pi/2, np.pi/2, n_class1)
    radius = 2 + np.random.normal(0, 0.2, n_class1)
    X1 = np.vstack([
        radius * np.cos(theta) + np.random.normal(0, noise, n_class1),
        radius * np.sin(theta) + 1 + np.random.normal(0, noise, n_class1)
    ]).T
    
    # Combine data
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n_class0), np.ones(n_class1)])
    
    # Shuffle
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]

class ComplexNeuralNetwork:
    def __init__(self, layer_sizes=[2, 8, 4, 1]):
        """Initialize weights and biases"""
        self.layer_sizes = layer_sizes # store layer sizes
        self.weights = {} # store weights 
        self.biases = {} # store biases
        self.cache = {} # store intermediate values
        
        # Xavier/Glorot initialization
        for l in range(len(layer_sizes)-1): # for each layer
            limit = np.sqrt(6 / (layer_sizes[l] + layer_sizes[l+1])) # limit for uniform distribution [-limit, limit]
            self.weights[l] = np.random.uniform(-limit, limit, (layer_sizes[l], layer_sizes[l+1])) # weights calculated from uniform distribution by Xavier/Glorot initialization
            self.biases[l] = np.zeros((1, layer_sizes[l+1])) # biases calculated same way

    def relu(self, x):
        return np.maximum(0, x) # ReLU activation function
    
    def relu_derivative(self, x):
        return (x > 0).astype(float) # derivative of ReLU activation function
    
    def forward(self, X):
        self.cache['A0'] = X # input layer
        
        # First hidden layer
        Z1 = np.dot(X, self.weights[0]) + self.biases[0] # linear transformation (dot product of input and weights + biases)
        A1 = self.relu(Z1) # activation function (ReLU)
        self.cache['Z1'], self.cache['A1'] = Z1, A1 # store intermediate values
        
        # Second hidden layer
        # Same process but instead of having X as input we have A1
        Z2 = np.dot(A1, self.weights[1]) + self.biases[1]
        A2 = self.relu(Z2)
        self.cache['Z2'], self.cache['A2'] = Z2, A2
        
        # Output layer
        Z3 = np.dot(A2, self.weights[2]) + self.biases[2]
        A3 = 1/(1 + np.exp(-Z3))  # sigmoid because it's a binary classification
        self.cache['Z3'], self.cache['A3'] = Z3, A3
        
        return A3

    def backward(self, X, y, learning_rate=0.01):
        m = X.shape[0] # number of samples

        # Opposite order of forward pass: output -> hidden -> hidden -> input

        # Output layer
        dZ3 = self.cache['A3'] - y.reshape(-1, 1) # derivative of loss function with respect to Z3
        dW2 = np.dot(self.cache['A2'].T, dZ3) / m # derivative of loss function with respect to W2
        db2 = np.sum(dZ3, axis=0, keepdims=True) / m # derivative of loss function with respect to b2
        
        # Second hidden layer
        dA2 = np.dot(dZ3, self.weights[2].T) # derivative of loss function with respect to A2
        dZ2 = dA2 * self.relu_derivative(self.cache['Z2']) # derivative of loss function with respect to Z2
        dW1 = np.dot(self.cache['A1'].T, dZ2) / m # derivative of loss function with respect to W1
        db1 = np.sum(dZ2, axis=0, keepdims=True) / m # derivative of loss function with respect to b1
        
        # First hidden layer
        dA1 = np.dot(dZ2, self.weights[1].T) # derivative of loss function with respect to A1
        dZ1 = dA1 * self.relu_derivative(self.cache['Z1']) # derivative of loss function with respect to Z1
        dW0 = np.dot(self.cache['A0'].T, dZ1) / m # derivative of loss function with respect to W0
        db0 = np.sum(dZ1, axis=0, keepdims=True) / m # derivative of loss function with respect to b0
        
        # Update with momentum
        if not hasattr(self, 'momentum'): # if momentum is not calculated yet
            self.momentum = {f'W{i}': 0 for i in range(3)} # initialize momentum for weights
            self.momentum.update({f'b{i}': 0 for i in range(3)}) # initialize momentum for biases
        
        beta = 0.9  # momentum coefficient
        self.momentum['W2'] = beta * self.momentum['W2'] - learning_rate * dW2 # update momentum for W2
        self.momentum['W1'] = beta * self.momentum['W1'] - learning_rate * dW1 # update momentum for W1
        self.momentum['W0'] = beta * self.momentum['W0'] - learning_rate * dW0 # update momentum for W0
        
        self.weights[2] += self.momentum['W2'] # update weights for W2
        self.weights[1] += self.momentum['W1'] # update weights for W1
        self.weights[0] += self.momentum['W0'] # update weights for W0
        self.biases[2] -= learning_rate * db2 # update biases for b2
        self.biases[1] -= learning_rate * db1 # update biases for b1
        self.biases[0] -= learning_rate * db0 # update biases for b0

import matplotlib.pyplot as plt
def plot_decision_boundary(model, X, y, epoch, metrics):
    """Plot decision boundary with metrics"""
    plt.figure(figsize=(15, 6))
    
    # Decision boundary plot
    plt.subplot(1, 2, 1)
    
    # Create mesh grid
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    # Make predictions
    Z = model.forward(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    plt.contour(xx, yy, Z, colors='k', levels=[0.5], linestyles='--')
    
    # Plot data points
    plt.scatter(X[y==0][:, 0], X[y==0][:, 1], c='blue', label='Class 0', edgecolors='w')
    plt.scatter(X[y==1][:, 0], X[y==1][:, 1], c='red', label='Class 1', edgecolors='w')
    
    plt.title(f'Decision Boundary (Epoch {epoch})', fontsize=14)
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Metrics display
    plt.subplot(1, 2, 2)
    plt.axis('off')
    metrics_text = (
        f"Training Metrics at Epoch {epoch}:\n\n"
        f"Loss: {metrics['loss']:.4f}\n"
        f"Accuracy: {metrics['accuracy']:.4f}\n"
        f"Precision: {metrics['precision']:.4f}\n"
        f"Recall: {metrics['recall']:.4f}\n"
        f"F1 Score: {metrics['f1']:.4f}"
    )
    plt.text(0.1, 0.5, metrics_text, fontsize=14, 
            bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def train_and_visualize():
    X, y = generate_complex_data(n_samples=400, noise=0.3)
    model = ComplexNeuralNetwork([2, 8, 4, 1])
    # Use smaller learning rate
    learning_rate = 0.005
    epochs = 1000
    visualization_epochs = [0, 50, 200, 500, 999]
    
    # Training history
    losses = []
    accuracies = []
    
    
    for epoch in range(epochs):
        # Forward pass
        output = model.forward(X)
        
        # Compute metrics
        predictions = (output > 0.5).astype(int)
        accuracy = np.mean(predictions.flatten() == y)
        loss = -np.mean(y * np.log(output + 1e-8) + (1-y) * np.log(1-output + 1e-8))
        
        # Store metrics
        losses.append(loss)
        accuracies.append(accuracy)
        
        # Backward pass
        model.backward(X, y, learning_rate=learning_rate)
        
        # Visualize at specific epochs
        if epoch in visualization_epochs:
            tp = np.sum((predictions.flatten() == 1) & (y == 1))
            fp = np.sum((predictions.flatten() == 1) & (y == 0))
            fn = np.sum((predictions.flatten() == 0) & (y == 1))
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            
            metrics = {
                'loss': loss,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            plot_decision_boundary(model, X, y, epoch, metrics)
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class NetworkVisualizer:
    def __init__(self, network):
        self.network = network
        
    def visualize_network_state(self, epoch, X_sample, y_sample):
        """Create a comprehensive visualization of network state"""
        # Forward pass to get activations
        output = self.network.forward(X_sample)
        
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(3, 3, figure=fig)
        
        # 1. Network Architecture Overview
        ax_arch = fig.add_subplot(gs[0, :])
        self._plot_network_architecture(ax_arch)
        
        # 2. Layer Activations
        ax_input = fig.add_subplot(gs[1, 0])
        self._plot_layer_values(ax_input, X_sample, 'Input Features', 
                              show_values=True)
        
        ax_hidden1 = fig.add_subplot(gs[1, 1])
        self._plot_layer_values(ax_hidden1, self.network.cache['A1'], 
                              'Hidden Layer 1 (After ReLU)',
                              show_values=True)
        
        ax_output = fig.add_subplot(gs[1, 2])
        self._plot_layer_values(ax_output, output, 'Output Layer',
                              show_values=True)
        
        # 3. Weight Distributions
        ax_w1 = fig.add_subplot(gs[2, 0])
        self._plot_weight_distribution(ax_w1, self.network.weights[0], 
                                     'Weight Distribution - Layer 1')
        
        ax_w2 = fig.add_subplot(gs[2, 1])
        self._plot_weight_distribution(ax_w2, self.network.weights[1], 
                                     'Weight Distribution - Layer 2')
        
        plt.suptitle(f'Neural Network Internal State - Epoch {epoch}', 
                    fontsize=16, y=0.95)
        plt.tight_layout()
        plt.show()
        
        # Additional plot for decision boundary
        self.plot_decision_boundary(X_sample, y_sample, epoch)

    def _plot_network_architecture(self, ax):
        """Plot network architecture diagram"""
        layer_sizes = self.network.layer_sizes
        max_size = max(layer_sizes)
        
        # Plot neurons
        for i, size in enumerate(layer_sizes):
            x = np.ones(size) * i
            y = np.linspace(0, 1, size)
            ax.scatter(x, y, s=100, zorder=2)
            
            # Add layer labels
            if i == 0:
                label = 'Input\nLayer'
            elif i == len(layer_sizes)-1:
                label = 'Output\nLayer'
            else:
                label = f'Hidden\nLayer {i}'
            ax.text(i, -0.1, label, ha='center')
            
            # Draw connections to next layer
            if i < len(layer_sizes)-1:
                next_size = layer_sizes[i+1]
                next_y = np.linspace(0, 1, next_size)
                for y1 in y:
                    for y2 in next_y:
                        ax.plot([i, i+1], [y1, y2], 'gray', alpha=0.1)
        
        ax.set_title('Network Architecture')
        ax.axis('off')

    def _plot_layer_values(self, ax, values, title, show_values=False):
        """Plot layer activations with optional value display"""
        if len(values.shape) == 1:
            values = values.reshape(-1, 1)
        im = ax.imshow(values, cmap='viridis', aspect='auto')
        plt.colorbar(im, ax=ax)
        
        if show_values:
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    ax.text(j, i, f'{values[i,j]:.2f}', 
                           ha='center', va='center')
        
        ax.set_title(title)
        ax.set_xlabel('Neuron')
        ax.set_ylabel('Sample')

    def _plot_weight_distribution(self, ax, weights, title):
        """Plot histogram of weight values"""
        ax.hist(weights.flatten(), bins=30, color='blue', alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Count')

    def plot_decision_boundary(self, X, y, epoch):
        """Plot decision boundary and data points"""
        plt.figure(figsize=(10, 8))
        
        # Create mesh grid
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Get predictions
        Z = self.network.forward(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', 
                   edgecolors='black', s=100)  # Fixed parameter name
        
        plt.title(f'Decision Boundary - Epoch {epoch}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.colorbar(label='Prediction Probability')
        plt.show()
        
def train_with_visualization():
    # Generate data
    X, y = generate_complex_data(n_samples=50)  # Smaller sample for clarity
    
    # Create network and visualizer
    model = ComplexNeuralNetwork([2, 8, 4, 1])
    visualizer = NetworkVisualizer(model)
    
    # Training loop with visualization
    epochs = 1000
    visualization_epochs = [0, 100, 500, 999]
    
    for epoch in range(epochs):
        output = model.forward(X)
        model.backward(X, y, learning_rate=0.005)
        
        if epoch in visualization_epochs:
            visualizer.visualize_network_state(epoch, X[:5], y[:5])

import numpy as np
import matplotlib.pyplot as plt

class BatchNormalization:
    def __init__(self, num_features, epsilon=1e-5, momentum=0.99): 
        """Initialize parameters and buffers"""
        self.epsilon = epsilon # small value to prevent division by zero
        self.momentum = momentum  # smoothing parameter for running mean and variance
        self.gamma = np.ones((1, num_features)) # scale parameter
        self.beta = np.zeros((1, num_features)) # shift parameter
        self.running_mean = np.zeros((1, num_features)) # running mean of input data
        self.running_var = np.ones((1, num_features))  # running variance of input data
        self.training = True # flag to indicate training mode

    def forward(self, X):
        """Normalize input data and apply the scale and shift parameters"""
        if self.training: # during training
            mu = np.mean(X, axis=0, keepdims=True) # mean of input data
            var = np.var(X, axis=0, keepdims=True) # variance of input data
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu # update running mean by exponential moving average
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var # update running variance by exponential moving average 
        else:
            mu = self.running_mean # during inference, use running mean
            var = self.running_var # during inference, use running variance

        X_norm = (X - mu) / np.sqrt(var + self.epsilon) # normalize input data
        return self.gamma * X_norm + self.beta # scale and shift

    def backward(self, dY, X):
        """Backward pass to compute gradients"""
        m = X.shape[0] # number of samples
        # Same methods as forwared pass but in reverse order
        mu = np.mean(X, axis=0, keepdims=True) # mean of input data
        var = np.var(X, axis=0, keepdims=True) # variance of input data
        X_norm = (X - mu) / np.sqrt(var + self.epsilon) # normalize input data as before
        dGamma = np.sum(dY * X_norm, axis=0, keepdims=True) # gradient of gamma by chain rule
        dBeta = np.sum(dY, axis=0, keepdims=True) # gradient of beta by chain rule
        dX_norm = dY * self.gamma # gradient of normalized input data
        dVar = np.sum(dX_norm * (X - mu) * -0.5 * np.power(var + self.epsilon, -1.5), axis=0, keepdims=True) # gradient of variance by chain rule
        dMu = np.sum(dX_norm * -1 / np.sqrt(var + self.epsilon), axis=0, keepdims=True) + dVar * np.mean(-2 * (X - mu), axis=0, keepdims=True) # gradient of mean by chain rule
        dX = dX_norm / np.sqrt(var + self.epsilon) + dVar * 2 * (X - mu) / m + dMu / m # gradient of input data by chain rule
        return dX, dGamma, dBeta

class AdvancedNeuralNetwork:
    """Neural network with dropout and batch normalization"""
    def __init__(self, layer_sizes, dropout_rates):
        """Initialize weights, biases, and batch normalization layers"""
        self.layer_sizes = layer_sizes
        self.dropout_rates = dropout_rates
        self.weights = {}
        self.biases = {}
        self.batch_norms = {}
        for l in range(len(layer_sizes) - 1): # for each layer
            limit = np.sqrt(6 / (layer_sizes[l] + layer_sizes[l+1])) # limit for uniform distribution [-limit, limit]
            self.weights[l] = np.random.uniform(-limit, limit, (layer_sizes[l], layer_sizes[l+1])) # weights calculated from uniform distribution by Xavier/Glorot initialization
            self.biases[l] = np.zeros((1, layer_sizes[l+1])) # biases calculated same way
            if l < len(layer_sizes) - 2: # for hidden layers
                self.batch_norms[l] = BatchNormalization(layer_sizes[l+1]) # batch normalization layer

    def relu(self, X):
        return np.maximum(0, X)

    def relu_derivative(self, X):
        return (X > 0).astype(float)

    def apply_dropout(self, X, dropout_rate, training=True):
        """Apply dropout to the input data"""
        if not training or dropout_rate == 0: 
            return X # if not in training mode or dropout rate is 0, return input data
        mask = (np.random.rand(*X.shape) > dropout_rate).astype(float) # create a mask with 0s and 1s. 
        return X * mask / (1 - dropout_rate) # apply dropout and scale the output

    def forward(self, X, training=True):
        """Forward pass through the network"""
        self.cache = {'A0': X} # input layer
        for l in range(len(self.layer_sizes) - 1): # for each layer
            Z = np.dot(self.cache[f'A{l}'], self.weights[l]) + self.biases[l] # linear transformation (dot product of input and weights + biases)
            if l < len(self.layer_sizes) - 2: # for hidden layers
                Z = self.batch_norms[l].forward(Z) # apply batch normalization
            A = self.relu(Z) if l < len(self.layer_sizes) - 2 else 1 / (1 + np.exp(-Z)) # the conditional switch to ReLU activation for hidden layers and sigmoid for output layer
            if l < len(self.layer_sizes) - 2: # for hidden layers
                A = self.apply_dropout(A, self.dropout_rates[l], training) # apply dropout
            self.cache[f'Z{l+1}'] = Z # store intermediate values Z
            self.cache[f'A{l+1}'] = A # store intermediate values A
        return A

    def backward(self, X, y):
        """Backward pass to compute gradients"""
        m = X.shape[0] # Number of samples
        y = y.reshape(-1, 1) # Ensure y has shape (400, 1) in our case
        gradients = {} # Store gradients
        dA = self.cache[f'A{len(self.layer_sizes)-1}'] - y # Compute derivative of loss function with respect to A

        # Hidden and Final Layer are dealt at the same time
        for l in reversed(range(len(self.layer_sizes) - 1)): # Reverse order
            dZ = dA * (self.relu_derivative(self.cache[f'Z{l+1}']) if l < len(self.layer_sizes) - 2 else 1) # Batch norm only for hidden layers (the conditional tell it explicitely)
            gradients[f'W{l}'] = np.dot(self.cache[f'A{l}'].T, dZ) / m # Compute gradient of loss function with respect to W
            gradients[f'b{l}'] = np.sum(dZ, axis=0, keepdims=True) / m # Compute gradient of loss function with respect to b
            dA = np.dot(dZ, self.weights[l].T) # Compute derivative of loss function with respect to A_prev
        
        return gradients

    def train(self, X, y, epochs=1000, learning_rate=0.001):
        losses, accuracies = [], [] # Store metrics
        self.momentum = {f'W{l}': np.zeros_like(self.weights[l]) for l in range(len(self.layer_sizes) - 1)} # Initialize momentum 
        self.momentum.update({f'b{l}': np.zeros_like(self.biases[l]) for l in range(len(self.layer_sizes) - 1)}) # Initialize momentum updates 
        beta = 0.9
        for epoch in range(epochs): # For each epoch
            output = self.forward(X, training=True) # Forward pass
            loss = -np.mean(y * np.log(output + 1e-8) + (1 - y) * np.log(1 - output + 1e-8)) # Compute loss
            losses.append(loss) # Store loss
            accuracy = np.mean((output > 0.5).astype(int).flatten() == y.flatten()) # Compute accuracy
            accuracies.append(accuracy) # Store accuracy
            gradients = self.backward(X, y) # Backward pass
            for l in range(len(self.layer_sizes) - 1): # For each layer
                self.momentum[f'W{l}'] = beta * self.momentum[f'W{l}'] - learning_rate * (gradients[f'W{l}'] + 0.001 * self.weights[l])  # L2 regularization
                self.momentum[f'b{l}'] = beta * self.momentum[f'b{l}'] - learning_rate * gradients[f'b{l}'] # Update momentum
                self.weights[l] += self.momentum[f'W{l}'] # Update weights
                self.biases[l] += self.momentum[f'b{l}'] # Update biases
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
        return losses, accuracies


def visualize_training_results(model, X, y):
    losses, accuracies = model.train(X, y)

    # Plot Loss and Accuracy
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title('Training Accuracy')
    plt.grid(True)
    plt.show()

    # Decision Boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = model.forward(np.c_[xx.ravel(), yy.ravel()], training=False).reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='RdYlBu')

    # Predictions
    predictions = (model.forward(X, training=False) > 0.5).astype(int).flatten()

    # Correctly Classified
    correct_class0 = (predictions == y) & (y == 0)
    correct_class1 = (predictions == y) & (y == 1)

    # Misclassified
    misclassified_class0 = (predictions != y) & (y == 0)
    misclassified_class1 = (predictions != y) & (y == 1)

    # Plot points
    plt.scatter(X[correct_class0, 0], X[correct_class0, 1], c='green', label='Correct Class 0', edgecolors='w', alpha=0.8)
    plt.scatter(X[correct_class1, 0], X[correct_class1, 1], c='blue', label='Correct Class 1', edgecolors='w', alpha=0.8)
    plt.scatter(X[misclassified_class0, 0], X[misclassified_class0, 1], c='orange', marker='x', label='Misclassified Class 0', alpha=0.8)
    plt.scatter(X[misclassified_class1, 0], X[misclassified_class1, 1], c='red', marker='x', label='Misclassified Class 1', alpha=0.8)

    # Labels and Legend
    plt.legend()
    plt.title('Decision Boundary with Classification Details')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)
    plt.show()


def main():
    X, y = generate_complex_data(n_samples=400, noise=0.3)
    model = AdvancedNeuralNetwork(
        layer_sizes=[2, 16, 8, 1],  # Increased complexity
        dropout_rates=[0.3, 0.3]
    )
    visualize_training_results(model, X, y)