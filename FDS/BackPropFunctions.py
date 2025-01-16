import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # Initialize weights with better scaling
        self.W1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size) # Xavier initialization for ReLU
        self.W2 = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size) # more info here: https://paperswithcode.com/method/xavier-initialization
        self.lr = learning_rate
        self.memory = {} # Store intermediate values for backpropagation
        
    def relu(self, x):
        return np.maximum(0, x) # ReLU activation function
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0) # Derivative of ReLU
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True)) # Subtract max for numerical stability
        return exp_x / np.sum(exp_x, axis=1, keepdims=True) # Softmax activation function
    
    def forward(self, X):
        self.memory['X'] = X # Store input for backpropagation
        self.memory['Z1'] = np.dot(X, self.W1) # Linear transformation (dot product) 1
        self.memory['A1'] = self.relu(self.memory['Z1']) # Activation function 1
        self.memory['Z2'] = np.dot(self.memory['A1'], self.W2) # Linear transformation 2
        self.memory['A2'] = self.softmax(self.memory['Z2']) # Activation function 2
        return self.memory['A2'] 
    
    def backward(self, y_true):
        batch_size = y_true.shape[0] # Get number of samples
        
        dZ2 = self.memory['A2'] - y_true # Derivative of cross-entropy loss with softmax
        dW2 = np.dot(self.memory['A1'].T, dZ2) / batch_size # Derivative of weights 2
        
        dA1 = np.dot(dZ2, self.W2.T) # Derivative of activation 1
        dZ1 = dA1 * self.relu_derivative(self.memory['Z1']) # Derivative of ReLU
        dW1 = np.dot(self.memory['X'].T, dZ1) / batch_size # Derivative of weights 1
        
        # Smaller regularization and learning rate
        lambda_reg = 0.001 # Regularization strength
        self.W2 -= self.lr * (dW2 + lambda_reg * self.W2) # Update weights 2
        self.W1 -= self.lr * (dW1 + lambda_reg * self.W1) # Update weights 1

def generate_data():
    # Generate more balanced dataset
    np.random.seed(42)
    n_samples = 100
    
    # Generate two classes with clear separation
    X1 = np.random.randn(n_samples//2, 2) + np.array([1, 1]) 
    X2 = np.random.randn(n_samples//2, 2) - np.array([1, 1])
    X = np.vstack([X1, X2])
    
    # Create labels
    y = np.zeros((n_samples, 2))
    y[:n_samples//2, 0] = 1
    y[n_samples//2:, 1] = 1
    
    return X, y

def plot_decision_boundary(net, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.02  # Smaller step size for smoother boundary
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    Z = net.forward(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.argmax(axis=1).reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y.argmax(axis=1), 
               cmap=plt.cm.RdYlBu, alpha=0.8)
    plt.title('Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

def train_network():
    # Generate data
    X, y = generate_data()
    
    # Create network with larger hidden layer
    net = NeuralNetwork(2, 5, 2, learning_rate=0.01) 
    
    # Training loop
    n_epochs = 1000
    for epoch in range(n_epochs):
        y_pred = net.forward(X) # Forward pass
        net.backward(y) # Backward pass
        
        if epoch % 100 == 0:
            loss = -np.mean(y * np.log(y_pred + 1e-8)) # Cross-entropy loss
            print(f"Epoch {epoch}, Loss: {loss:.4f}") 
    
    plot_decision_boundary(net, X, y)

class SimpleComputationNode:
    def __init__(self):
        self.cache = {}
        
    def forward_step_by_step(self, x, y, z):
        """
        Detailed step-by-step forward pass of f(x,y,z) = (x + y)z
        """
        # Step 1: Addition (x + y)
        q = x + y
        self.cache['q'] = q
        print(f"Step 1 - Addition (x + y): {q}")
        
        # Step 2: Multiplication with z
        f = q * z
        self.cache['f'] = f
        print(f"Step 2 - Multiplication (q * z): {f}")
        
        return f
    
    def backward_step_by_step(self, dL_df=1.0):
        """
        Detailed step-by-step backward pass
        dL_df: gradient of loss with respect to final output (default=1.0)
        """
        # Get cached values
        q = self.cache['q']
        
        # Step 1: Derivative with respect to q and z
        df_dq = self.cache.get('z', 0)  # ∂f/∂q = z
        df_dz = q                        # ∂f/∂z = q
        print(f"Step 1a - Derivative wrt q (∂f/∂q = z): {df_dq}")
        print(f"Step 1b - Derivative wrt z (∂f/∂z = q): {df_dz}")
        
        # Step 2: Derivative with respect to x and y
        dq_dx = 1  # ∂q/∂x = 1 (from addition)
        dq_dy = 1  # ∂q/∂y = 1 (from addition)
        print(f"Step 2a - Derivative wrt x (∂q/∂x): {dq_dx}")
        print(f"Step 2b - Derivative wrt y (∂q/∂y): {dq_dy}")
        
        # Step 3: Apply chain rule
        dL_dx = dL_df * df_dq * dq_dx
        dL_dy = dL_df * df_dq * dq_dy
        dL_dz = dL_df * df_dz
        
        print("\nFinal gradients (after chain rule):")
        print(f"dL/dx: {dL_dx}")
        print(f"dL/dy: {dL_dy}")
        print(f"dL/dz: {dL_dz}")
        
        return dL_dx, dL_dy, dL_dz

def visualize_computation():
    """Visualize the computation steps"""
    # Create sample data
    x = np.linspace(-5, 5, 100)
    y = 2  # Fixed value for visualization
    z = 3  # Fixed value for visualization
    
    plt.figure(figsize=(15, 5))
    
    # Plot 1: x + y
    plt.subplot(1, 3, 1)
    q = x + y
    plt.plot(x, q, 'b-')
    plt.title('Step 1: x + y')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.grid(True)
    
    # Plot 2: Final result (x + y)z
    plt.subplot(1, 3, 2)
    f = q * z
    plt.plot(x, f, 'r-')
    plt.title('Step 2: (x + y)z')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.grid(True)
    
    # Plot 3: Gradients
    plt.subplot(1, 3, 3)
    df_dx = z * np.ones_like(x)  # Gradient with respect to x
    df_dz = q                    # Gradient with respect to z
    plt.plot(x, df_dx, 'g-', label='∂f/∂x')
    plt.plot(x, df_dz, 'b--', label='∂f/∂z')
    plt.title('Gradients')
    plt.legend()
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def run_detailed_example():
    # Initialize computation node
    node = SimpleComputationNode()
    
    # Example values
    x, y, z = 2.0, 3.0, 4.0
    
    print("=== Forward Pass ===")
    print(f"Input values: x={x}, y={y}, z={z}")
    f = node.forward_step_by_step(x, y, z)
    
    print("\n=== Backward Pass ===")
    dx, dy, dz = node.backward_step_by_step()
    
    print("\n=== Visualization ===")
    visualize_computation()
    
    # Verify results
    print("\n=== Verification ===")
    print(f"f(x,y,z) = (x + y)z = ({x} + {y}) * {z} = {f}")
    print(f"Expected gradients:")
    print(f"∂f/∂x = z = {z}")
    print(f"∂f/∂y = z = {z}")
    print(f"∂f/∂z = x + y = {x + y}")

class SigmoidNode:
    def __init__(self):
        self.memory = {}
        
    def sigmoid(self, x):
        """
        Compute sigmoid function
        f(x) = 1 / (1 + e^(-x))
        """
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """
        Compute sigmoid derivative
        f'(x) = f(x) * (1 - f(x))
        """
        s = self.sigmoid(x)
        return s * (1 - s)
    
    # Forward pass computes the sigmoid output
    def forward(self, w, x):
        """
        Forward pass through sigmoid
        Stores intermediate values for backward pass (for gradient computation
        """
        # Store inputs
        self.memory['w'] = w # Weights
        self.memory['x'] = x # Inputs
        
        # Linear combination
        z = np.dot(w[:-1], x) + w[-1]  # w0*x0 + w1*x1 + w2
        self.memory['z'] = z # Store intermediate value
        
        # Sigmoid activation
        output = self.sigmoid(z)
        self.memory['output'] = output
        
        return output
    
    # Backward pass computes gradients using the chain rule
    def backward(self, dL_df):
        """
        Backward pass computing gradients
        dL_df: gradient of loss with respect to final output
        """
        # Get stored values
        w = self.memory['w']
        x = self.memory['x']
        z = self.memory['z']
        
        # Compute sigmoid derivative
        dsigmoid = self.sigmoid_derivative(z)
        
        # Compute gradients
        dL_dz = dL_df * dsigmoid # Chain rule for gradient
        
        # Gradients with respect to weights and inputs
        dL_dw = np.zeros_like(w) # Initialize gradients
        dL_dw[:-1] = dL_dz * x  # For w0, w1 (inputs) 
        dL_dw[-1] = dL_dz      # For w2 (bias) 
        
        dL_dx = dL_dz * w[:-1]  # For x0, x1 
        
        return dL_dw, dL_dx

def visualize_sigmoid():
    """Visualize sigmoid function and its derivative"""
    x = np.linspace(-10, 10, 100)
    node = SigmoidNode()
    
    # Compute sigmoid and its derivative
    y_sigmoid = node.sigmoid(x)
    y_derivative = node.sigmoid_derivative(x)
    
    plt.figure(figsize=(12, 5))
    
    # Plot sigmoid
    plt.subplot(1, 2, 1)
    plt.plot(x, y_sigmoid, 'b-', label='Sigmoid')
    plt.title('Sigmoid Function')
    plt.grid(True)
    plt.legend()
    
    # Plot derivative
    plt.subplot(1, 2, 2)
    plt.plot(x, y_derivative, 'r-', label='Derivative')
    plt.title('Sigmoid Derivative')
    plt.grid(True)
    plt.legend()
    
    plt.show()

# Example usage
def run_example():
    # Initialize node
    node = SigmoidNode()
    
    # Example inputs
    w = np.array([0.5, -0.5, 0.1])  # w0, w1, w2
    x = np.array([1.0, 2.0])        # x0, x1
    
    # Forward pass
    output = node.forward(w, x)
    print(f"Forward pass output: {output:.4f}")
    
    # Backward pass (assume dL/df = 1 for simplicity)
    dL_dw, dL_dx = node.backward(1.0)
    print("\nGradients:")
    print(f"dL/dw: {dL_dw}")
    print(f"dL/dx: {dL_dx}")
    
    # Visualize sigmoid
    visualize_sigmoid()

class SigmoidNeuron:
    def __init__(self):
        # Store intermediate values for backpropagation
        self.cache = {}
        
    def forward_step_by_step(self, w, x):
        """
        Detailed step-by-step forward pass through sigmoid
        """
        # Step 1: Linear combination
        z = w[0]*x[0] + w[1]*x[1] + w[2]
        self.cache['z'] = z
        print(f"Step 1 - Linear combination (z): {z}")
        
        # Step 2: Negation
        neg_z = -z
        self.cache['neg_z'] = neg_z
        print(f"Step 2 - Negation (-z): {neg_z}")
        
        # Step 3: Exponential
        exp_neg_z = np.exp(neg_z)
        self.cache['exp_neg_z'] = exp_neg_z
        print(f"Step 3 - Exponential (e^-z): {exp_neg_z}")
        
        # Step 4: Addition
        one_plus_exp = 1 + exp_neg_z
        self.cache['one_plus_exp'] = one_plus_exp
        print(f"Step 4 - Addition (1 + e^-z): {one_plus_exp}")
        
        # Step 5: Final sigmoid output
        output = 1 / one_plus_exp
        self.cache['output'] = output
        print(f"Step 5 - Final sigmoid output: {output}")
        
        return output
    
    def backward_step_by_step(self, dL_df):
        """
        Detailed step-by-step backward pass through sigmoid
        dL_df: gradient of loss with respect to final output
        """
        # Get cached values
        z = self.cache['z']
        output = self.cache['output']
        
        # Step 1: Derivative of sigmoid
        dsigmoid = output * (1 - output)
        print(f"Step 1 - Sigmoid derivative: {dsigmoid}")
        
        # Step 2: Multiply by incoming gradient
        dL_dz = dL_df * dsigmoid
        print(f"Step 2 - Gradient w.r.t z: {dL_dz}")
        
        return dL_dz

def visualize_sigmoid_components():
    """Visualize all components of sigmoid computation"""
    x = np.linspace(-10, 10, 100)
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: -x
    plt.subplot(2, 2, 1)
    plt.plot(x, -x, 'r-')
    plt.title('Step 1: Negation (-x)')
    plt.grid(True)
    
    # Plot 2: e^(-x)
    plt.subplot(2, 2, 2)
    plt.plot(x, np.exp(-x), 'g-')
    plt.title('Step 2: Exponential (e^-x)')
    plt.grid(True)
    
    # Plot 3: 1 + e^(-x)
    plt.subplot(2, 2, 3)
    plt.plot(x, 1 + np.exp(-x), 'b-')
    plt.title('Step 3: Addition (1 + e^-x)')
    plt.grid(True)
    
    # Plot 4: Final sigmoid
    plt.subplot(2, 2, 4)
    plt.plot(x, 1/(1 + np.exp(-x)), 'purple')
    plt.title('Step 4: Final Sigmoid (1/(1 + e^-x))')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Example usage with step-by-step explanation
def run_detailed_example():
    # Initialize neuron
    neuron = SigmoidNeuron()
    
    # Example inputs
    w = np.array([0.5, -0.5, 0.1])  # weights and bias
    x = np.array([1.0, 2.0])        # inputs
    
    print("=== Forward Pass ===")
    output = neuron.forward_step_by_step(w, x)
    
    print("\n=== Backward Pass ===")
    dL_dz = neuron.backward_step_by_step(1.0)
    
    print("\n=== Visualization ===")
    visualize_sigmoid_components()