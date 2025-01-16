import numpy as np

import numpy as np

def convolve(input_volume, filters, stride=1, padding=0):
    """
    Performs 2D convolution operation.
    
    Args:
        input_volume (np.ndarray): Input volume with shape (N, H, W, D).
        filters (np.ndarray): Filters with shape (F, F, D, K).
        stride (int): Stride value.
        padding (int): Padding value.
    
    Returns:
        np.ndarray: Output volume with shape (N, H_out, W_out, K).
    """
    N, H, W, D = input_volume.shape # N is the number of images, H is the height, W is the width, D is the depth
    F, _, _, K = filters.shape # F is the height and width of the filters, K is the number of filters
    
    print(f"Input volume shape: {input_volume.shape}")
    print(f"Filters shape: {filters.shape}")
    
    # Determine output size
    # Padding is the number of pixels to add to each side, stride is the number of pixels to move the filter each time
    H_out = (H - F + 2 * padding) // stride + 1  # H_out is the height of the output volume 
    W_out = (W - F + 2 * padding) // stride + 1 # W_out is the width of the output volume
    
    print(f"Output size: ({H_out}, {W_out}, {K})")
    
    # Apply zero padding to the input to handle the border pixels
    padded_input = np.pad(input_volume, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant')
    
    # Initialize the output volume
    output_volume = np.zeros((N, H_out, W_out, K))
    
    # Perform convolution
    for n in range(N): # For each image
        for i in range(H_out): # For each row
            for j in range(W_out): # For each column
                for k in range(K): # For each filter
                    x_start = i * stride # Starting x coordinate
                    x_end = x_start + F # Ending x coordinate
                    y_start = j * stride # Starting y coordinate
                    y_end = y_start + F # Ending y coordinate
                    
                    # Extract the current patch
                    patch = padded_input[n, x_start:x_end, y_start:y_end, :] # Extract the patch from the input volume by slicing
                    
                    # Compute the dot product between the patch and the filter
                    output_volume[n, i, j, k] = np.sum(patch * filters[:, :, :, k]) 
    
    return output_volume

def max_pool(input_volume, pool_size=2, stride=2):
    """
    Performs max pooling operation.
    
    Args:
        input_volume (np.ndarray): Input volume with shape (N, H, W, D).
        pool_size (int): Size of the pooling window.
        stride (int): Stride value.
    
    Returns:
        np.ndarray: Output volume with shape (N, H_out, W_out, D).
    """
    N, H, W, D = input_volume.shape
    
    # Determine output size
    H_out = (H - pool_size) // stride + 1 
    W_out = (W - pool_size) // stride + 1
    
    print(f"Max pooling input shape: {input_volume.shape}")
    print(f"Max pooling output size: ({N}, {H_out}, {W_out}, {D})")
    
    # Initialize the output volume
    output_volume = np.zeros((N, H_out, W_out, D))
    
    # Perform max pooling
    for n in range(N):
        for i in range(H_out):
            for j in range(W_out):
                for d in range(D):
                    x_start = i * stride
                    x_end = x_start + pool_size
                    y_start = j * stride
                    y_end = y_start + pool_size
                    
                    # Extract the current pooling window by slicing
                    window = input_volume[n, x_start:x_end, y_start:y_end, d] 
                    
                    # Compute the maximum value in the window
                    output_volume[n, i, j, d] = np.max(window)
    
    return output_volume

def fully_connected(input_volume, weights, bias):
    """
    Performs the fully connected layer operation.
    
    Args:
        input_volume (np.ndarray): Input volume with shape (N, D).
        weights (np.ndarray): Weights with shape (D, M).
        bias (np.ndarray): Bias with shape (1, M).
    
    Returns:
        np.ndarray: Output volume with shape (N, M).
    """
    N, D = input_volume.shape
    _, M = weights.shape
    
    print(f"Fully connected input shape: {input_volume.shape}")
    print(f"Fully connected weights shape: {weights.shape}")
    print(f"Fully connected bias shape: {bias.shape}")
    
    # Compute the dot product between the input and the weights
    output_volume = np.dot(input_volume, weights) + bias
    
    return output_volume

def relu(x):
    """
    Applies the ReLU activation function.
    """
    return np.maximum(0, x)

def softmax(x):
    """
    Applies the softmax activation function.
    """
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cnn_forward(input_volume, conv_filters, fc_weights, fc_bias):
    """
    Performs the forward pass through a CNN.
    
    Args:
        input_volume (np.ndarray): Input volume with shape (N, H, W, D).
        conv_filters (list of np.ndarray): List of convolutional filters.
        fc_weights (np.ndarray): Weights for the fully connected layer.
        fc_bias (np.ndarray): Bias for the fully connected layer.
    
    Returns:
        np.ndarray: Output volume with shape (N, M), where M is the number of classes.
    """
    # Convolutional layers
    conv_output = input_volume # Initialize the convolutional output with the input volume
    for filters in conv_filters: # For each set of filters
        conv_output = relu(convolve(conv_output, filters, stride=1, padding=1)) # Apply convolution operation with ReLU activation
        conv_output = max_pool(conv_output, pool_size=2, stride=2) # Apply max pooling
    
    # Flatten the convolutional output
    N, _, _, _ = conv_output.shape # N is the number of images 
    flat_conv_output = conv_output.reshape(N, -1) # Flatten the convolutional output to a 1D array
    
    # Fully connected layer
    output = fully_connected(flat_conv_output, fc_weights, fc_bias) # Apply the fully connected layer operation
    output = softmax(output) # Apply the softmax activation function to get the class probabilities
    
    return output