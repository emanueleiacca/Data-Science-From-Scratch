import numpy as np
import matplotlib.pyplot as plt

# KMeans and Kmeans++ for Section 2.3
def initialize_centroids(data, k, method="random",seed=42):
    """
    Initialize centroids using the chosen method.
    Parameters:
        - data: NumPy array of data points.
        - k: Number of clusters.
        - method: "random" for basic initialization or "kmeans++" for K-me ()ans++ initialization.
    """
    if method == "random":
        np.random.seed(seed)  # Set the random seed for reproducibility
        # Randomly select k unique indices
        indices = np.random.choice(data.shape[0], k, replace=False) # Choose k random points as initial centroids
        return data[indices] # Return the initial centroids

    elif method == "kmeans++":
        np.random.seed(seed)
        # K-means++ initialization
        centroids = [data[np.random.choice(data.shape[0])]]  # First centroid randomly chosen
        for _ in range(1, k):
            # Compute distances from nearest centroid for all points
            distances = np.min([np.linalg.norm(data - centroid, axis=1) for centroid in centroids], axis=0) 
            # Compute probabilities proportional to squared distances
            probabilities = distances ** 2 / np.sum(distances ** 2)
            # Choose next centroid based on probabilities
            next_centroid_index = np.random.choice(data.shape[0], p=probabilities)
            centroids.append(data[next_centroid_index])
        return np.array(centroids)

    else:
        raise ValueError("Invalid method. Choose 'random' or 'kmeans++'.")

def compute_distance(point, centroids):
    """Compute the distance of a point to all centroids and return the nearest one."""
    distances = np.linalg.norm(centroids - point, axis=1)
    return np.argmin(distances)  # Return the index of the closest centroid

def assign_clusters(data, centroids):
    """Assign each point to the nearest centroid."""
    clusters = []
    for point in data: # Iterate over each point
        cluster_id = compute_distance(point, centroids) # Find the nearest centroid
        clusters.append(cluster_id)
    return np.array(clusters)

def update_centroids(data, clusters, k):
    """Update centroids as the mean of points in each cluster."""
    new_centroids = []
    for cluster_id in range(k): # Iterate over each cluster
        cluster_points = data[clusters == cluster_id] # Get points in the cluster
        if len(cluster_points) > 0: # Check if the cluster is non-empty
            new_centroids.append(cluster_points.mean(axis=0)) # Compute the mean of cluster points
        else:  # Handle empty cluster
            new_centroids.append(np.zeros(data.shape[1])) # Set the centroid to zero vector
    return np.array(new_centroids)

def kmeans(data, k, method="random", max_iterations=100, tolerance=1e-4, seed = 42):
    """
    K-means clustering algorithm with option for basic or K-means++ initialization.
    Parameters:
        - data: NumPy array of data points.
        - k: Number of clusters.
        - method: "random" for basic K-means or "kmeans++" for K-means++.
        - max_iterations: Maximum number of iterations.
        - tolerance: Convergence tolerance.
    """
    # Initialize centroids
    centroids = initialize_centroids(data, k, method=method)

    for iteration in range(max_iterations):
        # Assign clusters
        clusters = assign_clusters(data, centroids)

        # Update centroids
        new_centroids = update_centroids(data, clusters, k)

        # Check for convergence
        if np.all(np.abs(new_centroids - centroids) < tolerance):
            print(f"Converged at iteration {iteration}")
            break

        centroids = new_centroids

    return centroids, clusters

# KMeans Tracking Section 3

def kmeans_iterations(data, k, method="random", max_iterations=100, tolerance=1e-4):
    """
    Perform K-means clustering and track iterations.
    
    Parameters:
        data: numpy.ndarray
            The dataset to cluster.
        k: int
            Number of clusters.
        method: str
            Initialization method ("random" or "kmeans++").
        max_iterations: int
            Maximum number of iterations.
        tolerance: float
            Convergence tolerance.
    
    Returns:
        centroids_history: list of numpy.ndarray
            History of centroid positions at each iteration.
        cluster_history: list of numpy.ndarray
            History of cluster assignments at each iteration.
    """
    """
    It's the same function we already did before in our KMeans clustering Function, 
    this time btw we need to store each iteration to retrieve it later, 
    using the already implemented function was saving only the initial and final iterations
    """
    centroids = initialize_centroids(data, k, method)  # Initialize centroids
    centroids_history = [centroids]  # Store initial centroids
    cluster_history = []  # Store cluster assignments for each iteration

    for iteration in range(max_iterations):
        # Assign clusters based on current centroids
        clusters = assign_clusters(data, centroids)
        cluster_history.append(clusters)

        # Update centroids
        new_centroids = update_centroids(data, clusters, k)
        centroids_history.append(new_centroids)

        # Check for convergence
        if np.all(np.abs(new_centroids - centroids) < tolerance):
            print(f"Converged at iteration {iteration}")
            break

        centroids = new_centroids

    return centroids_history, cluster_history

def visualize_kmeans_iterations(data, k, selected_features, selected_features_labels, method="random", max_iterations=100):
    """
    Visualize K-means clustering progress over iterations using the predefined kmeans_iterations function.

    Parameters:
    - data: The dataset (NumPy array or DataFrame) for clustering.
    - k: Number of clusters.
    - selected_features: Indices of features to use for visualization.
    - selected_features_labels: Names of the selected features for plotting.
    - method: Initialization method for centroids ("random" or "kmeans++").
    - max_iterations: Maximum number of iterations to run K-means.
    """
    selected_data = data[:, selected_features]

    centroids_history, cluster_history = kmeans_iterations(data, k=k, method=method, max_iterations=max_iterations)

    # Visualize 
    for iteration, (centroids, clusters) in enumerate(zip(centroids_history, cluster_history)):
        plt.figure(figsize=(8, 6))
        for cluster_id in np.unique(clusters):
            cluster_points = selected_data[clusters == cluster_id]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster_id}")
        plt.scatter(centroids[:, selected_features[0]], centroids[:, selected_features[1]],
                    color='red', marker='x', s=100, label='Centroids')
        plt.title(f"Iteration {iteration + 1}")
        plt.xlabel(selected_features_labels[0])
        plt.ylabel(selected_features_labels[1])
        plt.legend()
        plt.show()

    print("Visualization complete.")
