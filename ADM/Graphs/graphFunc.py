# Test the Girvan-Newman Algorithm

import time
from networkx.algorithms.community import girvan_newman as nx_girvan_newman
import numpy as np
# --- Custom Girvan-Newman ---
def test_girvan_newman(graph, custom_method, component_method):
    adj_matrix, nodes = graph.adjacency_matrix()  # Extract the adjacency matrix
    print(f"\n ⚠️--- Testing with Custom Method: {custom_method} | Component Method: {component_method} ---")
    
    # Measure time for Custom Girvan-Newman
    try:
        start_time = time.time()
        custom_communities, custom_removed_edges = custom_girvan_newman(graph, custom_method, component_method)
        end_time = time.time()
        custom_time = end_time - start_time
        print(f"⏱️ Custom Girvan-Newman Time: {custom_time:.4f} seconds")
        print("Custom Communities:", custom_communities)
        print("Modularity:", calculate_modularity(adj_matrix, custom_communities))
        print("Custom Removed Edges:", custom_removed_edges)
    except NotImplementedError as e:
        print(f"Custom Girvan-Newman: {e}")
        custom_time = None

    # Measure time for NetworkX Girvan-Newman
    print("\n ⚠️Running NetworkX Girvan-Newman Algorithm...")
    start_time = time.time()
    nx_gen = nx_girvan_newman(nx.Graph(graph.graph))  # Generate community splits
    nx_communities = next(iter(nx_gen))  # Extract first split
    end_time = time.time()
    nx_time = end_time - start_time
    nx_communities = [sorted(list(c)) for c in nx_communities]  # Sort nodes in each community
    print(f"⏱️ NetworkX Girvan-Newman Time: {nx_time:.4f} seconds")
    print("NetworkX Communities:", nx_communities)

    # Comparison
    if custom_time:
        print("\n--- Comparison Results ---")
        if sorted([sorted(c) for c in custom_communities]) == sorted(nx_communities):
            print("✅ Communities match!")
        else:
            print("❌ Communities do NOT match!")
        print(f"⏱️ Time Difference: {abs(custom_time - nx_time):.4f} seconds")

# Spectral Clustering

# KMeans and Kmeans++ for Section 2.3 for HW4
# Source: https://github.com/emanueleiacca/ADM-HW4/blob/main/functions/functions.py#L329
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
        indices = np.random.choice(data.shape[0], k, replace=False)
        return data[indices]

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
    for point in data:
        cluster_id = compute_distance(point, centroids)
        clusters.append(cluster_id)
    return np.array(clusters)

def update_centroids(data, clusters, k):
    """Update centroids as the mean of points in each cluster."""
    new_centroids = []
    for cluster_id in range(k):
        cluster_points = data[clusters == cluster_id]
        if len(cluster_points) > 0:
            new_centroids.append(cluster_points.mean(axis=0))
        else:  # Handle empty cluster
            new_centroids.append(np.zeros(data.shape[1]))
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

# Pre implemented functions for Louvain and Spectral Clustering

import numpy as np
from community import community_louvain  # python-louvain package
from sklearn.cluster import SpectralClustering
import networkx as nx
from collections import defaultdict
import time

# Pre-implemented Spectral Clustering
def pre_implemented_spectral(graph, k):
    adj_matrix = nx.to_numpy_array(graph)
    nodes = list(graph.nodes())

    sc = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=42)
    labels = sc.fit_predict(adj_matrix)
    
    communities = defaultdict(list)
    for i, label in enumerate(labels):
        communities[label].append(nodes[i])
    return list(communities.values())

# Pre-implemented Louvain Method
def pre_louvain_method(nx_graph):
    partition = community_louvain.best_partition(nx_graph)
    grouped = defaultdict(list)
    for node, comm in partition.items():
        grouped[comm].append(node)
    return list(grouped.values())

# Compare method and evaluate metrics
from tabulate import tabulate 

def compare_methods(custom_method, pre_method, name):
    print(f"\n--- Comparing {name} ---")
    print("Custom Communities:", custom_method)
    print("Pre-Implemented Communities:", pre_method)

    if sorted([sorted(c) for c in custom_method]) == sorted([sorted(c) for c in pre_method]):
        print(f"✅ {name} Communities Match!")
    else:
        print(f"❌ {name} Communities Do NOT Match!")

def compare_communities_overlap(custom, pre, name):

    print(f"\n--- {name} Overlap Comparison ---")

    overlap_data = []

    for i, custom_comm in enumerate(custom):
        overlap_scores = []
        for j, pre_comm in enumerate(pre):
            overlap = len(set(custom_comm) & set(pre_comm))
            overlap_scores.append((j, overlap))

        overlap_scores = sorted(overlap_scores, key=lambda x: -x[1])
        best_match = overlap_scores[0]
        
        overlap_data.append([
            f"Custom {i}", 
            f"Pre-Implemented {best_match[0]}",
            len(custom_comm),  # Custom community size
            len(pre[best_match[0]]),  # Best match size
            best_match[1]  # Overlap count
        ])

    headers = ["Custom Community", "Best Match", "Custom Size", "Best Match Size", "Overlap Nodes"]
    print(tabulate(overlap_data, headers=headers, tablefmt="grid"))

# --- Metrics Calculation Function ---
def evaluate_community_metrics(adj_matrix, communities, title):
    """
    Calculate and display Lambiotte Coefficient for nodes and Clauset's Parameter for communities.
    :param adj_matrix: NumPy adjacency matrix of the graph.
    :param communities: List of detected communities.
    :param title: Title for the display output.
    """
    n = adj_matrix.shape[0]
    node_degree = np.sum(adj_matrix, axis=1)

    # --- Calculate Lambiotte Coefficient ---
    lambiotte_coeff = {}
    for node in range(n):
        community = next(c for c in communities if node in c)
        internal_edges = np.sum(adj_matrix[node][community])
        lambiotte_coeff[node] = internal_edges / node_degree[node] if node_degree[node] > 0 else 0

    # --- Calculate Clauset's Parameter ---
    clauset_param = {}
    for i, community in enumerate(communities):
        internal_edges = sum(
            adj_matrix[node_i][node_j] for node_i in community for node_j in community if node_i != node_j
        )
        external_edges = sum(
            adj_matrix[node_i][node_j] for node_i in community for node_j in range(n) if node_j not in community
        )
        clauset_param[i] = internal_edges / (internal_edges + external_edges) if (internal_edges + external_edges) > 0 else 0

    # --- Display Results in Tabular Format ---
    print(f"\n--- {title} ---")
    print("\nLambiotte Coefficient (Node Importance):")
    lambiotte_table = [[node, f"{lambiotte_coeff[node]:.4f}"] for node in sorted(lambiotte_coeff.keys())]
    print(tabulate(lambiotte_table, headers=["Node", "Lambiotte Coefficient"], tablefmt="grid"))

    print("\nClauset's Parameter (Community Strength):")
    clauset_table = [[f"Community {i}", f"{clauset_param[i]:.4f}"] for i in clauset_param.keys()]
    print(tabulate(clauset_table, headers=["Community", "Clauset's Parameter"], tablefmt="grid"))
