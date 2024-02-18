import numpy as np
from scipy.spatial.distance import cdist
from cluster import cluster

class My_KMeans(cluster):
    def __init__(self, k=5, max_iterations=100, balanced=False):
        self.k = k
        self.max_iterations = max_iterations
        self.balanced = balanced

    def fit(self, X):
        # Convert input to numpy array
        X = np.array(X)
        
        # Randomly initialize centroids
        centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]
        
        for _ in range(self.max_iterations):
            # Assign each instance to the nearest centroid
            distances = cdist(X, centroids)
            cluster_hypotheses = np.argmin(distances, axis=1)

            # Update centroids based on mean of instances in each cluster
            if self.balanced:
                new_centroids = np.zeros_like(centroids)
                for i in range(self.k):
                    cluster_instances = X[cluster_hypotheses == i]
                    if len(cluster_instances) > 0:
                        new_centroids[i] = cluster_instances.mean(axis=0)
                    else:
                        new_centroids[i] = centroids[i]
            else:
                new_centroids = np.array([X[cluster_hypotheses == i].mean(axis=0) for i in range(self.k)])
            
            # Check for convergence
            if np.all(centroids == new_centroids):
                break
                
            centroids = new_centroids
        
        return cluster_hypotheses.tolist(), centroids.tolist()
