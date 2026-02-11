from sklearn.metrics.pairwise import euclidean_distances
import numpy as np


class KMeans(object):
    def __init__(self, n_clusters=2, dist=euclidean_distances, random_state=42):
        self.n_clusters = n_clusters
        self.dist = dist
        self.rstate = np.random.RandomState(random_state)
        self.cluster_centers_ = []

    def fit(self, X):
        rint = self.rstate.randint

        initial_indices = [rint(X.shape[0])]
        for _ in range(self.n_clusters - 1):
            i = rint(X.shape[0])
            while i in initial_indices:
                i = rint(X.shape[0])
            initial_indices.append(i)

        self.cluster_centers_ = X[initial_indices, :]

        continue_condition = True
        while continue_condition:
            old_centroids = self.cluster_centers_.copy()

            self.y_pred = np.argmin(self.dist(X, self.cluster_centers_), axis=1)

            for i in set(self.y_pred):
                cluster_points = X[self.y_pred == i]
                self.cluster_centers_[i] = np.mean(cluster_points, axis=0)

            if (old_centroids == self.cluster_centers_).all():
                continue_condition = False

    def predict(self, X):
        distances = self.dist(X, self.cluster_centers_)
        y_pred = np.argmin(distances, axis=1)
        return y_pred
