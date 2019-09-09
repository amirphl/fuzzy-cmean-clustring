import numpy as np
import random
import logging

EPSILON = 0.00001


def compute_distance_squared(x, c):
    sum_of_sq = 0.0
    for i in range(len(x)):
        sum_of_sq += (x[i] - c[i]) ** 2
    return sum_of_sq


class FCM:

    def __init__(self, n_clusters=2, m=2, max_iter=10):
        """
                n_clusters: number of clusters
                m: weighting parameter
                max_iter: number of iterations
            """
        self.n_clusters = n_clusters
        self.cluster_centers_ = None
        self.u = None  # The membership matrix
        self.m = m
        self.max_iter = max_iter
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(logging.StreamHandler())
        self.logger.setLevel(logging.DEBUG)

    def initiate_random_membership(self, num_of_points):
        self.u = np.zeros((num_of_points, self.n_clusters))
        for i in range(num_of_points):
            row_sum = 0.0
            for c in range(self.n_clusters):
                if c == self.n_clusters - 1:
                    self.u[i][c] = 1.0 - row_sum
                else:
                    rand_clus = random.randint(0, self.n_clusters - 1)
                    rand_num = random.random()
                    rand_num = round(rand_num, 2)
                    if rand_num + row_sum <= 1.0:  # to prevent membership sum for a point to be larger than 1.0
                        self.u[i][rand_clus] = rand_num
                        row_sum += self.u[i][rand_clus]

    def compute_cluster_centers(self, X):
        num_of_points = X.shape[0]
        num_of_features = X.shape[1]
        centers = []

        for c in range(self.n_clusters):
            sum1_vec = np.zeros(num_of_features)
            sum2_vec = 0.0
            for i in range(num_of_points):
                interm1 = (self.u[i][c] ** self.m)
                interm2 = interm1 * X[i]
                sum1_vec += interm2
                sum2_vec += interm1
                if np.any(np.isnan(sum1_vec)):
                    raise Exception("There is a nan in compute_cluster_centers method.")
            if sum2_vec == 0:
                sum2_vec = 0.000001
            centers.append(sum1_vec / sum2_vec)

        self.cluster_centers_ = centers
        return centers

    def compute_distances(self, X):
        distances = np.zeros((X.shape[0], len(self.cluster_centers_)), dtype=np.float)
        for i in range(X.shape[0]):
            for c in range(len(self.cluster_centers_)):
                distances[i][c] = compute_distance_squared(X[i], self.cluster_centers_[c])
        return distances

    def update_membership(self, X):
        distances = self.compute_distances(X)
        for i in range(X.shape[0]):
            for clus in range(len(self.cluster_centers_)):
                d1 = distances[i][clus]
                sum1 = 0.0
                for c in range(len(self.cluster_centers_)):
                    d2 = distances[i][c]
                    if d2 == 0.0:
                        d2 = EPSILON
                    sum1 += (d1 / d2) ** (2.0 / (self.m - 1))
                    if np.any(np.isnan(sum1)):
                        raise Exception("NaN is found in update_membership_method in the inner for")
                if sum1 == 0:  # TODO
                    self.u[i][clus] = 1.0 - EPSILON
                if np.any(np.isnan(sum1 ** -1)):
                    raise Exception("NaN is found in update_memberhip method")
                self.u[i][clus] = sum1 ** -1

    def learn(self, X):
        num_of_points = X.shape[0]
        self.initiate_random_membership(num_of_points)
        list_of_centers = []
        last_centers = None
        for i in range(self.max_iter):
            centers = self.compute_cluster_centers(X)
            list_of_centers.append(centers)
            last_centers = centers
            self.update_membership(X)
        print("centers:\n")
        print(last_centers)
        print("\n\n")
        return self.u
