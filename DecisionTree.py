import numpy as np

np.random.seed(1)
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTree:
    def __init__(self, max_depth=100, randomizing_features=False):
        self.root = None
        self.max_depth = max_depth
        self.randomizing_features = randomizing_features

    def fit(self, X, y):
        self.root = self.build_tree(X, y)

    def build_tree(self, X, y, current_depth=0):
        if len(np.unique(y)) == 1 or current_depth == self.max_depth:
            return Node(value=np.argmax(np.bincount(y)))

        feature, threshold = self._best_split(X, y)
        if feature is None:
            return Node(value=np.argmax(np.bincount(y)))

        left_indices, right_indices = self._split_data(X, feature, threshold)
        left_subtree = self.build_tree(X[left_indices], y[left_indices], current_depth + 1)
        right_subtree = self.build_tree(X[right_indices], y[right_indices], current_depth + 1)

        return Node(feature=feature, threshold=threshold, left=left_subtree, right=right_subtree)

    def _entropy(self, y):
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

    def _best_split(self, X, y):
        max_info = 0
        best_feature = None
        best_threshold = None

        num_features = X.shape[1]
        if self.randomizing_features:
            # Randomly select a subset of features
            feature_indices = np.random.choice(num_features, int(np.sqrt(num_features)), replace=False)
        else:
            feature_indices = range(num_features)

        for feature in feature_indices:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._info_gain(X, y, feature, threshold)
                if gain > max_info:
                    max_info = gain
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    def _info_gain(self, X, y, feature, threshold):
        left_indices, right_indices = self._split_data(X, feature, threshold)
        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0
        parent_entropy = self._entropy(y)
        n = len(y)
        n_left, n_right = len(left_indices), len(right_indices)
        e_left, e_right = self._entropy(y[left_indices]), self._entropy(y[right_indices])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right
        return parent_entropy - child_entropy

    def _split_data(self, X, feature, threshold):
        left_indices = np.where(X[:, feature] <= threshold)[0]
        right_indices = np.where(X[:, feature] > threshold)[0]
        return left_indices, right_indices

    def predict(self, X):
        predictions = [self._predict_single_input(x, self.root) for x in X]
        return np.array(predictions)

    def _predict_single_input(self, x, node):
        if node.value is not None:  # If we are at a leaf node
            return node.value

        if x[node.feature] <= node.threshold:  # Go left
            return self._predict_single_input(x, node.left)
        else:  # Go right
            return self._predict_single_input(x, node.right)


def accuracy(predictions, y_true):
    correct = np.sum(np.array(predictions) == np.array(y_true))
    return correct / len(y_true)
