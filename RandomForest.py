import numpy as np
from collections import Counter

from DecisionTree import DecisionTree
class RandomForest:
    def __init__(self, max_depth=100, B = 64, number_feature=None, randomizing_features= False):
        self.max_depth = max_depth
        self.B = B
        self.trees = []
        self.number_features = number_feature
        self.randomizing_features = randomizing_features

    def fit(self, X, y):
        self.trees = []

        for b in range(self.B):
            tree = DecisionTree(max_depth=self.max_depth,randomizing_features=self.randomizing_features)
            x_sample, y_sample = self._bootstrap(X, y)
            tree.fit(x_sample,y_sample)
            self.trees.append(tree)

    def _bootstrap(self, X, y):
        number_sample = X.shape[0]
        indices = np.random.choice(number_sample, size=number_sample, replace=True)
        return X[indices], y[indices]

    def predict(self, X):
        all_predictions = []
        for tree in self.trees:
            predictions = tree.predict(X)
            all_predictions.append(predictions)

        all_predictions = np.array(all_predictions).T

        final_predictions = []
        for sample_predictions in all_predictions:
            counter = Counter(sample_predictions)
            most_common = counter.most_common(1)[0][0]
            final_predictions.append(most_common)

        return final_predictions








