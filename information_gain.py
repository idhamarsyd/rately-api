import numpy as np

class InformationGain():
    def entropy(self, labels):
        _, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))  # Adding a small value to prevent log(0)

    def conditional_entropy(self, feature, labels):
        unique_values, value_counts = np.unique(feature, return_counts=True)
        total_entropy = 0

        for value, count in zip(unique_values, value_counts):
            subset_labels = labels[feature == value]
            total_entropy += (count / len(feature)) * self.entropy(subset_labels)

        return total_entropy

    def mutual_info_classif(self, X, y):
        mi_scores = []

        # Calculate entropy of the target variable
        target_entropy = self.entropy(y)

        # Calculate mutual information for each feature
        for feature_column in X.T:
            mi_score = target_entropy - self.conditional_entropy(feature_column, y)
            mi_scores.append(mi_score)

        return np.array(mi_scores)

# Example usage:
# X_train is your feature matrix, and y_train is your target variable
# mi_scores = mutual_info_classif(X_train, y_train)
