def calculate_accuracy(actual_labels, predicted_labels):
    if len(actual_labels) != len(predicted_labels):
        raise ValueError("The number of actual labels and predicted labels must be the same.")

    correct_predictions = sum(1 for actual, predicted in zip(actual_labels, predicted_labels) if actual == predicted)
    total_predictions = len(actual_labels)

    accuracy = correct_predictions / total_predictions
    return accuracy

def calculate_precision_recall_f1(actual_labels, predicted_labels, target_class):
    if len(actual_labels) != len(predicted_labels):
        raise ValueError("The number of actual labels and predicted labels must be the same.")

    target_class = target_class.value

    true_positives = sum(1 for actual, predicted in zip(actual_labels, predicted_labels) if actual == target_class and predicted == target_class)
    false_positives = sum(1 for actual, predicted in zip(actual_labels, predicted_labels) if actual != target_class and predicted == target_class)
    false_negatives = sum(1 for actual, predicted in zip(actual_labels, predicted_labels) if actual == target_class and predicted != target_class)
    true_negatives = sum(1 for actual, predicted in zip(actual_labels, predicted_labels) if actual != target_class and predicted != target_class)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return precision, recall, f1_score

