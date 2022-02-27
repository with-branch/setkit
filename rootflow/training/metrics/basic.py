import numpy as np


def calculate_confusion_matrix(predictions, targets, num_classes):
    assert len(predictions) == len(
        targets
    ), "Cannot calculate confusion matrix for logits and labels of different lengths"
    confusion_matrix = np.zeros((num_classes, num_classes))
    for pred, ref in zip(predictions, targets):
        confusion_matrix[pred, ref] += 1
    return confusion_matrix


def calculate_class_f1(confusion_matrix, class_index):
    true_positive = confusion_matrix[class_index, class_index]
    false_positive = np.sum(confusion_matrix[class_index, :]) - true_positive
    false_negative = np.sum(confusion_matrix[:, class_index]) - true_positive
    precision = true_positive / (true_positive + false_positive + 1e-20)
    recall = true_positive / (true_positive + false_negative + 1e-20)
    return (2 * precision * recall) / (precision + recall + 1e-20)


class Accuracy:
    def __call__(self, predictions, targets):
        accuracy_scores = []
        for task_predictions, task_targets in zip(predictions, targets):
            total = len(task_predictions)
            total_correct = np.sum(np.where(task_predictions == task_targets, 1, 0))
            accuracy_scores.append(total_correct / total)
        return {"accuracy": accuracy_scores}


class F1:
    # Can we just determine this dynamically, so we dont have any constructor arguments?
    def __init__(self, num_classes):
        if not isinstance(num_classes, (tuple, list)):
            num_classes = [num_classes]
        self.num_classes = num_classes

    def __call__(self, predictions, targets):
        f1_scores = []
        for num_classes_task, predictions_task, targets_task in zip(
            self.num_classes, predictions, targets
        ):
            f1_scores_task = []
            confusion_matrix = calculate_confusion_matrix(
                predictions_task, targets_task, num_classes_task
            )
            for class_index in range(num_classes_task):
                f1_scores_task.append(calculate_class_f1(confusion_matrix, class_index))
            f1_scores.append(f1_scores_task)
        return {"f1": f1_scores}
