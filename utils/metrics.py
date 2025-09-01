import numpy as np
from sklearn.metrics import balanced_accuracy_score



def compute_balanced_accuracy(preds, labels):
    """
    Compute the mean balanced accuracy over all classes.
        1) Iterate over each class present and compute the balanced accuracy over all samples using balanced_accuracy_score(...)
        2) Compute the mean score over all classes
        
    args:
        preds: Numpy array of shape (num_samples, num_classes) with predictions.
        labels: Numpy array of shape (num_samples, num_classes) with labels.
    returns:
        mean_balanced_accuracy: A single average balanced accuracy score computed over all classes
        accuracies: A list of balanced accuracy for each class
    """
    
    accuracies = []
    
    # Iterate over each class and compute balanced accuracy
    for class_idx in range(labels.shape[1]):  # num_classes
        class_accuracy = balanced_accuracy_score(labels[:, class_idx], preds[:, class_idx], adjusted=False)
        accuracies.append(class_accuracy)  # Store the accuracy of this class

    # Compute mean balanced accuracy across all classes
    mean_balanced_accuracy = np.mean(accuracies)
  
    return mean_balanced_accuracy, accuracies