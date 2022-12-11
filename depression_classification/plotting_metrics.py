import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion_matrix(labels, predictions, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    """
    # in case running on gpu, need to convert to cpu
    labels = labels.cpu()
    predictions = predictions.cpu()

    disp = ConfusionMatrixDisplay.from_predictions(
        y_true=labels,
        y_pred=predictions,
        display_labels=class_names,
        xticks_rotation=90,
    )
    plt.tight_layout()
    return disp.figure_


def plot_prf1_per_class(prf1, class_names):
    """make matrix from prf1 dictionary"""
    precision = prf1['p-none']
    recall = prf1['r-none']
    f1 = prf1['f1-none']
    matrix = torch.stack([precision, recall, f1]).cpu().numpy()

    num_classes = len(class_names)
    figure = plt.figure()
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.RdYlGn,
               vmax=1, vmin=0)
    plt.title(f"prf1 per class")
    plt.colorbar()
    xtick_marks = np.arange(num_classes)
    plt.xticks(xtick_marks, class_names, rotation=45, ha='right', va='top')
    ytick_marks = np.arange(matrix.shape[0])
    plt.yticks(ytick_marks, ['precision', 'recall', 'f1'])

    # Use white text if squares are dark; otherwise black.
    for i, j in itertools.product(range(matrix.shape[0]),
                                  range(matrix.shape[1])):
        color = "white" if matrix[i, j] < .2 or matrix[i, j] > .8 else "black"
        plt.text(j, i, f"{matrix[i, j]:.2f}",
                 horizontalalignment="center", color=color)

    plt.ylabel('Metric')
    plt.xlabel('Class')
    plt.tight_layout()
    return figure
