import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix


def plot_confusion_matrix(cm, title="Confusion matrix", cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, "masked", rotation=45)
    plt.yticks(tick_marks, "masked")
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


def sigmod(x):
    return 1 / (1 + np.exp(-x))


def plot_unmasked_and_masked(unmasked, masked):
    fig, axs = plt.subplots(1, 2, figsize=(12, 8))

    axs[0].imshow(unmasked)
    axs[0].axis("off")
    axs[0].set_title("unmasked face")

    # plot image and add the mask
    axs[1].imshow(masked)
    axs[1].axis("off")
    axs[1].set_title("masked")

    # set suptitle
    plt.suptitle("unmasked vs masked")
    plt.show()


def correct_name(x):
    return x[:-1]
