# This is the file where we create all the graphs for the model's results

from typing import Any
import matplotlib.pyplot as plt


def plot_training_history(history: Any) -> None:
    """
    Plot the training and validation loss and accuracy for the model.
    """
    loss: list[float] = history.history['loss']
    val_loss: list[float] = history.history['val_loss']
    epochs: range = range(1, len(loss) + 1)

    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    acc: list[float] = history.history['accuracy']
    val_acc: list[float] = history.history['val_accuracy']

    plt.plot(epochs, acc, 'y', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
