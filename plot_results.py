# This is the file where we create all the graphs for the model's results

# I chose to use matplotlib for the graphical showing of my model's results
import matplotlib.pyplot as plt  # Importing the matplotlib library for plotting graphs

def plot_training_history(history):
    """
    Plot the training and validation loss and accuracy for the model.
    """
    # Extracting the loss values from the history object
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)  # Creating a range of epochs for the x-axis

    # Plotting the training and validation loss
    plt.plot(epochs, loss, 'y', label='Training loss')  # 'y' for yellow color
    plt.plot(epochs, val_loss, 'r', label='Validation loss')  # 'r' for red color
    plt.title('Training and validation loss')  # Title of the graph
    plt.xlabel('Epochs')  # Label for the x-axis
    plt.ylabel('Loss')  # Label for the y-axis
    plt.legend()  # Displaying the legend
    plt.show()  # Display the graph

    # Extracting the accuracy values from the history object
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    # Plotting the training and validation accuracy
    plt.plot(epochs, acc, 'y', label='Training accuracy')  # 'y' for yellow color
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')  # 'r' for red color
    plt.title('Training and validation accuracy')  # Title of the graph
    plt.xlabel('Epochs')  # Label for the x-axis
    plt.ylabel('Accuracy')  # Label for the y-axis
    plt.legend()  # Displaying the legend
    plt.show()  # Display the graph



