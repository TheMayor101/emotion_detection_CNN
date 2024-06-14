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
    epochs = range(1, len(loss) + 1)  

    # Plotting the training and validation loss
    plt.plot(epochs, loss, 'y', label='Training loss')  
    plt.plot(epochs, val_loss, 'r', label='Validation loss')  
    plt.title('Training and validation loss')  
    plt.xlabel('Epochs')  
    plt.ylabel('Loss')  
    plt.legend()  
    plt.show()  

    # Extracting the accuracy values from the history object
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    # Plotting the training and validation accuracy
    plt.plot(epochs, acc, 'y', label='Training accuracy') 
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')  
    plt.title('Training and validation accuracy')  
    plt.xlabel('Epochs')  
    plt.ylabel('Accuracy')  
    plt.legend() 
    plt.show()  



