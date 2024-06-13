from keras.preprocessing.image import ImageDataGenerator
import os
from matplotlib import pyplot as plt
from collections import defaultdict

# Image dimensions - the images in the dataset are 48 by 48 pixels
IMG_HEIGHT = 48
IMG_WIDTH = 48

# Batch size - chosen after testing different options
batch_size = 32

# Directories for training and validation datasets
train_data_dir = 'data/train/'
validation_data_dir = 'data/test/'
#I used the images inside the test folder for validation for the project

# Function to count images in each category within a directory for the graphs
def count_images(data_dir):
    category_counts = defaultdict(int)
    for category in os.listdir(data_dir):
        category_folder = os.path.join(data_dir, category)
        if os.path.isdir(category_folder):
            # Count the number of files (images) in each category folder
            num_images = len([img for img in os.listdir(category_folder) if os.path.isfile(os.path.join(category_folder, img))])
            category_counts[category] = num_images
    return category_counts

# Count the number of images in each category for training and validation datasets
train_category_num = count_images(train_data_dir)
validation_category_num = count_images(validation_data_dir)

# Calculate total and average number of images for training and validation datasets
total_train_images = sum(train_category_num.values())
average_train_images_per_category = total_train_images / len(train_category_num)

total_validation_images = sum(validation_category_num.values())
average_validation_images_per_category = total_validation_images / len(validation_category_num)

# Print dataset statistics
print(f"Total number of training images: {total_train_images}")
print(f"Average number of training images per category: {average_train_images_per_category:.2f}")
print(f"Total number of validation images: {total_validation_images}")
print(f"Average number of validation images per category: {average_validation_images_per_category:.2f}")

# Plot the data distribution for training and validation datasets
train_categories = list(train_category_num.keys())
train_counts = list(train_category_num.values())
validation_categories = list(validation_category_num.keys())
validation_counts = list(validation_category_num.values())

plt.figure(figsize=(18, 10))

# Bar chart for training data
plt.subplot(2, 2, 1)
plt.bar(train_categories, train_counts, color='skyblue')
plt.xlabel('Categories')
plt.ylabel('Number of Images')
plt.title('Number of Training Images per Category')
plt.xticks(rotation=45)

# Pie chart for training data
plt.subplot(2, 2, 2)
plt.pie(train_counts, labels=train_categories, autopct='%1.1f%%', startangle=140)
plt.title('Training Image Distribution per Category')

# Bar chart for validation data
plt.subplot(2, 2, 3)
plt.bar(validation_categories, validation_counts, color='lightgreen')
plt.xlabel('Categories')
plt.ylabel('Number of Images')
plt.title('Number of Validation Images per Category')
plt.xticks(rotation=45)

# Pie chart for validation data
plt.subplot(2, 2, 4)
plt.pie(validation_counts, labels=validation_categories, autopct='%1.1f%%', startangle=140)
plt.title('Validation Image Distribution per Category')

plt.tight_layout()
plt.show()

# Data augmentation for training images using ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1. / 255, # Normalize pixel values to 0-1 range
    rotation_range=10, # Randomly rotate images by up to 10 degrees
    shear_range=0.1, # Shear transformations
    zoom_range=0.1, # Randomly zoom in on images
    height_shift_range=0.1, # Randomly shift images vertically
    width_shift_range=0.1, # Randomly shift images horizontally
    brightness_range=[0.9, 1.1], # Randomly change brightness
    horizontal_flip=True, # Randomly flip images horizontally
    fill_mode='nearest' # Fill in missing pixels with the nearest pixel values
)

# Data augmentation for validation images (only normalization)
validation_datagen = ImageDataGenerator(rescale=1. / 255)

# Create data generators for training and validation datasets
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    color_mode='grayscale', # Use grayscale images
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    class_mode='categorical', # Multi-class classification
    shuffle=True # Shuffle images each epoch
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    color_mode='grayscale',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

# Function to display a batch of augmented images
def display_augmented_images(generator, num_images):
    images, _ = next(generator)
    plt.figure(figsize=(20, 10))
    for i in range(num_images):
        plt.subplot(4, 8, i + 1)
        plt.imshow(images[i].reshape(IMG_HEIGHT, IMG_WIDTH), cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Display a batch of augmented images
    display_augmented_images(train_generator, 32)

