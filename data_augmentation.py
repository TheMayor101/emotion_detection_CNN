from keras.preprocessing.image import ImageDataGenerator
import os
from matplotlib import pyplot as plt
from collections import defaultdict

# The images I found in the dataset are 48 by 48 pixels.
IMG_HEIGHT = 48
IMG_WIDTH = 48

# I chose to use a batch size of 32 after testing other options.
batch_size = 32

train_data_dir = 'data/train/'
validation_data_dir = 'data/test/'

# Function to count images in each category
def count_images(data_dir):
    category_counts = defaultdict(int)
    for category in os.listdir(data_dir):
        category_folder = os.path.join(data_dir, category)
        if os.path.isdir(category_folder):
            num_images = len([img for img in os.listdir(category_folder) if os.path.isfile(os.path.join(category_folder, img))])
            category_counts[category] = num_images
    return category_counts

# Count images in training and validation datasets
train_category_counts = count_images(train_data_dir)
validation_category_counts = count_images(validation_data_dir)

# Calculate additional statistics
total_train_images = sum(train_category_counts.values())
average_train_images_per_category = total_train_images / len(train_category_counts)

total_validation_images = sum(validation_category_counts.values())
average_validation_images_per_category = total_validation_images / len(validation_category_counts)

# Print dataset statistics
print(f"Total number of training images: {total_train_images}")
print(f"Average number of training images per category: {average_train_images_per_category:.2f}")
print(f"Total number of validation images: {total_validation_images}")
print(f"Average number of validation images per category: {average_validation_images_per_category:.2f}")

# Plot the data
train_categories = list(train_category_counts.keys())
train_counts = list(train_category_counts.values())
validation_categories = list(validation_category_counts.keys())
validation_counts = list(validation_category_counts.values())

plt.figure(figsize=(18, 10))

# Training data bar chart
plt.subplot(2, 2, 1)
plt.bar(train_categories, train_counts, color='skyblue')
plt.xlabel('Categories')
plt.ylabel('Number of Images')
plt.title('Number of Training Images per Category')
plt.xticks(rotation=45)

# Training data pie chart
plt.subplot(2, 2, 2)
plt.pie(train_counts, labels=train_categories, autopct='%1.1f%%', startangle=140)
plt.title('Training Image Distribution per Category')

# Validation data bar chart
plt.subplot(2, 2, 3)
plt.bar(validation_categories, validation_counts, color='lightgreen')
plt.xlabel('Categories')
plt.ylabel('Number of Images')
plt.title('Number of Validation Images per Category')
plt.xticks(rotation=45)

# Validation data pie chart
plt.subplot(2, 2, 4)
plt.pie(validation_counts, labels=validation_categories, autopct='%1.1f%%', startangle=140)
plt.title('Validation Image Distribution per Category')

plt.tight_layout()
plt.show()

# Data augmentation
# I used the function ImageDataGenerator to prepare my images
train_datagen = ImageDataGenerator(
    rescale=1. / 255, # we prefer to work with values between 0-1 and not 0-255 because it will be easier for the models calculation
    rotation_range=10, # we rotate the image by 10 degrees each time
    shear_range=0.1,
    zoom_range=0.1,
    height_shift_range=0.1,
    width_shift_range=0.1,
    brightness_range=[0.9, 1.1], # changing the brightness of the images to values of between 0.9 to 1.1
    horizontal_flip=True,
    fill_mode='nearest' # in case that a pixel is erased or added we will fill the new one or create a new one with a value of the average of the values around him
)
# using the above function I was able to create many more images and prevent overfitting

# we prefer to work with values between 0-1 and not 0-255 because it will be easier for the models calculation
validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    color_mode='grayscale', # I chose to use grayscale because color is not important for my models accuracy and using RGB will make it much slower
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True # I chose that on each epoch the model takes new random images in order that it will not just memorize the order
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    color_mode='grayscale',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

# Here I show a few examples of how the images look after they are augmented.
def display_augmented_images(generator, num_images):
    images, _ = next(generator)
    plt.figure(figsize=(20, 10))
    for i in range(num_images):
        plt.subplot(4, 8, i + 1)
        plt.imshow(images[i].reshape(IMG_HEIGHT, IMG_WIDTH), cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Display a few augmented images from each category
def display_augmented_images_per_category(generator, num_images):
    for category in generator.class_indices.keys():
        print(f"Category: {category}")
        category_generator = generator
        images, _ = next(category_generator)
        plt.figure(figsize=(20, 10))
        for i in range(num_images):
            plt.subplot(4, 8, i + 1)
            plt.imshow(images[i].reshape(IMG_HEIGHT, IMG_WIDTH), cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    display_augmented_images(train_generator, 32) # showing the newly created images after the augmentation
    display_augmented_images_per_category(train_generator, 8) # showing augmented images for each category
