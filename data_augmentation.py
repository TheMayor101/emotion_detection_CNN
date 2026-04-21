"""
Contains function responsible for modifying the images before sending them as input for the model
"""

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import DirectoryIterator
import os
from matplotlib import pyplot as plt
from collections import defaultdict

# Image dimensions - the images in the dataset are 48 by 48 pixels
IMG_HEIGHT: int = 48
IMG_WIDTH: int = 48

# Batch size - chosen after testing different options
batch_size: int = 32

# Directories for training and validation datasets
train_data_dir: str = 'data/train/'
validation_data_dir: str = 'data/test/'
#I used the images inside the test folder for validation for the project


def count_images(data_dir_path: str) -> defaultdict[str, int]:
    """
    Counts number of images in the dataset, in each emotion category
    """
    category_num: defaultdict[str, int] = defaultdict(int)
    for category in os.listdir(data_dir_path):
        category_folder: str = os.path.join(data_dir_path, category)
        if os.path.isdir(category_folder):
            num_images: int = len([
                img for img in os.listdir(category_folder)
                if os.path.isfile(os.path.join(category_folder, img))
            ])
            category_num[category] = num_images
    return category_num


# Count the number of images in each category for training and validation datasets
train_category_num: defaultdict[str, int] = count_images(train_data_dir)
validation_category_num: defaultdict[str, int] = count_images(
    validation_data_dir
)

# Calculate total and average number of images for training and validation datasets
total_train_images: int = sum(train_category_num.values())
average_train_images_per_category: float = (
    total_train_images / len(train_category_num)
)

total_validation_images: int = sum(validation_category_num.values())
average_validation_images_per_category: float = (
    total_validation_images / len(validation_category_num)
)

print("The Total amount of training images is: " + str(total_train_images))
print(
    "The Average amount of training images in every category is: "
    + "{:.2f}".format(average_train_images_per_category)
)
print(
    "The Total number of validation images: "
    + str(total_validation_images)
)
print(
    "The Average number of validation images per category: "
    + "{:.2f}".format(average_validation_images_per_category)
)

# Plot the data distribution for training and validation datasets
train_categories: list[str] = list(train_category_num.keys())
train_amount: list[int] = list(train_category_num.values())
validation_categories: list[str] = list(validation_category_num.keys())
validation_counts: list[int] = list(validation_category_num.values())

plt.figure(figsize=(18, 10))

# Bar chart for training data
plt.subplot(2, 2, 1)
plt.bar(train_categories, train_amount, color='skyblue')
plt.xlabel('Categories')
plt.ylabel('Number of Images')
plt.title('Number of Training Images per Category')
plt.xticks(rotation=45)

# Pie chart for training data
plt.subplot(2, 2, 2)
plt.pie(train_amount, labels=train_categories, autopct='%1.1f%%', startangle=140)
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
plt.pie(
    validation_counts,
    labels=validation_categories,
    autopct='%1.1f%%',
    startangle=140,
)
plt.title('Validation Image Distribution per Category')

plt.tight_layout()
plt.show()

# Data augmentation for training images using ImageDataGenerator
train_datagen: ImageDataGenerator = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=10,
    shear_range=0.1,
    zoom_range=0.1,
    height_shift_range=0.1,
    width_shift_range=0.1,
    brightness_range=[0.9, 1.1],
    horizontal_flip=True,
    fill_mode='nearest',
)

# Data augmentation for validation images (only normalization)
validation_datagen: ImageDataGenerator = ImageDataGenerator(
    rescale=1. / 255
)

# Create data generators for training and validation datasets
train_generator: DirectoryIterator = train_datagen.flow_from_directory(
    train_data_dir,
    color_mode='grayscale',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
)

validation_generator: DirectoryIterator = (
    validation_datagen.flow_from_directory(
        validation_data_dir,
        color_mode='grayscale',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
    )
)


def display_augmented_images(
    generator: DirectoryIterator,
    num_images: int,
) -> None:
    """
    Displaying a few Augmented images to show how the images look after augmentation
    """
    images, _ = next(generator)
    plt.figure(figsize=(20, 10))
    for i in range(num_images):
        plt.subplot(4, 8, i + 1)
        plt.imshow(images[i].reshape(IMG_HEIGHT, IMG_WIDTH), cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    display_augmented_images(train_generator, 32)
