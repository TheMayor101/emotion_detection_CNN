#This is the file which in we do all the part of preapring the dataset for the models


from keras.preprocessing.image import ImageDataGenerator
import os
from matplotlib import pyplot as plt
from collections import defaultdict

#The images I found in the data set are 48 by 48 pixels.
IMG_HEIGHT = 48
IMG_WIDTH = 48

# I chose to use a batch size of 32 after testing other options.
batch_size = 32


#from here the model takes the images that it trains on and get's tested on.
train_data_dir = 'data/train/'
validation_data_dir = 'data/test/'

# Initialize a dictionary to store the count of images in each category
category_counts = defaultdict(int)

# Iterate over each category folder in the dataset
for category in os.listdir(train_data_dir):
    category_folder = os.path.join(train_data_dir, category)
    if os.path.isdir(category_folder):
        # Count the number of images in the current category
        num_images = len([img for img in os.listdir(category_folder) if os.path.isfile(os.path.join(category_folder, img))])
        category_counts[category] = num_images

# Calculate additional statistics
total_images = sum(category_counts.values())
average_images_per_category = total_images / len(category_counts)

# Print dataset statistics
print(f"Total number of images: {total_images}")
print(f"Average number of images per category: {average_images_per_category:.2f}")

# Plot the data
categories = list(category_counts.keys())
counts = list(category_counts.values())

plt.figure(figsize=(14, 7))

# Bar chart
plt.subplot(1, 2, 1)
plt.bar(categories, counts, color='skyblue')
plt.xlabel('Categories')
plt.ylabel('Number of Images')
plt.title('Number of Images per Category')
plt.xticks(rotation=45)

# Pie chart
plt.subplot(1, 2, 2)
plt.pie(counts, labels=categories, autopct='%1.1f%%', startangle=140)
plt.title('Image Distribution per Category')

plt.tight_layout()
plt.show()

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=10,
    shear_range=0.1,
    zoom_range=0.1,
    height_shift_range=0.1,
    width_shift_range=0.1,
    brightness_range=[0.9, 1.1],
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    color_mode='grayscale',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    color_mode='grayscale',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

#Here I show a few expamples for how the images look after they are augmented.
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
    display_augmented_images(train_generator, 32)
