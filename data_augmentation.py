from keras.preprocessing.image import ImageDataGenerator
import os
from matplotlib import pyplot as plt
from collections import defaultdict

class DatasetPreparation:
    def __init__(self, train_data_dir='data/train/', validation_data_dir='data/test/', img_height=48, img_width=48, batch_size=32):
        self.train_data_dir = train_data_dir
        self.validation_data_dir = validation_data_dir
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size

    def get_dataset_statistics(self):
        category_counts = defaultdict(int)
        total_images = 0

        for category in os.listdir(self.train_data_dir):
            category_folder = os.path.join(self.train_data_dir, category)
            if os.path.isdir(category_folder):
                num_images = len([img for img in os.listdir(category_folder) if os.path.isfile(os.path.join(category_folder, img))])
                category_counts[category] = num_images
                total_images += num_images

        average_images_per_category = total_images / len(category_counts)
        return total_images, average_images_per_category, category_counts

    def plot_dataset_statistics(self, total_images, average_images_per_category, category_counts):
        categories = list(category_counts.keys())
        counts = list(category_counts.values())

        plt.figure(figsize=(14, 7))

        plt.subplot(1, 2, 1)
        plt.bar(categories, counts, color='skyblue')
        plt.xlabel('Categories')
        plt.ylabel('Number of Images')
        plt.title('Number of Images per Category')
        plt.xticks(rotation=45)

        plt.subplot(1, 2, 2)
        plt.pie(counts, labels=categories, autopct='%1.1f%%', startangle=140)
        plt.title('Image Distribution per Category')

        plt.tight_layout()
        plt.show()

    def prepare_data_generators(self):
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
            self.train_data_dir,
            color_mode='grayscale',
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )

        validation_generator = validation_datagen.flow_from_directory(
            self.validation_data_dir,
            color_mode='grayscale',
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )

        return train_generator, validation_generator

class ImageDisplay:
    @staticmethod
    def display_augmented_images(generator, num_images, img_height, img_width):
        images, _ = next(generator)
        plt.figure(figsize=(20, 10))
        for i in range(num_images):
            plt.subplot(4, 8, i + 1)
            plt.imshow(images[i].reshape(img_height, img_width), cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    dataset_prep = DatasetPreparation()
    total_images, average_images_per_category, category_counts = dataset_prep.get_dataset_statistics()
    dataset_prep.plot_dataset_statistics(total_images, average_images_per_category, category_counts)
    train_generator, _ = dataset_prep.prepare_data_generators()
    ImageDisplay.display_augmented_images(train_generator, 32, dataset_prep.img_height, dataset_prep.img_width)
