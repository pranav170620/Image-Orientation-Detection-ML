import joblib
import numpy as np
import os
import cv2  # OpenCV for resizing images

def generate_rotated_sub_images(images, size):
    sub_images = []
    labels = []
    for img in images:
        # Randomly select a starting point for the square sub-image
        x = np.random.randint(0, img.shape[0] - size)
        y = np.random.randint(0, img.shape[1] - size)
        sub_img = img[x:x + size, y:y + size]

        # Generate rotated versions
        for i in range(4):
            rotated_img = np.rot90(sub_img, k=i)
            sub_images.append(rotated_img.flatten())
            labels.append(i)

    return np.array(sub_images), np.array(labels)

def save_data(data, filename):
    joblib.dump(data, filename, compress=('zlib', 3), protocol=4)

def prepare_data(file_path, sizes, max_file_size):
    images, _ = joblib.load(file_path)
    data = {}

    for size in sizes:
        sub_images, labels = generate_rotated_sub_images(images, size)

        # Save the data to a temporary file
        temp_filename = f'model_{size}_temp.joblib'
        save_data({'images': sub_images, 'labels': labels}, temp_filename)

        data[str(size)] = {
            'images': sub_images,
            'labels': labels
        }

    return data

# Define the sizes of the sub-images
sub_image_sizes = [90, 50, 30]

# Define the maximum file size in bytes (20 MB)
max_file_size = 20 * 1024 * 1024

# Load and process the training data
train_data = prepare_data('train.full.joblib', sub_image_sizes, max_file_size)

# Save the processed data
joblib.dump(train_data, 'prepared_data.joblib', compress=('zlib', 3), protocol=4)