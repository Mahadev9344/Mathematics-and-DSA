import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Define dataset and output folders
dataset_folder = r"C:\Face Recognition Dataset\processed\dataset"  # Input dataset
output_folder = r"C:\Face Recognition Dataset\processed\PCA and LBP"  # Output folder
os.makedirs(output_folder, exist_ok=True)

image_size = (200, 200)  # Resize images
radius = 1
n_points = 8 * radius

# Lists to store features and labels
lbp_images = []
labels = []
image_files = []

# Debugging: Print detected folders
if not os.path.exists(dataset_folder):
    raise ValueError(f"❌ Error: Folder '{dataset_folder}' does not exist!")

print("📂 Folders found:", os.listdir(dataset_folder))

# Loop through each person's folder
for person_folder in os.listdir(dataset_folder):
    person_path = os.path.join(dataset_folder, person_folder)
    if os.path.isdir(person_path):  # Ensure it's a folder
        print(f"🔍 Processing folder: {person_folder}")  # Debugging

        for image_file in os.listdir(person_path):
            if image_file.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(person_path, image_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    print(f"⚠ Warning: Failed to load {img_path}")
                    continue  # Skip if image cannot be loaded

                img = cv2.resize(img, image_size)

                # Extract LBP features
                lbp = local_binary_pattern(img, n_points, radius, method="uniform")
                lbp_flattened = lbp.flatten()

                lbp_images.append(lbp_flattened)  # Store LBP features
                labels.append(person_folder)  # Assign label
                image_files.append(img_path)  # Store image path

# Convert lists to NumPy arrays
if len(lbp_images) == 0:
    raise ValueError("❌ No images found! Check dataset folder path or image formats.")

lbp_images = np.array(lbp_images)  # Shape: (num_samples, num_features)
labels = np.array(labels)

print(f"✅ Successfully processed {len(lbp_images)} images.")

# Normalize LBP features before PCA
scaler = StandardScaler()
lbp_images = scaler.fit_transform(lbp_images)

# Apply PCA (retain 95% variance)
pca = PCA(0.95)
pca_features = pca.fit_transform(lbp_images)

# Save processed data
np.save(os.path.join(output_folder, "pca_features.npy"), pca_features)
np.save(os.path.join(output_folder, "labels.npy"), labels)

print(f"✅ Features and labels saved in '{output_folder}' folder!")

# Save example images (Fix shape issues)
lbp_uint8 = (lbp / lbp.max() * 255).astype(np.uint8)  # Normalize for saving
cv2.imwrite(os.path.join(output_folder, "lbp_example.png"), lbp_uint8)

reconstructed_image = pca.inverse_transform(pca_features[0]).reshape(image_size)
reconstructed_image = (reconstructed_image - reconstructed_image.min()) / (reconstructed_image.max() - reconstructed_image.min()) * 255  # Normalize
reconstructed_image = reconstructed_image.astype(np.uint8)
cv2.imwrite(os.path.join(output_folder, "pca_reconstructed_example.png"), reconstructed_image)

print("✅ Sample images saved!")
