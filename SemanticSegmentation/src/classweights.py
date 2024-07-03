import numpy as np
from pathlib import Path
import tifffile

# Define the base directory containing all the tile folders
base_dir = Path(r"hw3-semantic-segmentation-california-roll\data\raw\Train")

# Define the number of classes
num_classes = 4

# Initialize an array to hold the counts for each class
class_counts = np.zeros(num_classes)

# Iterate over each tile folder (assuming they are named Tile1, Tile2, ..., Tile60)
for tile_num in range(1, 61):
    tile_dir = base_dir / f'Tile{tile_num}'
    gt_file = tile_dir / 'groundTruth.tif'

    # Check if the ground truth file exists in the current tile directory
    if gt_file.exists():
        # Load the ground truth image
        gt_image = tifffile.imread(str(gt_file))

        # Debug: Print the unique values in the ground truth image
        unique_values = np.unique(gt_image)
        print(f"Tile {tile_num} unique values: {unique_values}")

        # Count the number of pixels for each class
        for class_idx in range(num_classes):
            class_counts[class_idx] += np.sum(gt_image == class_idx + 1)  # Assuming classes are labeled as 1, 2, 3, 4
    else:
        print(f"Ground truth file not found for Tile {tile_num}")

# Print the counts for each class
print("Class counts: ", class_counts)

# Calculate the total number of samples
total_samples = np.sum(class_counts)

# Calculate the frequency of each class
freq_classes = class_counts / total_samples

# Calculate the inverse of the frequency
inv_freq_classes = 1 / freq_classes

# Normalize the weights
class_weights = inv_freq_classes / np.sum(inv_freq_classes)

print("Class Weights: ", class_weights)