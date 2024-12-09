import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(input_dir, output_dir, test_size=0.2):
    """
    Splits a dataset with subfolders into training and testing sets.

    Args:
    - input_dir: Path to the dataset with subfolders for each class.
    - output_dir: Path where train/test folders will be created.
    - test_size: Fraction of data to allocate to the test set.
    """
    # Define train and test directories
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Loop through each class folder
    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if os.path.isdir(class_path):
            # Get all image paths
            image_files = [os.path.join(class_path, f) for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
            
            # Split into train and test
            train_files, test_files = train_test_split(image_files, test_size=test_size, random_state=42)

            # Create class subfolders
            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

            # Move files
            for file in train_files:
                shutil.copy(file, os.path.join(train_dir, class_name))
            for file in test_files:
                shutil.copy(file, os.path.join(test_dir, class_name))
    
    print("Dataset split completed.")

# Example usage
input_dir = "/Users/shanawazeshaik/myrepo/ml-models/vehicle-recognition/vehicles"
output_dir = "/Users/shanawazeshaik/myrepo/ml-models/vehicle-recognition/"
split_dataset(input_dir, output_dir)
