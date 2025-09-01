import json
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join



def load_image_data(image_dir):
    """
    Load list of all image filenames
    
    args:
        image_dir: String of the path to the images.
    
    returns:
        all_image_files: List of strings containing the filename of each images of our dataset.
    
    """
    
    all_image_files = []
    
    # List all files in the directory
    all_image_files = [f.split(".jpg")[0] for f in listdir(image_dir) 
                    if isfile(join(image_dir, f)) and f.endswith(".jpg")]
    
    
    return all_image_files


def load_labels_and_classes(label_dir, classes_dir):
    """
    Load labels and classes from json files
    
    args: 
        label_dir: String of the path to the labels json file.
        classes_dir: String of the path to the classes json file.
    
    returns: 
        labels: Dict of labels for each image.
        classes: Dict of class label (number) and class name pairs.
    """
    
    labels = None
    classes = None
    
    # Load labels JSON file
    with open(label_dir, "r") as f:
        labels = json.load(f)  

    # Load class mapping JSON file
    with open(classes_dir, "r") as f:
        classes = json.load(f)      
    

    return labels, classes


def split_images_into_train_test(all_image_files, labels):
    """
    Split dataset into train and test images based on which image has labels or not.
    
    args:
        all_image_files: List of all images in our dataset (each element in the list is a filename of the image itself).
        labels: Dict with training set image filename as keys, and classes present in each image as values. 
    returns:
        train_images: List of images (image filename) with labels (100 images).
        test_images: List of images (image filename) without labels  (902 images).
    """
    
    train_images = []
    test_images = []
    
    # Images that have labels (training set)
    train_images = [img for img in all_image_files if img in labels]

    # Images that do not have labels (test set)
    test_images = [img for img in all_image_files if img not in labels]


    return train_images, test_images


def load_image(image_directory, image_filename):
    """
    Load a given image.
    
    args: 
        root_dir: String of image directory.
        image_filename: String of image filename to be loaded.
    returns:
        image_source: Numpy array representation of the image
        img_fn: String of the image filename without .jpg.
    """
    
    img_path = f'{image_directory}/{image_filename}.jpg'
    image_source = cv2.imread(img_path)
    image_source = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)[300:, 760:, :]
    
    img_fn = image_filename.split('.')[0]
    
    return image_source, img_fn


def convert_labels_to_multi_hot_encoding(labels, classes):
    """
    Convert a dict of image_filname: {img_nr, class_labels} to multi-hot encoding array
    of size (N, n_classes), where N is the number of samples in labels (length of labels)
    and n_classes is the number of classes.
    
    Note that each element in the multi-hot encoding can only take on values {0,1} 
    depending on whether the class is present in the sample or not. 
    Also note that each row represents one sample, and that, unlike one-hot encoding, several elements in the row can be 1 in our multi-hot encoding. 
    
    args:
        labels: Dict with image filename as keys, and classes present in each image as values.
        classes: Dict of class label (number) and class name pairs.
    returns:
        labels_multi_hot: NumPy array of shape (N, n_classes) representing multi-hot encoding (or binary encoding for multi-label classification).
    """
    
    num_samples = len(labels)
    num_classes = len(classes)
    
    labels_multi_hot = None
    
    labels_multi_hot = np.zeros((num_samples, num_classes), dtype=np.float32)

    for i, (image_filename, label_info) in enumerate(labels.items()):
        class_indices = label_info["labels"]  # Get class indices for this image
        labels_multi_hot[i, class_indices] = 1  # Set corresponding indices to 1

    
    return labels_multi_hot
            
            
def sort_labels(labels):
    """
    Convert to dict items and sort by image_nr
    """
    
    sorted_dict = dict(sorted(
        labels.items(), 
        key=lambda x: x[1]['image_nr']
    ))
    
    return sorted_dict