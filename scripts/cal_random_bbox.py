import json
import numpy as np
import cv2
import os
import random

def bbox_area(bbox):
    """Calculate the area of a bounding box."""
    x_min, y_min, width, height = bbox
    return width * height

def load_json(file_path):
    """Load JSON file."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def iou(bbox1, bbox2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x_min1, y_min1, width1, height1 = bbox1
    x_max1 = x_min1 + width1
    y_max1 = y_min1 + height1
    
    x_min2, y_min2, width2, height2 = bbox2
    x_max2 = x_min2 + width2
    y_max2 = y_min2 + height2

    # Calculate the coordinates of the intersection rectangle
    x_min_inter = max(x_min1, x_min2)
    y_min_inter = max(y_min1, y_min2)
    x_max_inter = min(x_max1, x_max2)
    y_max_inter = min(y_max1, y_max2)
    
    # Calculate area of intersection rectangle
    width_inter = max(0, x_max_inter - x_min_inter)
    height_inter = max(0, y_max_inter - y_min_inter)
    area_inter = width_inter * height_inter
    
    # Calculate areas
    area_bbox1 = bbox_area(bbox1)
    area_bbox2 = bbox_area(bbox2)
    
    # Calculate IoU
    iou_value = area_inter / (area_bbox1 + area_bbox2 - area_inter)
    return iou_value

def generate_random_bbox(image_shape, area_fraction_range=(0.1, 0.3)):
    """Generate a random bounding box that covers a given fraction of the image area."""
    img_height, img_width = image_shape[:2]
    img_area = img_height * img_width
    
    # Randomly select a fraction of the area
    target_area = random.uniform(*area_fraction_range) * img_area
    
    # Randomly select aspect ratio between 0.5 and 2.0
    aspect_ratio = random.uniform(0.5, 2.0)
    
    # Calculate width and height based on the target area and aspect ratio
    height = int(np.sqrt(target_area / aspect_ratio))
    width = int(aspect_ratio * height)
    
    # Ensure the bounding box fits within the image
    width = min(width, img_width)
    height = min(height, img_height)
    
    # Randomly select top-left corner within valid range
    x_min = random.randint(0, img_width - width)
    y_min = random.randint(0, img_height - height)
    
    return [x_min, y_min, width, height]

def overlay_bbox_on_image(image_path, bbox, output_path):
    """Overlay bounding box on image and save it."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    
    x_min, y_min, width, height = bbox
    x_max = x_min + width
    y_max = y_min + height
    
    # Draw bounding box
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)
    
    # Save the result image
    result_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving
    cv2.imwrite(output_path, result_image)
    return image.shape
    
def process_image_and_compare(json_file, original_image_path, output_overlay_dir, idx):
    """Process images and compare bounding boxes."""
    # Load bounding boxes from JSON file
    bbox_data = load_json(json_file)
    
    # Create output path for overlay result
    output_overlay_path = os.path.join(output_overlay_dir, f'overlay_result_{os.path.basename(json_file).split("_")[2].split(".")[0]}.jpg')
    
    shape = cv2.imread(original_image_path).shape
    # Extract bounding box from attention map
    target_bbox = generate_random_bbox(shape)
    # Overlay bbox on the original image
    shape = overlay_bbox_on_image(original_image_path, target_bbox, output_overlay_path)

    # Process each bbox in the JSON
    for entry in bbox_data:
        bbox = entry["bbox"]
        category_name = entry["category_name"]
    
        # Compute IOU between the bbox and the target bbox
        iou_value = iou(bbox, target_bbox)
        print(f"IOU {idx} between {category_name} bbox and target bbox: {iou_value:.4f}")
    return iou_value 

def process_directory(bbox_info_dir, images_dir, output_overlay_dir):
    """Process all files in the specified directories."""
    os.makedirs(output_overlay_dir, exist_ok=True)
    all_ious = []

    # Iterate over JSON files in the bbox_info directory
    for json_file in os.listdir(bbox_info_dir):
        if not json_file.endswith('.json'):
            continue  # Skip non-JSON files

        # Extract the index number from the filename
        index = json_file.split('_')[2].split('.')[0]

        # Construct corresponding paths
        json_file_path = os.path.join(bbox_info_dir, json_file)
        generated_image_path = os.path.join(images_dir, f'image_{index}_321.png')

        # Check if corresponding files exist
        if os.path.exists(generated_image_path):
            iou = process_image_and_compare(json_file_path, generated_image_path, output_overlay_dir, index)
            if iou is not None:
                if iou >= 0.00001:
                    all_ious.append(iou)
        else:
            pass
            #print(f"Missing corresponding files for index {index}")
    
    if all_ious:
        overall_avg_iou = sum(all_ious) / len(all_ious)
        print(f"Overall Average IoU for all images: {overall_avg_iou:.4f}")
    else:
        print("No IoU values calculated.")
        
# Example usage
bbox_info_dir = "./output_images/bbox_info"
images_dir = "./i2am/results"
output_overlay_dir = "./i2am/random_overlay"

process_directory(bbox_info_dir, images_dir, output_overlay_dir)
