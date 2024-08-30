import os
import json
import cv2

def bbox_area(bbox):
    """Calculate the area of a bounding box."""
    x_min, y_min, width, height = bbox
    return width * height

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

def load_json(file_path):
    """Load JSON file."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def calculate_image_bbox(image_path):
    """Calculate the bounding box for the entire image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return None
    height, width = image.shape[:2]
    return [0, 0, width, height]

def process_directory(bbox_info_dir):
    """Process all JSON files in the bbox_info directory to compare image bbox with GT bboxes."""
    iou_results = []

    # Iterate over JSON files in the bbox_info directory
    for json_file in os.listdir(bbox_info_dir):
        if not json_file.endswith('.json'):
            continue  # Skip non-JSON files

        # Construct full path to the JSON file
        json_file_path = os.path.join(bbox_info_dir, json_file)
        
        # Load bounding box data from JSON
        bbox_data = load_json(json_file_path)

        # Process each entry in the JSON
        for entry in bbox_data:
            gt_bbox = entry["bbox"]
            image_path = entry["image_path"]
            
            # Calculate the bounding box for the entire image
            image_bbox = calculate_image_bbox(image_path)
            if image_bbox is None:
                continue

            # Compute IOU between the image bbox and the GT bbox
            iou_value = iou(image_bbox, gt_bbox)
            print(f"IOU between image bbox and GT bbox: {iou_value:.4f}")

            # Store the IoU result
            iou_results.append(iou_value)

    # Calculate the average IoU
    if iou_results:
        average_iou = sum(iou_results) / len(iou_results)
        print(f"Average IoU: {average_iou:.4f}")
    else:
        print("No IoU results to calculate average.")

# Example usage
bbox_info_dir = "./output_images/bbox_info"  # Directory containing JSON files
process_directory(bbox_info_dir)