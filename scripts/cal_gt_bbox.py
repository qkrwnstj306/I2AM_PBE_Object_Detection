import os
import json
import cv2

def load_json(file_path):
    """Load JSON file."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def overlay_bbox_on_image(image_path, bbox, output_path):
    """Overlay bounding box on image and save it."""
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # Extract bounding box coordinates
    x_min, y_min, width, height = bbox
    x_max = x_min + width
    y_max = y_min + height
    
    # Draw the bounding box in red color
    cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=(0, 0, 255), thickness=2)
    
    # Save the image with overlay
    cv2.imwrite(output_path, image)

def process_directory(bbox_info_dir, output_overlay_dir):
    """Process all JSON files in the bbox_info directory to overlay bounding boxes on images."""
    os.makedirs(output_overlay_dir, exist_ok=True)

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
            bbox = entry["bbox"]
            image_path = entry["image_path"]
            
            # Construct the output path for the overlay image
            image_name = os.path.basename(image_path)
            output_image_path = os.path.join(output_overlay_dir, f'overlay_{image_name}')
            
            # Overlay the bounding box on the image
            overlay_bbox_on_image(image_path, bbox, output_image_path)
            print(f"Overlay saved to: {output_image_path}")

# Example usage
bbox_info_dir = "./output_images/bbox_info"  # Directory containing JSON files
output_overlay_dir = "./i2am/GT_overlay"     # Directory to save overlay images

process_directory(bbox_info_dir, output_overlay_dir)
