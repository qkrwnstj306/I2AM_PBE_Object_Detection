import json
import numpy as np
import cv2
import os

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

def set_edge_to_zero(binary_mask, edge_width=2):
    # 이미지 복사 (원본 이미지를 수정하지 않기 위해)
    mask_with_edges = binary_mask.copy()
    
    # 이미지의 크기 가져오기
    img_height, img_width = binary_mask.shape
    
    # 가장자리 영역을 0으로 설정
    mask_with_edges[:edge_width, :] = 0                 # 상단 가장자리
    mask_with_edges[-edge_width:, :] = 0                # 하단 가장자리
    mask_with_edges[:, :edge_width] = 0                 # 좌측 가장자리
    mask_with_edges[:, -edge_width:] = 0                # 우측 가장자리
    
    return mask_with_edges

def load_attention_map(image_path, attention_score_threshold):
    """Load attention map and convert to binary mask."""
    attention_map = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    attention_map = attention_map.astype(np.float32) / 255.0
    binary_mask = (attention_map >= attention_score_threshold).astype(np.uint8)
    
    binary_mask = set_edge_to_zero(binary_mask, edge_width=2)
    # 레이블링 수행
    from skimage import measure
    labels = measure.label(binary_mask, connectivity=2)
    properties = measure.regionprops(labels)

    # 필터링 기준 설정 (예: 최소 영역 크기)
    min_size = 1000  # 예를 들어, 500픽셀 이상만 유지

    # 빈 새 이미지 생성
    cleaned_img = np.zeros_like(binary_mask)

    # 크기가 기준보다 큰 객체만 유지
    for prop in properties:
        if prop.area >= min_size:
            for coord in prop.coords:
                cleaned_img[coord[0], coord[1]] = 255
    binary_mask = cleaned_img
    
    #Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    return binary_mask

def bbox_from_mask(binary_mask):
    """Extract bounding box from binary mask. 이진 이미지에서 개체의 윤곽선 찾기"""
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)

    return [x, y, w, h]

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

def process_image_and_compare(json_file, attention_map_path, original_image_path, output_overlay_dir, idx, attention_score_threshold):
    """Process images and compare bounding boxes."""
    # Load bounding boxes from JSON file
    bbox_data = load_json(json_file)

    # Load and process attention map
    attention_map = load_attention_map(attention_map_path, attention_score_threshold)
    
    # Extract bounding box from attention map
    target_bbox = bbox_from_mask(attention_map)
    
    if target_bbox is None:
        print(f"No valid bbox found in attention map: {attention_map_path}")
        return
    
    # Create output path for overlay result
    output_overlay_path = os.path.join(output_overlay_dir, f'overlay_result_{os.path.basename(json_file).split("_")[2].split(".")[0]}.jpg')
    
    # Overlay bbox on the original image
    overlay_bbox_on_image(original_image_path, target_bbox, output_overlay_path)

    # Process each bbox in the JSON
    for entry in bbox_data:
        bbox = entry["bbox"]
        category_name = entry["category_name"]
        
        # Compute IOU between the bbox and the target bbox
        iou_value = iou(bbox, target_bbox)
        print(f"IOU {idx} between {category_name} bbox and target bbox: {iou_value:.4f}")
    return iou_value 

def process_directory(bbox_info_dir, attention_maps_dir, images_dir, output_overlay_dir, attention_score_threshold):
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
        attention_map_path = os.path.join(attention_maps_dir, f'idx-{index}_attention_map_generated_image-True.png')
        generated_image_path = os.path.join(images_dir, f'image_{index}_321.png')

        # Check if corresponding files exist
        if os.path.exists(attention_map_path) and os.path.exists(generated_image_path):
            iou = process_image_and_compare(json_file_path, attention_map_path, generated_image_path, output_overlay_dir, index, attention_score_threshold)
            if iou is not None:
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
attention_maps_dir = "./i2am/attention"
images_dir = "./i2am/results"
output_overlay_dir = "./i2am/overlay"

attention_score_threshold = 0.5#0.75

process_directory(bbox_info_dir, attention_maps_dir, images_dir, output_overlay_dir, attention_score_threshold)
