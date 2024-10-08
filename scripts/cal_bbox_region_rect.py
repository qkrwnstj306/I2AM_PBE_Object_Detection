import json
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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

def load_attention_map(image_path):
    """Load attention map with jet colormap, convert to binary mask."""
    # 1. 이미지를 컬러로 불러오기
    image = plt.imread(image_path)

    # jet 컬러 맵을 사용하여 색상 값을 원래 값으로 변환
    jet = cm.get_cmap('jet')
    norm = plt.Normalize(vmin=0, vmax=1)
    
    def get_value_from_color(color):
        """Convert color from jet colormap to corresponding value."""
        norm_color = np.array(color[:3])
        values = np.linspace(0, 1, jet.N)
        colormap_colors = jet(values)[:, :3]
        diffs = np.sqrt(np.sum((colormap_colors - norm_color) ** 2, axis=1))
        closest_index = np.argmin(diffs)
        return values[closest_index]

    def image_to_values(image):
        """Convert the colormap image to value map."""
        values = np.zeros((image.shape[0], image.shape[1]))
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                color = image[i, j]
                values[i, j] = get_value_from_color(color)
        return values

    # 이미지에서 값을 추출
    values = image_to_values(image)

    # 2. 이진화 (예: 상위 40%의 값들을 관심 영역으로 간주)
    threshold_value = np.percentile(values, 65)  # 상위 35%의 값을 기준으로 threshold 설정
    binary_mask = (values > threshold_value).astype(np.uint8) * 255

    # 3. 형태학적 연산을 적용하여 마스크를 정제
    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)
    
    return binary_mask

# 겹치는 영역을 계산하는 함수
def compute_overlap(binary_map, bounding_box):
    x, y, w, h = bounding_box
    # Bounding box 영역을 binary map에 맞춰 자릅니다
    cropped_binary_map = binary_map[y:y+h, x:x+w]
    # 겹치는 영역에서 1인 픽셀의 개수를 계산합니다
    overlap = np.sum(cropped_binary_map == 255)
    return overlap

def get_bounding_coords(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)  # Get the 4 corners of the rectangle
    box = np.int0(box)  # Convert to integer
    
    # Get the bounding box coordinates
    x_min = min(box[:, 0])
    y_min = min(box[:, 1])
    x_max = max(box[:, 0])
    y_max = max(box[:, 1])
    
    w = x_max - x_min
    h = y_max - y_min
    
    return [x_min, y_min, w, h], box

def bbox_from_mask(binary_mask):
    """Extract bounding box from binary mask."""
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        print("ERROR")
        return None
    
    # Find the contour
    # Bounding box와 겹치는 영역을 계산하여 가장 큰 겹침 영역을 가진 contour 선택
    max_overlap = 0
    best_contour = None
    for contour in contours:
        bounding_box = cv2.boundingRect(contour)
        #bounding_box, box = get_bounding_coords(contour)
        overlap = compute_overlap(binary_mask, bounding_box)
        if overlap > max_overlap:
            max_overlap = overlap
            best_contour = contour
    
    return cv2.boundingRect(best_contour)

def overlay_bbox_on_image(image_path, bbox, output_path):
    """Overlay bounding box on image and save it."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    
    x_min, y_min, width, height = bbox
    x_max = x_min + width
    y_max = y_min + height
    
    # Draw bounding box
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)
    # Draw the bounding box (contour)
    #cv2.drawContours(image, [box], 0, (0, 255, 255), 2)
    
    # Save the result image
    result_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving
    cv2.imwrite(output_path, result_image)

def process_image_and_compare(json_file, attention_map_path, original_image_path, output_overlay_dir, idx):
    """Process images and compare bounding boxes."""
    # Load bounding boxes from JSON file
    bbox_data = load_json(json_file)

    # Load and process attention map
    attention_map = load_attention_map(attention_map_path)
    
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
        bbox = entry["adjusted_bbox"]
        category_name = entry["category_name"]
        
        # Compute IOU between the bbox and the target bbox
        iou_value = iou(bbox, target_bbox)
        print(f"IOU {idx} between {category_name} bbox and target bbox: {iou_value:.4f}")
    return iou_value 

def process_directory(bbox_info_dir, attention_maps_dir, images_dir, output_overlay_dir):
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
            iou = process_image_and_compare(json_file_path, attention_map_path, generated_image_path, output_overlay_dir, index)
            if iou is not None:
                #if iou >= 0.00001:
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

process_directory(bbox_info_dir, attention_maps_dir, images_dir, output_overlay_dir)
