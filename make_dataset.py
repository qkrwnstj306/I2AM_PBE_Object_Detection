import os
import numpy as np
import cv2
import json
from torchvision.datasets import CocoDetection

# COCO 데이터셋 경로 설정
data_dir = "./data/coco/val2017"
instance_file = "./data/coco/annotations/instances_val2017.json"

# 출력 디렉토리 설정
output_image_dir = "./output_images/images/"
output_mask_dir = "./output_images/masks/"
output_ref_image_dir = "./output_images/reference_images/"
output_bbox_dir = "./output_images/bbox_info/"

# 출력 디렉토리 생성
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)
os.makedirs(output_ref_image_dir, exist_ok=True)
os.makedirs(output_bbox_dir, exist_ok=True)

# COCO 데이터셋 로드
dataset = CocoDetection(root=data_dir, annFile=instance_file)

# COCO 카테고리 정보 로드
coco = dataset.coco
categories = coco.loadCats(coco.getCatIds())
category_id_to_name = {cat['id']: cat['name'] for cat in categories}

print("\nCOCO Categories:")
for cat_id, cat_name in category_id_to_name.items():
    print(f"ID: {cat_id}, Name: {cat_name}")

# 최종 데이터셋을 저장할 리스트
final_dataset = []

total_num = 0

# 이미지와 바운딩 박스를 처리
for i in range(len(dataset)):
    image, target = dataset[i]
    
    # 이미지를 numpy 배열로 변환 및 BGR로 변환 (OpenCV 호환)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # [H, W, C]
    
    # 이미지의 전체 면적 계산
    image_area = image.shape[0] * image.shape[1]
    
    # bbox 정보를 저장할 리스트
    bbox_info = []

    # 대상 객체 처리
    for obj in target:
        bbox = obj["bbox"]
        category_id = obj["category_id"]
        category_name = category_id_to_name.get(category_id, "unknown")
        x_min, y_min, width, height = map(int, bbox)
        bbox_area = width * height
        
        # bbox가 이미지의 10%에서 30% 사이의 영역을 차지하는 경우에만 처리
        if 0.1 * image_area <= bbox_area <= 0.3 * image_area:
            # 1. Mask 생성 (bbox 내부는 흰색, 나머지는 검은색)
            mask = np.zeros(image.shape[:2], dtype=np.uint8)  # [H, W]
            x_max = x_min + width
            y_max = y_min + height
            mask[y_min:y_max, x_min:x_max] = 255  # 흰색으로 채우기
            
            # 2. Reference Image 생성 (bbox 영역 크롭)
            reference_image = image[y_min:y_max, x_min:x_max]
            
            # 3. 데이터셋에 추가
            final_dataset.append((image, mask, reference_image))
            
            # 4. 디버깅을 위한 이미지, 마스크, 레퍼런스 이미지 저장
            output_image_path = os.path.join(output_image_dir, f"image_{i}.jpg")
            output_mask_path = os.path.join(output_mask_dir, f"mask_{i}.png")
            output_ref_image_path = os.path.join(output_ref_image_dir, f"reference_{i}.jpg")
            
            cv2.imwrite(output_image_path, image)
            cv2.imwrite(output_mask_path, mask)
            cv2.imwrite(output_ref_image_path, reference_image)
            
            # bbox 정보 저장
            bbox_info.append({
                "bbox": bbox,
                "category_id": category_id,
                "category_name": category_name,
                "image_path": output_image_path,
                "mask_path": output_mask_path,
                "reference_image_path": output_ref_image_path
            })
            
            print(f"Saved image, mask, and reference image for index {i}")
            total_num += 1
            # 한 이미지에 대해 하나의 bbox만 처리 후 다음 이미지로
            break

    # bbox 정보가 있는 경우 JSON 파일로 저장
    if bbox_info:
        bbox_info_path = os.path.join(output_bbox_dir, f"bbox_info_{i}.json")
        with open(bbox_info_path, 'w') as f:
            json.dump(bbox_info, f, indent=4)

print(f"Saved Total Images: {total_num}")

# 테스트를 위해 처음 50개에 대해서만 실험
# if i >= 50:
#     break

# 데이터셋을 적절한 형식으로 저장하거나 후처리
# 예를 들어, pickle 파일로 저장하는 등의 작업을 수행할 수 있습니다.
