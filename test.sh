python scripts/inference.py \
--plms --outdir results \
--config configs/v1.yaml \
--ckpt checkpoints/model.ckpt \
--image_path examples/image/example_1.png \
--mask_path examples/mask/example_1.png \
--reference_path examples/reference/example_1.jpg \
--seed 321 \
--scale 5

python scripts/inference.py \
--plms --outdir results \
--config configs/v1.yaml \
--ckpt checkpoints/model.ckpt \
--image_path examples/image/example_2.png \
--mask_path examples/mask/example_2.png \
--reference_path examples/reference/example_2.jpg \
--seed 5876 \
--scale 5

python scripts/inference.py \
--plms --outdir results \
--config configs/v1.yaml \
--ckpt checkpoints/model.ckpt \
--image_path examples/image/example_3.png \
--mask_path examples/mask/example_3.png \
--reference_path examples/reference/example_3.jpg \
--seed 5065 \
--scale 5


# for attention results, 새로운 inference file을 만듦
python inference.py --outdir results --config configs/v1.yaml --ckpt checkpoints/model.ckpt --image_path examples/image/example_1.png --mask_path examples/mask/example_1.png --reference_path examples/reference/example_1.jpg --seed 321 --scale 1

python inference.py --outdir results --config configs/v1.yaml --ckpt checkpoints/model.ckpt --image_path examples/image/example_2.png --mask_path examples/mask/example_2.png --reference_path examples/reference/example_2.jpg --seed 321 --scale 1

python inference.py --outdir results --config configs/v1.yaml --ckpt checkpoints/model.ckpt --image_path examples/image/example_3.png --mask_path examples/mask/example_3.png --reference_path examples/reference/example_3.jpg --seed 321 --scale 5

# for detection dataset, plms는 지워야된다
# 생성만 해보기

python inference.py \
--outdir results \
--config configs/v1.yaml \
--ckpt checkpoints/model.ckpt \
--image_path output_images/images/image_4.jpg \
--mask_path output_images/masks/mask_4.png \
--reference_path output_images/reference_images/reference_4.jpg \
--seed 321 \
--scale 5

# Detection dataset sample로 attribution map 시각화
python inference.py --outdir results --config configs/v1.yaml --ckpt checkpoints/model.ckpt --image_path output_images/images/image_4.jpg --mask_path output_images/masks/mask_4.png --reference_path output_images/reference_images/reference_4.jpg --seed 321 --scale 5

# 위의 코드로 생성된 attribution map으로 bbox 만들고 IOU계산
python scripts/cal_bbox_region.py 


# dir 단위로 attribution map 생성, inference_i2am.py
python inference.py --config configs/v1.yaml --ckpt checkpoints/model.ckpt --seed 321 --scale 5