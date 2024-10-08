# Paint by Example model + I2AM attribution map visualization 


## Requirements
A suitable [conda](https://conda.io/) environment named `Paint-by-Example` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate Paint-by-Example
```

## Pretrained Model
We provide the checkpoint ([Google Drive](https://drive.google.com/file/d/15QzaTWsvZonJcXsNv-ilMRCYaQLhzR_i/view?usp=share_link) | [Hugging Face](https://huggingface.co/Fantasy-Studio/Paint-by-Example/resolve/main/model.ckpt)) that is trained on [Open-Images](https://storage.googleapis.com/openimages/web/index.html) for 40 epochs. By default, we assume that the pretrained model is downloaded and saved to the directory `checkpoints`.

## Basic Testing

To sample from our model, you can use `scripts/inference.py`. For example, 
```
python scripts/inference.py \
--plms --outdir results \
--config configs/v1.yaml \
--ckpt checkpoints/model.ckpt \
--image_path examples/image/example_1.png \
--mask_path examples/mask/example_1.png \
--reference_path examples/reference/example_1.jpg \
--seed 321 \
--scale 5
```
or simply run:
```
sh test.sh
```
Visualization of inputs and output:

![](figure/result_1.png)
![](figure/result_2.png)
![](figure/result_3.png)

## Basic Training

### Data preparing
- Download separate packed files of Open-Images dataset from [CVDF's site](https://github.com/cvdfoundation/open-images-dataset#download-images-with-bounding-boxes-annotations) and unzip them to the directory `dataset/open-images/images`.
- Download bbox annotations of Open-Images dataset from [Open-Images official site](https://storage.googleapis.com/openimages/web/download_v7.html#download-manually) and save them to the directory `dataset/open-images/annotations`.
- Generate bbox annotations of each image in txt format.
    ```
    python scripts/read_bbox.py
    ```

The data structure is like this:
```
dataset
├── open-images
│  ├── annotations
│  │  ├── class-descriptions-boxable.csv
│  │  ├── oidv6-train-annotations-bbox.csv
│  │  ├── test-annotations-bbox.csv
│  │  ├── validation-annotations-bbox.csv
│  ├── images
│  │  ├── train_0
│  │  │  ├── xxx.jpg
│  │  │  ├── ...
│  │  ├── train_1
│  │  ├── ...
│  │  ├── validation
│  │  ├── test
│  ├── bbox
│  │  ├── train_0
│  │  │  ├── xxx.txt
│  │  │  ├── ...
│  │  ├── train_1
│  │  ├── ...
│  │  ├── validation
│  │  ├── test
```

### Download the pretrained model of Stable Diffusion
We utilize the pretrained Stable Diffusion v1-4 as initialization, please download the pretrained models from [Hugging Face](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original) and save the model to directory `pretrained_models`. Then run the following script to add zero-initialized weights for 5 additional input channels of the UNet (4 for the encoded masked-image and 1 for the mask itself).
```
python scripts/modify_checkpoints.py
```

### Training Paint by Example
To train a new model on Open-Images, you can use `main.py`. For example,
```
python -u main.py \
--logdir models/Paint-by-Example \
--pretrained_model pretrained_models/sd-v1-4-modified-9channel.ckpt \
--base configs/v1.yaml \
--scale_lr False
```
or simply run:
```
sh train.sh
```

## For + I2AM

Assume that there is a COCO val2017 dataset.
First, you have to prepare the custom detection dataset using make_dataset.py 

```
python make_dataset.py
```

### Additional Testing 1: Inference for one sample

Assume that there are sample images (image, mask, ref) in output_images directory. Automatically, generates attribution maps in results dir using ./scripts/hook.py. We don't use plms sampler but use DDIM.

```
python inference.py \
--outdir results \
--config configs/v1.yaml \
--ckpt checkpoints/model.ckpt \
--image_path output_images/images/image_4.jpg \
--mask_path output_images/masks/mask_4.png \
--reference_path output_images/reference_images/reference_4.jpg \
--seed 321 \
--scale 5
```

### Additional Testing 2: Inference for dir

Assume that the dataset for detection exists in output_images (bbox_info, images, masks, reference_images). Generates attribution maps in i2am dir using inference_i2am.py

```
python inference_i2am.py --config configs/v1.yaml --ckpt checkpoints/model.ckpt --seed 321 --scale 5
```

### Additional Testing 3: Generates bbox

For detection task, PBE + I2AM generates attribution maps and then creates bounding box in ./i2am/overlay via cv2. Finally, calculates IOU.

- Baseline model들은 기존 dataset의 bbox info를 그대로 사용해도 되지만, PBE의 경우 image를 $512 \times 512$로 resize해야 되기 때문에 bbox도 image size에 맞춰서 조정해야 한다. 따라서 PBE는 bbox_info의 adjusted_bbox dictionary를 baseline model은 bbox dictionary를 사용한다.

- 또한, attribution map은 colormap=jet으로 저장했기 때문에 gray scale 이미지로 바로 load하는 것이 아니라 jet color map을 이용해서 색상 값을 원래 값으로 변환해야 한다. (cal_bbox_region_rect.py에 구현되어 있음)

```
python scripts/cal_bbox_region.py 
```

Ovelay GT bounding box on GT image

```
python scripts/cal_gt_bbox.py
```

Calculate oveall image bounding box vs GT bounding box (worst and basic baseline)

```
python call_bbox_as_overall_img.py
```

### Detection Framework using I2AM


### Baseline Model 

1. Overall Image: 전체 이미지를 bounding box로 보고 계산 

```
python ./scripts/cal_bbox_as_overall_img.py
```

2. Random bbox: 이미지의 30~50%를 차지하는 bbox를 생성해서 IOU 계산

```
python ./scripts/cal_random_bbox.py
```

3. GT bbox: ground truth bbox를 그리는 script

```
python ./scripts/cal_gt_bbox.py
```

4. I2AM bbox: I2AM을 이용한 bounding box IOU 계산

```
python ./scripts/cal_bbox_region_rect.py
```
