# Intelligent Segmentation of Wildfire Region and Interpretation of Fire Front in Visible Light Images from the Viewpoint of an Unmanned Aerial Vehicle (UAV)

A new method is proposed for the intelligent segmentation and fire front interpretation of wildfire regions.

## Abstract

The acceleration of global warming and intensifying global climate anomalies have led to a rise in the frequency of wildfires. However, most existing research on wildfire fields focuses primarily on wildfire identification and prediction, with limited attention given to the intelligent interpretation of detailed information, such as fire front within fire region. To address this gap, advance the analysis of fire fronts in UAV-captured visible images, and facilitate future calculations of fire behavior parameters, a new method is proposed for the intelligent segmentation and fire front interpretation of wildfire regions. This proposed method comprises three key steps: deep learning-based fire segmentation, boundary tracking of wildfire regions, and fire front interpretation. Specifically, the YOLOv7-tiny model is enhanced with a Convolutional Block Attention Module (CBAM), which integrates channel and spatial attention mechanisms to improve the model's focus on wildfire regions and boost the segmentation precision. Experimental results show that the proposed method improved detection and segmentation precision by 3.8% and 3.6%, respectively, compared to existing approaches, and achieved an average segmentation frame rate of 64.72 Hz, which is well above the 30 Hz threshold required for real-time fire segmentation. Furthermore, the method’s effectiveness in boundary tracking and fire front interpreting was validated using an outdoor grassland fire fusion experiment’s real fire image data. Additional tests were conducted in southern New South Wales, Australia, using data that confirmed the robustness of the method in accurately interpreting the fire front. The findings of this research have potential applications in dynamic data-driven forest fire spread modeling and fire digital twinning areas. 

## Usage

### 1. Dataset

Get dataset: Download the dataset into the data folder. The dataset can be downloaded at:https://github.com/suyixuan123s/Fire-Segmentation-Dataset.git. There you can change for your own dataset.

Scale the dataset: use the following code to scale the dataset, the exact scale of the dataset can be reset in the code.

```
python3 splitdataset.py 
```

### 2. Requirement

Deployment environment: The detailed configuration is described in the paper, and dependencies can be installed using the following code (We recommend using Anaconda to deploy the environment).

```
pip3 install -r requirements.txt
```

### 3. Train

Preparation: adjusting parameters of configuration file hyp.scratch-high.yaml, training file train.py

Start training: Use the following code to start training, the corresponding parameters can be adjusted by yourself:

```
python3 segment/train.py --data data/custom.yaml --batch 100 --weights data/yolov7-seg.pt --cfg data/yolov7-seg.yaml --epochs 200 --name yolov7-seg --img 640 --hyp data/hyp.scratch-high.yaml
```

### 4. Val

The val file in the folder can be used to evaluate the model for each of the evaluation metrics.

```
python3 segment/val.py 
```

### 5. Predict

Predictions are made using the predict file, in which the target object mask map can be obtained by adjusting the code in the Mask drawing section of the folder.

```
python3 segment/predict.py 
```

### 6. Interpretation

Using the segmentation mask map as input, the fire boundary and fire front are obtained using the files in the folder interpretation. The specific fire front interpretation process can be found in the paper.

```
python3 segment/extract.py 
python3 segment/reconstruction.py 
python3 segment/superposition.py 
```

## Acknowledgment

We acknowledge the contributions of the following open-source projects and their authors：

- ```
  https://github.com/WongKinYiu/yolov7/tree/u7/seg
  https://github.com/suyixuan123s/Fire-Segmentation-Dataset.git
  ```

## Citation
