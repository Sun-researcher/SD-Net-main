# SD-Net-main
## Introduction
This is the official pytorch implementation of "Deepfake Detection via Multi-Scale Feature Pyramid Network with Spatial Attention"，which is submitted to The Visual Computer (TVC) 2025.
## Train Dataset

Download **FaceForensics++** from [here](https://github.com/ondyari/FaceForensics) and place it in the `/data` directory.
```
.
└── data
    └── FaceForensics++
        ├── original_sequences
        │   └── youtube
        │       └── raw
        │           └── videos
        │               └── *.mp4
        ├── train.json
        ├── val.json
        └── test.json
```
## Test Dataset
Other detailed information about the datasets is summarized below:


| Dataset                                | Real Videos | Fake Videos | Total Videos | Original Repository |
|----------------------------------------|-------------|-------------|--------------| --- |
| FaceForensics++                        | 1000        | 4000        | 5000         |  [Hyper-link](https://github.com/ondyari/FaceForensics/tree/master/dataset) |
| DeepfakeDetection                      | 363         | 3000        | 3363         |  [Hyper-link](https://github.com/ondyari/FaceForensics/tree/master/dataset) |
| Deepfake Detection Challenge (Preview) | 1131        | 4119        | 5250         | [Hyper-link](https://ai.facebook.com/datasets/dfdc/) |
| CelebDF-v1                             | 408         | 795         | 1203         | [Hyper-link](https://github.com/yuezunli/celeb-deepfakeforensics/tree/master/Celeb-DF-v1) |
| CelebDF-v2                             | 590         | 5639        | 6229         |  [Hyper-link](https://github.com/yuezunli/celeb-deepfakeforensics) |
| UADFV                                  | 49          | 49          | 98           | [Hyper-link](https://docs.google.com/forms/d/e/1FAIpQLScKPoOv15TIZ9Mn0nGScIVgKRM9tFWOmjh9eHKx57Yp-XcnxA/viewform) |
| FFIW                                   | 10000       | 1000        | 11000        |  [Hyper-link](https://github.com/tfzhou/FFIW) |
  

[//]: # (## Test Dataset)

[//]: # (Testing: [DFDCP]&#40;https://ai.meta.com/datasets/dfdc/&#41;, [CDF2]&#40;https://github.com/yuezunli/celeb-deepfakeforensics&#41;,)

[//]: # ([UADFV]&#40;https://github.com/SCLBD/DeepfakeBench&#41;, [DFD]&#40;https://github.com/ondyari/FaceForensics&#41;)
## Environment

Use the provided `data/Dockerfile` to set up the required environment.

## Data Preprocessing

Run the following commands to perform face cropping and data formatting:

```bash
python3 preprocess/crop_videos.py
```

## Training

Create the output directory first:
```bash
mkdir -p output/ssrt
# then
set CUDA_VISIBLE_DEVICES=*
python3 train.py
