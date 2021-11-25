# Non-Local Latent Relation Distillation for Self-Adaptive 3D Human Pose Estimation

This repository is the official implementation of Non-Local Latent Relation Distillation for Self-Adaptive 3D Human Pose Estimation

## Requirements

To install the requirements, please create a virtual environment and install the required packages. This codebase was created for and tested on Python 3.8:

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run the corresponding commands from the root directory:

### Pose autoencoder

```bash
cd pose_autoencoder
python train.py
```
### Motion autoencoder

```bash
cd motion_autoencoder
python train.py
```
### Relation Transformer Networks

1. To train Motion Relation Transformer Networks:

```bash
cd relation_transformer
python train_motion_rule.py
```
2. To train Pose Relation Transformer Network:

```bash
cd relation_transformer
python train_pose_rule.py
```

### Source _image-to-latent_ model

```bash
cd image_to_latent_encoder
python train.py 
```

### Target _image-to-latent_ model. (Self-adaptation)

```bash
cd target_adaptation
python train.py
```

## Evaluation

To evaluate the trained target model on the H3.6M test set, run the following commands:

```bash
cd evaluation
python eval.py 

```
<!---
## Pre-trained Models

The pretrained models can be found inside the `pretrained_weights` directory.

## Results

The detailed quantitative results can be found in the main paper and supplementary material.
'''
-->

