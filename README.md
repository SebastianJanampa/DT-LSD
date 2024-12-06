[![arXiv](https://img.shields.io/badge/arXiv-2108.03144-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2411.13005)

# DT-LSD: Deformable Transformer-based Line Segment Detection
 
This repository contains the official implementation for **DT-LSD** (**D**eformable **T**ransformer-based **L**ine **S**egment **D**etector)

<img src="figures/model.png" alt="Model Architecture"/>

## TODO
- [x] Training
- [x] Evaluation
- [x] Upload source code
- [x] Upload weight 
- [x] Inference
- [x] Upload arxiv paper 


## Results 
<div align="center">
<p float="left">
<img src="figures/sAP_curve_wireframe.png" alt="sap_wireframe" style="width:30%; height:auto;"/>
<img src="figures/aph_curve_wireframe.png" alt="aph_wireframe" style="width:30%; height:auto;"/>
</p>
</div>
<div align="center">
<p float="left">
<img src="figures/sAP_curve_york.png" alt="sap_york" style="width:30%; height:auto;"/>
<img src="figures/aph_curve_york.png" alt="aph_york" style="width:30%; height:auto;"/>
</p>
</div>

|Dataset| sAP10 | sAP15 | APH | FH | FPS
| --- | --- | --- | --- | --- | ---|
|Wireframe| 71.7 | 73.9 | 89.1 | 85.8 | 8.9 |
|YorkUrban| 33.2 | 35.1 | 65.9 | 68.0 | 8.9 |


## Installation


1. Clone this repository.
   ```sh
   git clone https://github.com/SebastianJanampa/DTLSD.git
   cd DTLSD
   ```

2. Install Pytorch and torchvision

   Follow the instructions on https://pytorch.org/get-started/locally/.
   ```sh
   # an example:
   conda install -c pytorch pytorch torchvision
   ```

3. Install other needed packages
   ```sh
   pip install -r requirements.txt
   ```
      
4. Compiling CUDA operators
   ```sh
   cd models/dtlsd/ops
   python setup.py build install
   # unit test (should see all checking is True)
   python test.py
   cd ../../..
   ```

## Dataset

To reproduce our results, you need to process two datasets, [ShanghaiTech](https://github.com/huangkuns/wireframe) and [YorkUrban](https://www.elderlab.yorku.ca/resources/york-urban-line-segment-database-information/). 
```
mkdir data
cd data
wget https://github.com/SebastianJanampa/storage/releases/download/v1.0.0/wireframe_processed.zip
wget https://github.com/SebastianJanampa/storage/releases/download/v1.0.0/york_processed.zip

unzip wireframe_processed.zip
unzip york_processed.zip

rm *zip
cd ..
```
    
## Pretraining weights
Download the weights from DINO_SWIN_4scales_36_epochs from the [DINO repo](https://github.com/IDEA-Research/DINO/tree/main), and place it in the pretrain folder.
 
## Run

1. Training
```sh
bash scripts/train/DTLSD_SWIN_4_scales_24_epochs.sh 
```

2. Testing
```sh
bash scripts/train/DTLSD_SWIN_4_scales_24_epochs.sh
```

## Demo
Download the DTLSD weights 

```
wget https://github.com/SebastianJanampa/storage/releases/download/v1.0.0/DTLSD_checkpoint0035.pth
```
If wget doesn't work, download it from this [link](https://github.com/SebastianJanampa/storage/releases/download/v1.0.0/DTLSD_checkpoint0035.pth), and place it in the main folder. 
Then run
```
python demo.py
```

## Citation 
```
@article{janampa2024dt,
  title={DT-LSD: Deformable Transformer-based Line Segment Detection},
  author={Janampa, Sebastian and Pattichis, Marios},
  journal={arXiv preprint arXiv:2411.13005},
  year={2024}
}
```
