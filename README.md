FADMFormer: Frequency-Aware and Deep-Guided Multi-Semantic Transformer for Infrared Small Target Detection
By Xiaoxi Liao, Kai Wang*, Wei Jiang, Hongke Zhang
<img width="1398" height="682" alt="image" src="https://github.com/user-attachments/assets/8b639a1e-fa55-4dbd-ab08-e7eb166e0918" />
<img width="1419" height="439" alt="image" src="https://github.com/user-attachments/assets/c83ab62d-b606-4fe2-9400-0981f6d31ab2" />
## Requirements

* Python 3.8
* torch 2.1.2+cu118
* torchvision 0.16.2+cu118
* opencv-python 4.13.0.92
* kornia 0.6.3
* numpy 1.26.4
* Pillow 10.3.0
* tqdm 4.64.1
* matplotlib 3.8.2
* imageio 2.34.0
* einops 0.7.0
* thop 0.1.1.post2209072238
* PyYAML 6.0.2
* colorama 0.4.6
* tensorboard 2.14.0
* albumentations 1.4.18
* scikit-image 0.21.0
* ptflops 0.7.3
* fvcore 0.1.5.post20221221

You can install the main dependencies with:

```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install opencv-python==4.13.0.92 kornia==0.6.3 numpy==1.26.4 pillow==10.3.0 tqdm==4.64.1 matplotlib==3.8.2 imageio==2.34.0 einops==0.7.0 thop==0.1.1.post2209072238
pip install pyyaml==6.0.2 colorama==0.4.6 tensorboard==2.14.0 albumentations==1.4.18 scikit-image==0.21.0 ptflops==0.7.3 fvcore==0.1.5.post20221221
## To test
```bash
python test.py
```
## To train
```bash
python train.py
```
### Dataset Download

The three single-frame infrared small target detection datasets used in this work are publicly available. Their download links can be found below.

* **NUDT-SIRST**

    NUDT-SIRST is an infrared small target detection dataset with diverse clutter backgrounds, target shapes, and target sizes.

    * [NUDT-SIRST GitHub](https://github.com/YeRen123455/Infrared-Small-Target-Detection)

* **IRSTD-1k**

    IRSTD-1k is a realistic infrared small target detection dataset containing 1,000 manually annotated infrared images with various target shapes, sizes, and cluttered backgrounds.

    * [ISNet GitHub](https://github.com/RuiZhang97/ISNet)
    * [CVF Dataset Page](https://cove.thecvf.com/datasets/716)

* **NUAA-SIRST**

    NUAA-SIRST is a single-frame infrared small target detection dataset collected from infrared image sequences, with pixel-level annotations for small target segmentation.

    * [SIRST GitHub](https://github.com/YimianDai/sirst)
Contact 2481177956@qq.com if you have other questions.
