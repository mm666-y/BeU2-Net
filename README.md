# BeU2-Net
The official implementation for the JBHI-2025 paper "Boundary-Enhanced U2-Net for Simultaneous Four-Chamber Segmentation in Transthoracic Echocardiography".

## Datasets
### Public datasets
* CAMUS：https://www.creatis.insa-lyon.fr/Challenge/camus/
* EchoNet-Dynamic：https://echonet.github.io/dynamic/
### Private datasets
* Private A4C：contact us if you want to use this dataset mengyuanqin@stu.kust.edu.cn.
## Getting Started
### Install
first, download the project's code:
```bash
#clone the project
git clone 
cd project-name
conda create -n myenv python=3.8 -y
conda activate myenv
```
then, install all requirements:
```bash
pip install -r requirements.txt
```
prepare the dataset:
modify the data_path in config_setting.py.
The dataset for private dataset is as following organized:
```
├── [Your Data Path]
    ├── edges
    ├── images
    ├── masks
    └── label.csv

```
run train.py.

## Citation
if you find this project useful, please consider citing:

```bibtex
@article{meng2025boundary,
  title={Boundary-Enhanced $ U\^{}$\{$2$\}$ $-Net for Simultaneous Four-Chamber Segmentation in Transthoracic Echocardiography},
  author={Meng, Yuanqin and Chai, Shengjie and Xiao, Haoyu and Meng, Zhaohui and Wang, Qingwang and Shen, Tao},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2025},
  publisher={IEEE}
}
```