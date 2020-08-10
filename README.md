## Prerequisites
- Python 3.6
- GPU Memory >= 6G
- Numpy
- Pytorch 0.4+

Preparation 1: create folder for dataset.

first, download Market-1501 dataset from the links below:

google drive: https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view?usp=sharing

baidu disk: https://pan.baidu.com/s/1ntIi2Op

second,
```bash
mkdir data
unzip Market-1501-v15.09.15.zip
ln -s Market-1501-v15.09.15 market
``` 
then, get the directory structure
``` 
├── data_aug_reid
        ├── data
                ├── market
                ├── Market-1501-v15.09.15
``` 


Preparation 2: Put the images with the same id in one folder. You may use 
```bash
python prepare.py
python changeIndex.py
```

Then, conduct ID data augmentation
```bash
python self_supervise_augment.py
```

Finally, conduct training, testing and evaluating with one command
```bash
python run.py
```

This code is related to our paper _Self-supervised Data Augmentation for Person Re-Identification_.

If you use this code, please cite our paper as:
```
@article{CHEN202048,
title = "Self-supervised data augmentation for person re-identification",
journal = "Neurocomputing",
volume = "415",
pages = "48 - 59",
year = "2020",
issn = "0925-2312",
doi = "https://doi.org/10.1016/j.neucom.2020.07.087",
url = "http://www.sciencedirect.com/science/article/pii/S0925231220312170",
author = "Feng Chen and Nian Wang and Jun Tang and Dong Liang and Hao Feng",
}
```






## reference link:
https://github.com/layumi/Person_reID_baseline_pytorch

