# CKD
This repository is the official PyTorch implementation of "Distilling Linguistic Context for Language Model Compression" by GeondoPark, Gyeongman Kim and Eunho Yang.
Currently, it only contains code for task-specific distillation. Experimental code for the task-agnostic and dynabert in the paper will be posted soon.

## Setup
Currently, install other dependencies with conda environments
```
conda create -n ckd python=3.7
conda activate ckd
pip install -r requirement.txt
```
## Prepare the datasets
### Glue dataset
Before running, download the GLUE data using this repository(https://github.com/nyu-mll/GLUE-baselines) and unpack it to directory $datas/glue.
In $datas/glue directory, check that a folder is created for each task, such as CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE, WNLI. (For simplicity, we rename the folder in CoLA to COLA)
### SQuAD dataset 
Before running, download the SQuADv1.1 data using this website(https://worksheets.codalab.org/worksheets/0xd53d03a48ef64b329c16b9baf0f99b0c/) and put it to directory $datas/squad.
In $datas/squad directory, check the file named train-v1.1.json and dev-v1.1.json. 

