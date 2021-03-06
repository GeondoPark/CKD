# CKD - Distilling Context Linguistic for Language model compression
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![DOI](https://zenodo.org/badge/434422268.svg)](https://zenodo.org/badge/latestdoi/434422268)  
This repository is the official PyTorch implementation of ["Distilling Linguistic Context for Language Model Compression"](https://arxiv.org/pdf/2109.08359.pdf) by GeondoPark, Gyeongman Kim and Eunho Yang.  
Currently, it only contains code for task-specific distillation.

## Setup
Currently, install other dependencies with conda environments
```
conda create -n ckd python=3.7
conda activate ckd
pip install -r requirement.txt
```
## Prepare the datasets
### GLUE dataset
Before running, download the GLUE data using [this repository](https://github.com/nyu-mll/GLUE-baselines) and unpack it to directory $datas/glue.  
In $datas/glue directory, check that a folder is created for each task - CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE, WNLI. (We rename the folder in CoLA to COLA)
### SQuADv1.1 dataset 
Before running, download the SQuADv1.1 data using [this website](https://worksheets.codalab.org/worksheets/0xd53d03a48ef64b329c16b9baf0f99b0c/) and put it to directory $datas/squad.  
In $datas/squad directory, check the file named train-v1.1.json and dev-v1.1.json. 

## Prepare the pre-trained smaller BERT provided by [Turc etal,.2019](https://arxiv.org/abs/1908.08962)
Before task-specific distillation, you download the pre-trained teacher and student model from [this repository](https://github.com/google-research/bert).  
For example, download the bert-base-uncased model to be used as a teacher and the bert 6/768 model to be used as a student and unpack it to directory $download_models.

## Training the teacher model
Overall hyperparameters for training are set by default.  Hyperparameters for each task including the number of epochs and learning rate for each tasks, please refer the [our paper](https://arxiv.org/abs/2109.08359) and original [BERT](https://arxiv.org/abs/1810.04805).
### GLUE Training

```
python main_glue.py --exp_name teachers_glue \    
                    --do_train \  
                    --model_type bert \  
                    --model_path ./download_models/bert-base-uncased \  
                    --do_lower_case \  
                    --task_name [TASK_NAME] \  
                    --data_dir ./datas/glue \  
                    --per_gpu_batch_size 32 \  
                    --num_train_epochs [#EPOCHS] \  
                    --learning_rate [LR] \  
```

## Training the student model with task-specific distillation
Perform task specific distillation with **CKD**. 
### GLUE distillation training
We included specific hyperparameters of CKD for each task in the BERT 6/768 student model setting.  
For other smaller BERTs or hyperparameter tuning, comment out 'args = change_args(args)' in main_glue_distill.py  

```
python main_glue_distill.py --exp_name distill_student \  
                    --distil_loss kl+wrdist+wrangle_window+ltrdist+ltrangle \  
                    --do_lower_case \   
                    --do_train \  
                    --temperature [TEMPERATURE] \  
                    --task_name [TASK_NAME] \  
                    --matching_strategy_wr emb+skip \  
                    --matching_strategy_ltr emb+skip \  
                    --wrdist [COEFFICIENT FOR WRDIST] \  
                    --wrangle [COEFFICIENT FOR WRANGLE] \  
                    --ltrdist_w [COEFFICIENT FOR LTRDIST] \  
                    --ltrangle_w [COEFFICIENT FOR LTRANGLE] \  
                    --num_train_epoch [#EPOCHS] \  
                    --learning_rate [#LR] \
                    --window_size 21 \  
                    --distance cos \  
                    --lossfunc_d kldiv \  
                    --lossfunc_a l2loss \  
                    --teacher_type bert \  
                    --teacher_path ./teachers_glue \  
                    --student_type bert_small \  
                    --student_path ./download_models/bert-6-768 \  
                    --logging_step 500 \  
                    --eval_steps 500 \  
                    --seed [SEED]
 
```
## Citation
```
@inproceedings{park2021distilling,
  title={Distilling Linguistic Context for Language Model Compression},
  author={Park, Geondo and Kim, Gyeongman and Yang, Eunho},
  booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  pages={364--378},
  year={2021}
}
```
## Acknowledgement
```
This work was supported by the National Research Foundation of Korea (NRF) grants (2018R1A5A1059921, 2019R1C1C1009192) and Institute of Information & Communications Technology Planning & Evaluation (IITP) grants(No.2017-0-01779, A machine learning and statistical inference framework for explainable artificial intelligence, No.2019-0-01371, Development
of brain-inspired AI with human-like intelligence, No.2019-0-00075, Artificial Intelligence Graduate School Program (KAIST)) funded by the Korea government (MSIT).
```
