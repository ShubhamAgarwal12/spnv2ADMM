# SPNv2AMM: This is an extension of spacecraft pose estimation project (SPNv2) where we use LADMM to perform domian adaptation

## This project was done as part of Model Based Deep Learning course at Ben Gurion University, Israel
## Student Members: Shubham Agarwal (agarwals@post.bgu.ac.il) and Caitong Peng (caitong@post.bgu.ac.il)

## Setup Instructions
1. Clone this repository to your machine.
2. Follow the Readme.txt from SPNv2[https://github.com/tpark94/spnv2]  project to set up the environment.
3. Download the SPEED+ dataset and the pre-trained model as mentioned in the SPNv2 repo.
4. Update the corresponding paths in the config file[https://github.com/ShubhamAgarwal12/spnv2ADMM/edit/main/README.md#:~:text=offline_train_full_config_phi3_BN]

## Fine-tuning for real data
Make sure you have dataset and the pre-trained model from SPNv2 at correct path. 
Execute the following command to fine-tune the batch normalization layer.

'python tools\\ADMM.py --cfg experiments\offline_train_full_config_phi3_BN.yaml'

This will output 

## Inference
After the fine-tune is complete, execute the following command to test the fine-tuned model:

'python tools\\test_admm.py --cfg experiments\offline_train_full_config_phi3_BN.yaml'

## Results
