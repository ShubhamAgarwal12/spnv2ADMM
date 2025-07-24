SPNv2ADMM
=========

This is an extension of the spacecraft pose estimation project (SPNv2),
where we use LADMM to perform domain adaptation.

Project Info
------------
This project was done as part of the Model-Based Deep Learning course
at Ben Gurion University, Israel.

Student Members:
- Shubham Agarwal (agarwals@post.bgu.ac.il)
- Caitong Peng (caitong@post.bgu.ac.il)

In this project, we used the LADMM method for domain adaptation of a pre-trained satellite pose estimation model. 
The model was trained on synthetic data (from the SPNv2 repo) and then fine-tuned using real-world Lightbox data.

SPNv2 also supports fine-tuning on the Lightbox domain using online domain adaptation without labels. 
We use these two setups (SPNv2 synthetic baseline model and SPNv2 fine-tuned model) as baselines for our LADMM-based method.

Our LADMM implementation can be found in:
tools/ADMM.py — Lines 164 to 231.

Note:
We fine-tune only the first BatchNorm layer using LADMM. 
Other configurations (multiple BN layers) were tested but did not yield better results.

Setup Instructions
------------------
1. Clone this repository:
   git clone https://github.com/ShubhamAgarwal12/spnv2ADMM.git
   cd spnv2ADMM

2. Follow the setup guide from the original SPNv2 project:
   https://github.com/tpark94/spnv2

3. Download the SPEED+ dataset and the pretrained model as described in the SPNv2 repo.

4. Update the dataset and model paths in:
   experiments/offline_train_full_config_phi3_BN.yaml

Fine-tuning on Real Data
------------------------
Ensure the dataset and pretrained model are correctly set up. Then run:

   python tools/ADMM.py --cfg experiments/offline_train_full_config_phi3_BN.yaml

After training, the LADMM fine-tuned model will be saved as:
   ladmm_model_best.pth.tar

Fine-tuned Model
----------------
You can download the LADMM fine-tuned model here:
https://drive.google.com/file/d/15M59NriVGpGxkUnDIuSK93tE1oOe87xh/view?usp=sharing

Testing
-------
To evaluate the fine-tuned model:

   python tools/test_admm.py --cfg experiments/offline_train_full_config_phi3_BN.yaml

Results
-------
Lightbox Evaluation:

| Experiment                  | Rotation Error (°) | Translation Error (m) | SPEED Score |
|----------------------------|--------------------|------------------------|-------------|
| Baseline (SPNv2 pretrained)| 6.442              | 0.175                  | 0.141       |
| LADMM Fine-tuned (BN only) | 6.201              | 0.166                  | 0.136       |
| Full Fine-tune (SPNv2)     | 5.624              | 0.142                  | 0.122       |

