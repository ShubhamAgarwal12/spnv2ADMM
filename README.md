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

Setup Instructions
------------------
1. Clone this repository to your machine:
   git clone https://github.com/ShubhamAgarwal12/spnv2ADMM.git
   cd spnv2ADMM

2. Follow the setup instructions from the original SPNv2 project:
   https://github.com/tpark94/spnv2

3. Download the SPEED+ dataset and the pre-trained model, as mentioned in the SPNv2 repo.

4. Update the corresponding dataset and model paths in the config file:
   experiments/offline_train_full_config_phi3_BN.yaml

Fine-tuning on Real Data
------------------------
Make sure you have the dataset and the pre-trained model from SPNv2 at the correct paths.
Then execute the following command to fine-tune the batch normalization layers:

   ```python tools\ADMM.py --cfg experiments\offline_train_full_config_phi3_BN.yaml```

After training, the fine-tuned model will be saved as:
   ladmm_model_best.pth.tar

Testing
---------
To test the fine-tuned model, run:

   ```python tools\test_admm.py --cfg experiments\offline_train_full_config_phi3_BN.yaml```

Results
-------
Lightbox

| Experiment             | Rotation Error (in degrees)  | Translation Error (in meters) | SPEED  |
|------------------------|----------------|--------------|----------|
| Baseline               | 6.442           |     0.175     |    0.141      |
| LADMM Fine-tuned (BN layer)|   6.201       | 0.166     | 0.136 |
| Neural Network Fine-tuned | 5.624        | 0.142     | 0.122
