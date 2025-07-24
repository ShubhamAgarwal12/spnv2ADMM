'''
Copyright (c) 2022 SLAB Group
Licensed under MIT License (see LICENSE.md)
Author: Tae Ha Park (tpark94@stanford.edu)
STEPS TO RUN:

Open terminal
Go to the project folder
execute "conda activate yolo-torch"
execute "python tools\\PoseNet.py --cfg experiments\offline_train_full_config_phi3_BN.yaml"
2025/07/01 09:47:18  * Time:  83.330 ms heat_eR 6.441 deg effi_eT 0.175 m final_pose 0.141
2025/07/01 10:07:12  * Time:  84.083 ms heat_eR 11.554 deg effi_eT 0.295 m final_pose 0.251
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import argparse
import torch
import torch.nn as nn
import _init_paths
import itertools

from config import cfg, update_config
from nets   import build_spnv2
from dataset import get_dataloader
from engine.inference  import do_valid
from utils.utils import set_seeds_cudnn, create_logger_directories, \
                        load_camera_intrinsics, load_tango_3d_keypoints

import random
from torch.utils.data import Subset, DataLoader


class PoseNet:

    def __init__(self, cfg):
        args = self.parse_args()
        update_config(cfg, args)

        self.device = torch.device('cuda:0') if cfg.CUDA and torch.cuda.is_available() else torch.device('cpu')
        self.logger, self.output_dir, _ = create_logger_directories(cfg, 'test')
        self.layer_weights = {}

        test_model = osp.join(cfg.OUTPUT_DIR, cfg.TEST.MODEL_FILE)
        if not osp.exists(test_model) or osp.isdir(test_model):
            test_model = osp.join(cfg.OUTPUT_DIR, cfg.MODEL.BACKBONE.NAME,
                                cfg.EXP_NAME, 'model_best.pth.tar')
        cfg.defrost()
        cfg.TEST.MODEL_FILE = test_model
        cfg.freeze()

        set_seeds_cudnn(cfg, seed=cfg.SEED)

        self.model = build_spnv2(cfg)

        if cfg.TEST.MODEL_FILE:
            self.model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE, map_location='cpu'), strict=True)
            self.logger.info('   - Model loaded from {}'.format(cfg.TEST.MODEL_FILE))
        self.model = self.model.to(self.device)

        self.source_data_loader = get_dataloader(cfg, split='train', load_labels=True)
        self.target_data_loader = get_dataloader(cfg, split='test', load_labels=True)
        self.subset_loader = None

        self.camera = load_camera_intrinsics(cfg.DATASET.CAMERA)
        self.keypts_true_3D = load_tango_3d_keypoints(cfg.DATASET.KEYPOINTS)
        self.cfg = cfg

    def parse_args(self):
        parser = argparse.ArgumentParser(description='Test on SPNv2')

        # general
        parser.add_argument('--cfg',
                            help='experiment configure file name',
                            required=True,
                            type=str)

        parser.add_argument('opts',
                            help="Modify config options using the command-line",
                            default=None,
                            nargs=argparse.REMAINDER)

        args = parser.parse_args()

        return args

    def prepare_grad_only_first(self):
        # Set gradients to False for all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Enable gradients for the first layer (first convolutional layer)
        first_layer = self.model.backbone.efficientnet[0][1]  # Accessing the first layer
        
        for param in first_layer.parameters():
            param.requires_grad = True

    def get_first_layer_weights(self):
        # Access the first layer of the efficientnet backbone
        first_layer = self.model.backbone.efficientnet[0][1]
        
        # Get the weights of the first layer (the weights are typically in the 'weight' attribute)
        first_layer_weights = first_layer.weight.data  # Use .data to get raw tensor
        first_layer_bias = first_layer.bias.data  # Use .data to get raw tensor

        return torch.stack([first_layer_weights, first_layer_bias])

    def get_first_layer_output(self, data_loader):
        
        outputs = []
        losses = []
        # Get the first layer
        layer0 = self.model.backbone.efficientnet[0][0]
        first_layer = self.model.backbone.efficientnet[0][1]

        dataset = data_loader.dataset

        # Randomly select sample indices from the dataset
        random_indices = random.sample(range(int(len(dataset)*0.20)), 16 * data_loader.batch_size)

        # Create a subset and new loader
        subset = Subset(dataset, random_indices)
        subset_loader = DataLoader(subset, batch_size=data_loader.batch_size, shuffle=False)

        self.subset_loader = subset_loader

        # Get batches
        selected_batches = list(subset_loader)

        for idx, (images, targets) in enumerate(selected_batches):#data_loader
            # Ensure input is on the correct device (GPU/CPU)
            images = images.to(self.device)

            # Directly pass the input through the first layer
            first_layer_output = first_layer(layer0(images))
            loss, loss_items = self.model(images, is_train=True, gpu=self.device, **{})

            outputs.append(first_layer_output)
            losses.append(loss)
            torch.cuda.empty_cache()

        return outputs, losses

    def get_output_rt(self):
        # optional: run do_valid
        R_reg, T_reg = None, None
        score, R_reg, T_reg = do_valid(0, self.cfg, self.model, self.subset_loader, self.camera,
                                        self.keypts_true_3D, valid_fraction=None,
                                        log_dir=self.output_dir, device=self.device)
        return R_reg, T_reg
    
    def save_model(self, name):
        torch.save(self.model, name)

def main(cfg):
    posenet = PoseNet(cfg)
    posenet.prepare_grad_only_first()
    R_reg, T_reg, _ = posenet.get_output_rt()
    print("weights: "+ str(posenet.get_first_layer_weights()))
    print("weights shape: "+ str(posenet.get_first_layer_weights().shape))
    #print("output: "+ str(posenet.get_first_layer_output(posenet.source_data_loader)))
    # input image torch.Size([3, 512, 768])
    # print("Feature Target:", len(ftgt)) #list with each element of dim ([1, 40, 256, 384], [1, 40, 256, 384] ....)
    # print("Feature Source:", fsrc.shape) #list with each element of dim ([1, 40, 256, 384], [1, 40, 256, 384] ....)
    # print("Conv1 Weights:", w1.shape)  # [40, 3, 3, 3]
    # print("R_reg:", R[0].shape) # list with each element of dim [(3,3), (3,3) .......]
    # print("T_reg:", T[0].shape) # list with each element of dim [(3,), (3,) ........]
    # posenet.update_model_weights(w1+0.09)

if __name__=='__main__':
    main(cfg)
