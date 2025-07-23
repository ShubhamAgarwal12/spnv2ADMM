'''
Copyright (c) 2022 SLAB Group
Licensed under MIT License (see LICENSE.md)
Author: Tae Ha Park (tpark94@stanford.edu)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from scipy.io import savemat

import torch

from utils.utils       import AverageMeter, ProgressMeter
from utils.postprocess import solve_pose_from_heatmaps, rot_6d_to_matrix
from utils.metrics import *
from utils.visualize import *

logger = logging.getLogger("Testing")

def _get_metric_unit(name):
    unit = ''
    if 'eR' in name:
        unit = 'deg'
    elif 'eT' in name:
        unit = 'm'

    return unit

def do_valid(epoch, cfg, model, data_loader, camera, keypts_true_3D,
                valid_fraction=None, log_dir=None, device=torch.device('cpu')):

    batch_time = AverageMeter('', 'ms', ':6.1f')
    data_time  = AverageMeter('', 'ms', ':6.1f')

    metric_names = []
    if 'heatmap' in cfg.TEST.HEAD:
        metric_names += ['heat_eR', 'heat_eT', 'heat_eR_HIL', 'heat_eT_norm_HIL', 'heat_pose']
    if 'efficientpose' in cfg.TEST.HEAD:
        metric_names += ['effi_iou', 'effi_eR', 'effi_eT', 'effi_eR_HIL', 'effi_eT_norm_HIL', 'effi_pose']
    if 'segmentation' in cfg.TEST.HEAD:
        metric_names += ['segm_iou']

    metric_names += ['final_pose']

    metrics = {}
    per_sample_metrics = {}
    for name in metric_names:
        metrics[name] = AverageMeter(name, _get_metric_unit(name), ':6.2f')
        per_sample_metrics[name] = []

    # Predictions
    heatmaps = []
    h_reject = []
    q_hmap   = []
    t_hmap   = []
    R_reg    = []
    t_reg    = []
    bbox_reg = []

    # Check validation set fraction
    num_batches = len(data_loader)
    if valid_fraction:
        assert valid_fraction <= 1 and valid_fraction > 0, \
            'valid_fraction must be within (0, 1]'
        num_batches = int(len(data_loader) * valid_fraction)

    progress = ProgressMeter(
        num_batches,
        batch_time,
        [v for k, v in metrics.items() if k in ['heat_eR', 'effi_eT', 'final_pose']],
        prefix="Testing {:03d} ".format(epoch+1))

    # switch to eval mode
    model.eval()

    # Loop through dataloader
    end = time.time()
    for idx, (images, targets) in enumerate(data_loader):
        start = time.time()
        data_time.update((start - end)*1000)

        assert images.shape[0] == 1, 'Use batch size = 1 for testing'

        # Ground-truth
        domain    = targets['domain'][0]
        q_gt      = targets['quaternion'][0].numpy()
        R_gt      = targets['rotationmatrix'][0].numpy()
        t_gt      = targets['translation'][0].numpy()
        if 'efficientpose' in cfg.TEST.HEAD:
            bbox_gt = targets['boundingbox'][0].numpy()
        if 'segmentation' in cfg.TEST.HEAD:
            mask_gt = targets['mask'][0].numpy()

        with torch.no_grad():
            # Forward pass
            outputs, features = model(images,
                            is_train=False,
                            gpu=device)

            print(features[0].shape)
            # Debug (uncomment)
            # imshow(images[0])
            # imshowheatmap(images[0], heatmap[0], 8)

            # Post-process outputs for each task
            for i, name in enumerate(cfg.TEST.HEAD):
                iou, err_q, err_t       = None, None, None
                speed_q, speed_t, speed = None, None, None
                reject = False

                if name == 'heatmap':
                    keypts_pr, q_pr, t_pr, reject = solve_pose_from_heatmaps(
                        outputs[i].squeeze(0).cpu(),
                        cfg.DATASET.IMAGE_SIZE,
                        cfg.TEST.HEATMAP_THRESHOLD,
                        camera, keypts_true_3D
                    )

                    if not reject:
                        err_q = error_orientation(q_pr, q_gt, 'quaternion') # [deg]
                        err_t = error_translation(t_pr, t_gt)
                        speed_t, speed_q, speed = speed_score(t_pr, q_pr, t_gt, q_gt,
                            representation='quaternion',
                            applyThreshold=domain in ['lightbox', 'sunlamp'],
                            theta_q=cfg.TEST.SPEED_THRESHOLD_Q,
                            theta_t=cfg.TEST.SPEED_THRESHOLD_T)
                    else:
                        q_pr = t_pr = None

                    # Outputs
                    heatmaps.append(outputs[i].squeeze(0).cpu().numpy())
                    h_reject.append(reject)
                    q_hmap.append(q_pr)
                    t_hmap.append(t_pr)

                    # Metrics
                    result = {'_eR': err_q, '_eT': err_t, '_eR_HIL': speed_q, '_eT_norm_HIL': speed_t, '_pose': speed}

                elif name == 'segmentation':
                    mask_pr = outputs[i].sigmoid().cpu().numpy()
                    mask_pr[mask_pr > 0.5]  = 1
                    mask_pr[mask_pr <= 0.5] = 0

                    iou = segment_iou(mask_pr[0], mask_gt)

                    # Metrics
                    result = {'_iou': iou}

                elif name == 'efficientpose':
                    classification, bbox_prediction, \
                        rotation_raw, translation = outputs[i]
                    _, cls_argmax = torch.max(classification, dim=1)

                    # Bbox
                    bbox_pr = bbox_prediction[0,cls_argmax].squeeze().cpu().numpy()
                    R_pr    = rot_6d_to_matrix(rotation_raw[0,cls_argmax,:]).squeeze().cpu().numpy()
                    t_pr    = translation[0,cls_argmax].squeeze().cpu().numpy()

                    # IoU metric for bounding boxes
                    iou = np.squeeze(bbox_iou(bbox_pr, bbox_gt, x1y1x2y2=True))

                    # Pose metric(s)
                    err_q = error_orientation(R_pr, R_gt, 'rotationmatrix') # [deg]
                    err_t = error_translation(t_pr, t_gt)
                    speed_t, speed_q, speed = speed_score(t_pr, R_pr, t_gt, R_gt,
                        representation='rotationmatrix',
                        applyThreshold=domain in ['lightbox', 'sunlamp'],
                        theta_q=cfg.TEST.SPEED_THRESHOLD_Q,
                        theta_t=cfg.TEST.SPEED_THRESHOLD_T)

                    # Save & record
                    R_reg.append(R_pr)
                    t_reg.append(t_pr)

                    # Metrics
                    result = {'_iou': iou, '_eR': err_q, '_eT': err_t, '_eR_HIL': speed_q, '_eT_norm_HIL': speed_t, '_pose': speed}

                # Update metrics of this head
                if not reject:
                    for m, v in result.items():
                        metrics[name[:4]+m].update(v, 1)
                        per_sample_metrics[name[:4]+m].append(v)

            # Final pose score
            if not ('heatmap' in cfg.TEST.HEAD and h_reject[-1]):
                pose_r = per_sample_metrics['heat_eR_HIL'][-1] if 'heatmap' in cfg.TEST.HEAD else per_sample_metrics['effi_eR_HIL'][-1]
                pose_t = per_sample_metrics['effi_eT_norm_HIL'][-1] if 'efficientpose' in cfg.TEST.HEAD else per_sample_metrics['heat_eT_norm_HIL'][-1]
                metrics['final_pose'].update(pose_r + pose_t, 1)

        # Elapsed time
        batch_time.update((time.time() - start)*1000)
        end = time.time()

        if cfg.VERBOSE:
            progress.display(idx+1)
        if idx+1 == num_batches:
            progress.display_summary()

        # Break
        if valid_fraction and idx + 1 == num_batches:
            break

    # ----- Write results
    if log_dir is not None:
        if '/' in cfg.DATASET.DATANAME:
            partition = cfg.DATASET.DATANAME.split('/')[-1]
        else:
            partition = ''

        if cfg.DATASET.DATANAME == 'prisma25':
            domain = 'prisma25'

        log_dir = os.path.join(log_dir, partition, domain)
        os.makedirs(log_dir, exist_ok=True)

        # Aggregate different performances
        results_str = []
        results_mat = {}
        for name in cfg.TEST.HEAD:
            # Final results in string
            results_str += [f'Head: {name}\n']
            results_str += [
                f'{n[5:]} [{_get_metric_unit(n)}]: {m.avg:.5f}\n' for n, m in metrics.items() if n[:4] in name
            ]
            results_str += ['\n']

            # Final results in .mat
            for n, m in metrics.items():
                if n[:4] in name:
                    results_mat[n] = m.avg

            # Number of rejections for heatmap
            if name == 'heatmap':
                results_str += [f'# reject: {sum(h_reject):,d}\n\n']
                results_mat['h_reject'] = sum(h_reject)

        per_sample_metrics["h_reject"] = h_reject

        # Final pose
        results_str += ['Final Pose: {:.5f}\n'.format(metrics['final_pose'].avg)]
        results_mat['final_pose'] = metrics['final_pose'].avg

        # Write average performances
        resultfn = os.path.join(log_dir, 'mean_performance.txt')
        with open(resultfn, 'w') as f:
            [f.write(m) for m in results_str]
        logger.info(f'Mean performance (text) written to {resultfn}')

        # Save average performance
        resultfn = os.path.join(log_dir, 'mean_performance.mat')
        savemat(resultfn, results_mat, appendmat=False)
        logger.info(f'Mean performance (val) written to {resultfn}')

        # Save individual performances
        errorfn = os.path.join(log_dir, 'per_sample_performance.mat')
        savemat(errorfn, per_sample_metrics, appendmat=False)
        logger.info(f'Per-sample performances saved to {errorfn}')

        # Write predictions
        # predfn = os.path.join(log_dir, 'predictions_raw_shirt.mat')
        # savemat(predfn, {'hmap': heatmaps, 'hmap_q': q_hmap, 'hmap_t': t_hmap, 'eff_R': R_reg, 'eff_t': t_reg}, appendmat=False)

        #predfn = os.path.join(log_dir, 'predictions_pose.mat')
        #savemat(predfn, {'heat_q': q_hmap, 'heat_t': t_hmap, 'effi_R': R_reg, 'effi_t': t_reg, 'bbox': bbox_reg, 'reject': h_reject}, appendmat=False)
        #logger.info(f'Pose predictions saved to {predfn}')

    return metrics['final_pose'].avg, R_reg, t_reg



