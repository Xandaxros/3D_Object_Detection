import pickle
import time

import numpy as np
import torch
import torch.cuda
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils

import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import os

def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])

def get_image_names(img_dir):
    """Returns names of all .png files"""
    index_file = '/fs/ess/scratch/PLS0151/obj3d/detection3d/ImageSets/val.txt'
    imgs = np.loadtxt(index_file, dtype='str')
    return imgs

def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
        
    gpu_time = 0 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True) 
    start_time = time.time()
    
    start_cpu = time.process_time()
    with torch.no_grad(): 
        with torch.cuda.amp.autocast(): 
            starter.record() #gpu 
            #spatial = e_model(torch.cat([data.cell_data, data.x], axis=-1)) #.to(device)
    
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                pred_dicts, ret_dict = model(batch_dict)
        disp_dict = {}

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')
        
    
    end_time = time.time() 
    end_cpu = time.process_time() 
    #running_time.extend(((end_time - start_time),(end_cpu - start_cpu),gpu_time))
    ender.record() 
    torch.cuda.synchronize() 
    gpu_time = starter.elapsed_time(ender)/1000.0

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (end_time - start_time) / len(dataloader.dataset)
    fps = len(dataloader.dataset) / (end_time - start_time)
    logger.info('Time Taken to Evaluate: %.4f seconds' % (end_time - start_time))
    logger.info('CPU Time: %.4f seconds' % (end_cpu - start_cpu))
    logger.info('GPU Time: %.4f' % gpu_time)
    logger.info('Number of Frames processed: %.4f' % len(dataloader.dataset))
    logger.info('Frames Per Second(fps): %.4f' % fps)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)
    
    # logger.info(pred_dicts)
    # (N, 7) [x, y, z, l=Length, h=Height, w=Width, r]
    # [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    
    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))
    
    for i, anno in enumerate(det_annos):
        print(anno)
        break

    ## START
    img_dir = '/fs/ess/scratch/PLS0151/obj3d/kitti_raw/training/image_2'
    image_names = get_image_names(img_dir)
    
    for i, anno in enumerate(det_annos):
        # anno_items = anno.items()
        # anno_list = list(anno_items)
        anno = pd.DataFrame(anno, columns = ['name','truncated','occluded','alpha','bbox','dimensions','location','rotation_y'])
        f = image_names[i]
        image_file = img_dir + f
        img = cv2.imread(image_file)
        coordinates = anno[[col for col in anno.columns if col.startswith('bbox')]]
        corners = coordinates.apply(lambda x: ((x.bbox[1], x.bbox[3]), (x.bbox[1], x.bbox[2]), \
                            (x.bbox[0], x.bbox[3]), (x.bbox[0], x.bbox[2])), axis=1)
        for box in corners:
            for corner in box:
                cv2.circle(img, corner, 1, (255, 0, 0), 5)
                # cv.Circle(img, center, radius, color, thickness=1, lineType=8, shift=0)
                
        anno[['name'] + [col for col in anno.columns if col.startswith('bbox')]]
        boxes = anno.apply(lambda x: ((int(x.bbox_xmin), int(x.bbox_ymin)), (int(x.bbox_xmax), int(x.bbox_ymax))), axis=1)
        for i, box in enumerate(boxes):
            img = cv2.rectangle(img, box[0], box[1], (0, 255, 0))
            img = cv2.putText(img, str(i), (box[0][0] + 10, box[0][1] - 4) , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            
        img = Image.fromarray(img, 'RGB')
        img.save(f"/fs/ess/scratch/PLS0151/Output_Images/{x+1}.png")
        
    ## END
 
    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass
