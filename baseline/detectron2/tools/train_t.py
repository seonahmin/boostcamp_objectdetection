#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import logging
from nis import match
import os
import copy
from collections import OrderedDict
from detectron2.modeling.test_time_augmentation import DatasetMapperTTA

import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.utils.visualizer import Visualizer
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.data.datasets import register_coco_instances
import detectron2.data.transforms as T
import albumentations as A
from detectron2.data import detection_utils as utils
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.evaluation import (
    COCOEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model, test_time_augmentation
from detectron2.utils.events import EventStorage
from detectron2.solver import build_lr_scheduler, get_default_optimizer_params
from detectron2 import model_zoo
import wandb
import re
import glob
import yaml
from pathlib import Path

logger = logging.getLogger("detectron2")


def get_evaluator(cfg, dataset_name, output_folder=None):
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    return COCOEvaluator(dataset_name, output_dir=output_folder)

def Mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict['file_name'], format='BGR')
    
    

    transform_list = [
        T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        T.RandomBrightness(0.8, 1.8),
        T.RandomContrast(0.6, 1.3),
        T.RandomRotation(angle=[90, 90]),
        T.RandomCrop('relative', (0.5, 0.5))
    ]
    
    image, transforms = T.apply_transform_gens(transform_list, image)
    
    dataset_dict['image'] = torch.as_tensor(image.transpose(2,0,1).astype('float32'))
    
    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop('annotations')
        if obj.get('iscrowd', 0) == 0
    ]
    
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict['instances'] = utils.filter_empty_instances(instances)
    
    return dataset_dict

def setup(args, yaml_name, work_dir): ###fix
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(f'COCO-Detection/{yaml_name}.yaml'))
    cfg.FOLD = 1 ###fix
    cfg.SEED = 42 ###new
    try:
        register_coco_instances('coco_trash_train', {}, '/opt/ml/detection/baseline/detectron2/stratified_kfold/cv_train_'+str(cfg.FOLD)+'.json', '/opt/ml/detection/dataset/') ###fix
    except AssertionError:
        pass
    try:
        register_coco_instances('coco_trash_test', {}, '/opt/ml/detection/baseline/detectron2/stratified_kfold/cv_val_'+str(cfg.FOLD)+'.json', '/opt/ml/detection/dataset/') ###fix
    except AssertionError:
        pass
    
    MetadataCatalog.get('coco_trash_train').thing_classes = ["General trash", "Paper", "Paper pack", "Metal",
                                                             "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]
    cfg.DATASETS.TRAIN = ('coco_trash_train',)
    cfg.DATASETS.TEST = ('coco_trash_test',)
    cfg.DATALOADER.NUM_WOREKRS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f'COCO-Detection/{yaml_name}.yaml')
    cfg.SOLVER.OPTIMIZER = 'AdamW'
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 20000 # 20 epoch
    cfg.SOLVER.STEPS = (8000,12000)
    cfg.SOLVER.GAMMA = 0.005
    cfg.SOLVER.CHECKPOINT_PERIOD = 3000
    # cfg.OUTPUT_DIR = os.path.join('./detectron2/output/','_'.join([cfg.MODEL.META_ARCHITECTURE,cfg.SOLVER.OPTIMIZER, str(cfg.SOLVER.BASE_LR), str(cfg.SOLVER.GAMMA),'t']))
    cfg.OUTPUT_DIR = work_dir
    cfg.EXP = work_dir.split('/')[-1]
    cfg.CFG_FILE_NAME = yaml_name

    cfg.MODEL.RETINANET.NUM_CLASSES = 10


    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10

    cfg.TEST.EVAL_PERIOD = 1000
    
    default_setup(cfg, args)
    cfg.freeze()
    
    return cfg

def setup_wandb(cfg):
    wandb.init(
        project='object_detection_seonah',
        name=f'{cfg.EXP}-fold{cfg.FOLD}-{cfg.CFG_FILE_NAME}',
        entity='mg_generation',
        tags=[cfg.CFG_FILE_NAME],
        group=cfg.EXP,
        job_type=f'fold{cfg.FOLD}',
        config=yaml.safe_load(cfg.dump())
    )
    wandb.define_metric("train/*", summary="min")
    wandb.define_metric("val/*", summary="max")
    wandb.define_metric("cls/*", summary="max")

def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.
    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [
            re.search(rf"%s(\d+)" % path.stem, d) for d in dirs
        ] 
        i = [int(m.groups()[0]) for m in matches if m]  
        n = max(i) + 1 if i else 2
        return f"{path}{n}"

def inference(cfg, model):
    results = OrderedDict()
    dataset_name = cfg.DATASETS.TEST[0]
    if cfg.TEST.AUG.ENABLED:
        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapperTTA) # Use TTA
    else:
        data_loader = build_detection_test_loader(cfg, dataset_name)
    evaluator = get_evaluator(cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name))
    results_i = inference_on_dataset(model, data_loader, evaluator) # inference 결과저장
    results[dataset_name] = results_i
    if comm.is_main_process():
        logger.info("Evaluation results for {} in csv format:".format(dataset_name))
        print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]

    wandb.log(
        {'val/bbox_AP':results['bbox']['AP'], 'val/bbox_AP50':results['bbox']['AP50'], 'val/bbox_AP75':results['bbox']['AP75'],
         'val/bbox_APs':results['bbox']['APs'], 'val/bbox_APm':results['bbox']['APm'], 'val/bbox_APl':results['bbox']['APl'],
         'cls/01 General trash':results['bbox']['AP-General trash'], 'cls/02 Paper':results['bbox']['AP-Paper'],
         'cls/03 Paper pack':results['bbox']['AP-Paper pack'], 'cls/04 Metal':results['bbox']['AP-Metal'],
         'cls/05 Glass':results['bbox']['AP-Glass'], 'cls/06 Plastic':results['bbox']['AP-Plastic'],
         'cls/07 Styrofoam':results['bbox']['AP-Styrofoam'], 'cls/08 Plastic bag':results['bbox']['AP-Plastic bag'],
         'cls/09 Battery':results['bbox']['AP-Battery'], 'cls/10 Clothing':results['bbox']['AP-Clothing']}
    )

    return results

def get_output(cfg, vis, fname):
    dir_name = os.path.join(cfg.OUTPUT_DIR, 'vis_output')
    filepath = os.path.join(dir_name, fname)
    print(f"Saving to {filepath} ...")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    vis.save(filepath)

def check_data(cfg, batch, metadata, scale=1.0):
    for per_img in batch:
        img = per_img['image'].permute(1, 2, 0).cpu().detach().numpy()
        img = utils.convert_image_to_rgb(img, cfg.INPUT.FORMAT)
        visualizer = Visualizer(img, metadata=metadata, scale=scale)
        target_fields = per_img['instances'].get_fields()
        labels = [metadata.thing_classes[i] for i in target_fields['gt_classes']]
        vis = visualizer.overlay_instances(
            labels=labels,
            boxes=target_fields.get("gt_boxes", None),
            masks=target_fields.get("gt_masks", None),
            keypoints=target_fields.get("gt_keypoints", None),
        )
        get_output(cfg, vis, f"{per_img['image_id']}.jpg")

def train(cfg, model, best_AP50, resume=True):
    model.train()

    #optimizer
    if cfg.SOLVER.OPTIMIZER == 'ADAM':
        optimizer = optim.Adam(get_default_optimizer_params(model, weight_decay_norm=0), lr=cfg.SOLVER.BASE_LR)
    elif cfg.SOLVER.OPTIMIZER == 'ADAMW':
        optimizer = optim.AdamW(get_default_optimizer_params(model, weight_decay_norm=0), lr=cfg.SOLVER.BASE_LR)
    else:
        optimizer = optim.SGD(get_default_optimizer_params(model, weight_decay_norm=0), lr=cfg.SOLVER.BASE_LR)
    
    #scheduler
    scheduler = build_lr_scheduler(cfg, optimizer) # cfg [SOLVER.LR_SCHEDULER_NAME, SOLVER.STEPS, SOLVER.MAX_ITER, SOLVER.GAMMA, SOLVER.WARMUP_FACTOR, SOLVER.WARMUP_ITERS, SOLVER.WARMUP_METHOD]
    
    # checkpointer
    checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler) # -> model parameter, optimizer, scheduler를 저장할것.
    max_iter = cfg.SOLVER.MAX_ITER
    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )
    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []
    start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    max_iter = cfg.SOLVER.MAX_ITER
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    # dataloader
    data_loader = build_detection_train_loader(cfg, mapper=Mapper)
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            # check_data(cfg, data, metadata)
            # print('img shape',data[0]['image'].shape)
            storage.iter = iteration
            loss_dict = model(data)            
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            # print('>>loss_dict_reduced : ', loss_dict_reduced)    
            # >>loss_dict_reduced :  {'loss_cls': 2.3326451778411865, 'loss_box_reg': 0.4775199294090271, 'loss_rpn_cls': 0.12368661165237427, 'loss_rpn_loc': 0.030212724581360817}
            # retina net loss_dict_reduced : {'loss_cls' : , 'loss_box_reg' : }
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            # print('>>losses_reduced : ', losses_reduced)   >>losses_reduced :  2.8170977495610714
            # wandb.log({'train_cls_loss':loss_dict_reduced['loss_cls'], 'train_box_reg_loss':loss_dict_reduced['loss_box_reg'], 
            #             'train_rpn_cls_loss':loss_dict_reduced['loss_rpn_cls'], 'train_rpn_loc_loss':loss_dict_reduced['loss_rpn_loc'],
            #             'total_loss':losses_reduced})
            if cfg.MODEL.META_ARCHITECTURE == 'GeneralizedRCNN':
                wandb.log(
                    {'train/loss':losses_reduced, 'train/cls_loss':loss_dict_reduced['loss_cls'],
                    'train/rpn_cls_loss':loss_dict_reduced['loss_rpn_cls'], 'train/rpn_loc_loss':loss_dict_reduced['loss_rpn_loc'],
                    'train/box_reg_loss':loss_dict_reduced['loss_box_reg']}
                )
            elif cfg.MODEL.META_ARCHITECTURE == 'RetinaNet':
                wandb.log(
                    {'train/loss_cls':loss_dict_reduced['loss_cls'],
                    'train/loss_box_reg':loss_dict_reduced['loss_box_reg']}
                )
            else:
                NotImplementedError
            
            
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            # validation
            if (cfg.TEST.EVAL_PERIOD > 0) and ((iteration+1) % cfg.TEST.EVAL_PERIOD == 0) and (iteration != max_iter - 1):
                rst = inference(cfg, model)
                # if cfg.TEST.AUG.ENABLED:
                #     model = GeneralizedRCNNWithTTA(cfg, model)

                if rst['bbox']['AP50'] > best_AP50:
                    pred_best = best_AP50
                    now = rst['bbox']['AP50']
                    print(f'>>>>> BEST AP50 SCORES >>>>>>')
                    print(f'pred_best : {pred_best} -----> now_best : {now}')
                    best_AP50 = now
                    checkpointer.save('AP50_best')
                comm.synchronize()
            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)

def main(args):
    work_dir='/opt/ml/detection/work_dirs_det/det_exp'
    yaml_name = 'retinanet_R_101_FPN_3x'
    
    cfg = setup(args, yaml_name, work_dir) ###fix
    setup_wandb(cfg)
    # cfg_wandb = yaml.safe_load(cfg.dump())

    # wandb.init(
    #     project='object_detection_real_name',
    #     name='run_name',
    #     entity='mg_generation',
    #     tags=[yaml_name],
    #     group=work_dir.split('/')[-1],
    #     job_type=f'fold{cfg.fold}',
    #     config=cfg_wandb
    # ) ###fix

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return inference(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    best_AP50 = 0
    train(cfg, model, best_AP50, resume=True)
    return inference(cfg, model)

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )