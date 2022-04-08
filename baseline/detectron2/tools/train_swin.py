import itertools
import logging
import os
from collections import OrderedDict
from detectron2.engine.train_loop import HookBase
import torch
import numpy as np
import copy

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling.test_time_augmentation import DatasetMapperTTA
from detectron2.data import MetadataCatalog, build_detection_test_loader, build_detection_train_loader
import albumentations as A
from detectron2.data import detection_utils as utils
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.solver.build import maybe_add_gradient_clipping, get_default_optimizer_params

from swint import add_swint_config
from detectron2.data.datasets import register_coco_instances
import wandb
import re
import glob
import yaml
from pathlib import Path
import detectron2.data.transforms as T

class LossEvalHook(HookBase):
    def __init__(self, cfg):
        super.__init__()
        self.cfg = cfg.clone()
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.TEST
        self._loader = iter(build_detection_train_loader(self.cfg))

    def after_step(self):
        data = next(self._loader)
        with torch.no_grad():
            loss_dict = self.trainer.model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {"val_" + k: v.item() for k, v in 
                                 comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            wandb.log({'val_total_loss':losses_reduced})
            wandb.log({'val_'+k : v.item() for k, v in comm.reduce_dict(loss_dict).items()})
            if comm.is_main_process():
                self.trainer.storage.put_scalars(total_val_loss=losses_reduced, 
                                                 **loss_dict_reduced)

def Mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict['file_name'], format='BGR')
    
    transform_list = [
        T.RandomCrop('relative', (0.5, 0.5)),
        T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        T.RandomBrightness(0.8, 1.8),
        T.RandomContrast(0.6, 1.3),
        T.RandomRotation(angle=[90, 90]),
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

class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_train_loader(cls, cfg, sampler=None):
        return build_detection_train_loader(
        cfg, mapper = Mapper, sampler = sampler
        )
    
    @classmethod
    def build_test_loader(cls, cfg, sampler=None):
        if cfg.TEST.AUG.ENABLED:
            mapper = DatasetMapperTTA(cfg)
            return build_detection_test_loader(
            cfg, mapper= mapper,sampler = sampler
            )
        else:
            return build_detection_test_loader(
            cfg, sampler = sampler
            )

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    @classmethod
    def build_optimizer(cls, cfg, model):
        params = get_default_optimizer_params(
            model,
            base_lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
            bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
            weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
        )

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM,
                nesterov=cfg.SOLVER.NESTEROV,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        elif optimizer_type == "AdamW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR, betas=(0.9, 0.999),
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        return optimizer

    @classmethod
    def test(cls, cfg, model):
        logger = logging.getLogger(__name__)
        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name, mapper=DatasetMapperTTA)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            try:
                evaluator = cls.build_evaluator(cfg, dataset_name)
            except NotImplementedError:
                logger.warn(
                    "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                    "or implement its `build_evaluator` method."
                )
                results[dataset_name] = {}
                continue
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
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


def setup(args, yaml_name, work_dir):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_swint_config(cfg)
    cfg.merge_from_file(f'/opt/ml/detection/baseline/detectron2/configs/SwinT/{yaml_name}.yaml')
    cfg.merge_from_list(args.opts)
    cfg.FOLD = 3 ###fix
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
    cfg.SOLVER.GAMMA = 0.005
    cfg.SOLVER.MAX_ITER = 30000 # 20 epoch
    cfg.SOLVER.STEPS = (28000,29000)
    cfg.SOLVER.CHECKPOINT_PERIOD = 3000
    cfg.SOLVER.OPTIMIZER = 'AdamW'
    cfg.OUTPUT_DIR = work_dir
    cfg.EXP = work_dir.split('/')[-1]
    cfg.CFG_FILE_NAME = yaml_name

    cfg.MODEL.RETINANET.NUM_CLASSES = 10

    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def setup_wandb(cfg):
    wandb.init(
        project='object_detection_seonah',
        name=f'{cfg.EXP}-fold{cfg.FOLD}-{cfg.CFG_FILE_NAME}',
        entity='mg_generation',
        tags=[cfg.CFG_FILE_NAME],
        group=cfg.EXP,
        job_type=f'fold{cfg.FOLD}',
        config=yaml.safe_load(cfg.dump()),
        sync_tensorboard=True
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


def main(args):
    work_dir='/opt/ml/detection/baseline/detectron2/work_dirs/exp'
    if not os.path.exists(work_dir):
        os.makedirs(work_dir, exist_ok=True)
    yaml_name = 'retinanet_swint_T_FPN_3x_'
    cfg = setup(args, yaml_name, work_dir)
    setup_wandb(cfg)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    val_loss = LossEvalHook(cfg)
    trainer.register_hooks([val_loss])
    # swap the order of PeriodicWriter and ValidationLoss
    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


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