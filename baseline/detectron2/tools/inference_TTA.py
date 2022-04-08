import os
import copy
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import detectron2
from fvcore.transforms import HFlipTransform, NoOpTransform
from detectron2.data.transforms import (
    RandomFlip,
    ResizeShortestEdge,
    ResizeTransform,
    apply_augmentations,
)
from detectron2.data import detection_utils as utils
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.modeling.test_time_augmentation import DatasetMapperTTA
from detectron2.utils.visualizer import Visualizer

try:
    register_coco_instances('coco_trash_test', {}, '../dataset/test.json', '../dataset/')
except AssertionError:
    pass

cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'))
cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/retinanet_R_101_FPN_3x.yaml'))

# config 수정하기
cfg.DATASETS.TEST = ('coco_trash_test',)

cfg.DATALOADER.NUM_WOREKRS = 2

cfg.OUTPUT_DIR = './detectron2/output/RetinaNet_SGD_0.01_0.1'

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')

cfg.MODEL.RETINANET.NUM_CLASSES = 10
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.3

predictor = DefaultPredictor(cfg)

def MyMapperTTA(dataset_dict):
    min_sizes = cfg.TEST.AUG.MIN_SIZES
    max_size = cfg.TEST.AUG.MAX_SIZE
    flip = cfg.TEST.AUG.FLIP
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict['file_name'], format='BGR')
    dataset_dict['image'] = image
    # numpy_image = dataset_dict["image"].permute(1, 2, 0).numpy()
    # print(type(dataset_dict['image']))
    shape = dataset_dict['image'].shape
    # print(shape)
    orig_shape = (dataset_dict["height"], dataset_dict["width"])
    if shape[:2] != orig_shape:
        # It transforms the "original" image in the dataset to the input image
        pre_tfm = ResizeTransform(orig_shape[0], orig_shape[1], shape[0], shape[1])
    else:
        pre_tfm = NoOpTransform()
    aug_candidates = []  # each element is a list[Augmentation]
    for min_size in min_sizes:
        resize = ResizeShortestEdge(min_size, max_size)
        aug_candidates.append([resize])  # resize only
        if flip:
            flip = RandomFlip(prob=1.0)
            aug_candidates.append([resize, flip])  # resize + flip

    # Apply all the augmentations
    ret = []
    for aug in aug_candidates:
        new_image, tfms = apply_augmentations(aug, np.copy(dataset_dict['image']))

        dic = copy.deepcopy(dataset_dict)
        dic["transforms"] = pre_tfm + tfms
        dic["image"] = new_image
        ret.append(dic)
    return ret

test_loader = build_detection_test_loader(cfg, 'coco_trash_test', MyMapperTTA)

prediction_strings = []
file_names = []

class_num = 10
metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
import matplotlib.pyplot as plt

for transformed_testloaders in tqdm(test_loader):
    for data in transformed_testloaders:
    
        prediction_string = ''
        data = data[0]

        # data visualization
        img = utils.convert_image_to_rgb(data['image'], cfg.INPUT.FORMAT)
        outputs = predictor(data['image'])['instances']
        plt.imshow(img)
        plt.show()

        targets = outputs.pred_classes.cpu().tolist()
        boxes = [i.cpu().detach().numpy() for i in outputs.pred_boxes]
        scores = outputs.scores.cpu().tolist()
        
        for target, box, score in zip(targets,boxes,scores):
            prediction_string += (str(target) + ' ' + str(score) + ' ' + str(box[0]) + ' ' 
            + str(box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' ')
            
        prediction_strings.append(prediction_string)
        file_names.append(data['file_name'].replace('../dataset/',''))

# submission = pd.DataFrame()
# submission['PredictionString'] = prediction_strings
# submission['image_id'] = file_names
# submission.to_csv(os.path.join(cfg.OUTPUT_DIR, f'submission_det2_TTA.csv'), index=None)
# submission.head()