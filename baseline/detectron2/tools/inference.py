import os
import copy
import torch
from tqdm import tqdm
import pandas as pd
import detectron2
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
from swint import add_swint_config

# Register Dataset
try:
    register_coco_instances('coco_trash_test', {}, '../dataset/test.json', '../dataset/')
except AssertionError:
    pass

# config 불러오기
cfg = get_cfg()
add_swint_config(cfg)
# cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'))
# cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/retinanet_R_101_FPN_3x.yaml'))
cfg.merge_from_file(f'/opt/ml/detection/baseline/detectron2/configs/SwinT/retinanet_swint_T_FPN_3x_.yaml')
work_dir='/opt/ml/detection/baseline/detectron2/work_dirs/exp'
# config 수정하기
cfg.DATASETS.TEST = ('coco_trash_test',)

cfg.DATALOADER.NUM_WOREKRS = 2

cfg.OUTPUT_DIR = work_dir

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')

cfg.MODEL.RETINANET.NUM_CLASSES = 10
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.3

predictor = DefaultPredictor(cfg)

def MyMapper(dataset_dict):
    
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict['file_name'], format='BGR')
    
    dataset_dict['image'] = image
    
    return dataset_dict

# test loader
test_loader = build_detection_test_loader(cfg, 'coco_trash_test', mapper=MyMapper)

prediction_strings = []
file_names = []

class_num = 10

for data in tqdm(test_loader):
    # print(data)
    prediction_string = ''
    
    data = data[0]
    
    outputs = predictor(data['image'])['instances']
    
    targets = outputs.pred_classes.cpu().tolist()
    boxes = [i.cpu().detach().numpy() for i in outputs.pred_boxes]
    scores = outputs.scores.cpu().tolist()
    
    for target, box, score in zip(targets,boxes,scores):
        prediction_string += (str(target) + ' ' + str(score) + ' ' + str(box[0]) + ' ' 
        + str(box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' ')
        
    prediction_strings.append(prediction_string)
    file_names.append(data['file_name'].replace('../dataset/',''))

submission = pd.DataFrame()
submission['PredictionString'] = prediction_strings
submission['image_id'] = file_names
submission.to_csv(os.path.join(cfg.OUTPUT_DIR, f'submission_det2.csv'), index=None)
submission.head()