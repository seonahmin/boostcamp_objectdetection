import logging
import albumentations as A
import copy
import numpy as np
import torch
from detectron2.config import configurable
from detectron2.data import detection_utils as utils

__all__ = ["AlbumentationMapper"]

data_class_dict = {"General trash":0, "Paper":1, "Paper pack":2, "Metal":3, 
                "Glass":4, "Plastic":5, "Styrofoam":6, "Plastic bag":7, "Battery":8, "Clothing":9}

class AlbumentationMapper:
    @configurable
    def __init__(self, cfg, is_train:bool = True):
        aug_list = [
            A.RandomCrop(512, 512),
            A.Flip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ToGray(0.3),
            A.GaussNoise(),
            A.MotionBlur(),
            A.MedianBlur(),
            A.CLAHE(),
            A.Emboss(),
            A.Sharpen()
        ]
        self.metadata = cfg.DATASETS.METADATA
        self.transform = A.Compose(aug_list, bbox_params=A.BboxParams(format='coco', min_area=1024, min_visibility=0.1, label_fields=self.metadata))
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {self.transform}")

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format="BGR")

        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image

        prev_anno = dataset_dict["annotations"]
        bboxes = np.array([obj["bbox"] for obj in prev_anno], dtype=np.float32)
        category_id = np.array([obj["category_id"] for obj in dataset_dict["annotations"]], dtype=np.int64)
        # category_id = np.arange(len(dataset_dict["annotations"]))

        # transformed = self.transform(image=image, bboxes=bboxes, category_ids=category_id)
        transformed = self.transform(image=image, bboxes=bboxes, class_labels=self.metadata)
        image = transformed["image"]
        annos = []
        # for i, j in enumerate(transformed["category_ids"]):
        #     d = prev_anno[j]
        #     d["bbox"] = transformed["bboxes"][i]
        #     annos.append(d)
        dataset_dict.pop("annotations", None)  # Remove unnecessary field.

        # if not self.is_train:
        #     # USER: Modify this if you want to keep them for some reason.
        #     dataset_dict.pop("annotations", None)
        #     dataset_dict.pop("sem_seg_file_name", None)
        #     return dataset_dict

        image_shape = image.shape[:2]  # h, w
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        instances = utils.annotations_to_instances(annos, image_shape)
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict
        