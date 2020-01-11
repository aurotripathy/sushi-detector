import os
import numpy as np
import json
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
import itertools
import cv2
import random
from pudb import set_trace


def get_sushi_dicts(img_dir):
    json_file = os.path.join(img_dir, "ten_via_export_json.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        
        filename = os.path.join(img_dir, v["filename"])
        print(filename)
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annos = v["regions"]
        objs = []
        for anno in annos:
            assert anno["region_attributes"]
            if anno['region_attributes']['sushi'] == 'blk_sushi':
                category_id = 0
            else:
                category_id = 1

            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = list(itertools.chain.from_iterable(poly))

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": category_id,
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        print(record['annotations'])
        dataset_dicts.append(record)
    return dataset_dicts



from detectron2.data import DatasetCatalog, MetadataCatalog
for phase in ["train", "val"]:
    DatasetCatalog.register("sushi_" + phase, lambda phase=phase: get_sushi_dicts("../data/small_images/" + phase))
    MetadataCatalog.get("sushi_" + phase).set(thing_classes=["blk", "red"])
sushi_metadata = MetadataCatalog.get("sushi_train")
print('sushi_matadata', sushi_metadata)

dataset_dicts = get_sushi_dicts("../data/small_images/train")

# View a few
for dict in random.sample(dataset_dicts, 3):
    img = cv2.imread(dict["file_name"])
    visualizer = Visualizer(img[:, :, ::-1],  # reverse channels
                            metadata=sushi_metadata,
                            scale=1.0)
    vis = visualizer.draw_dataset_dict(dict)
    cv2.imshow('window', vis.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# train
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file("/home/auro/detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("sushi_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # has two class ["blk_sushi", "red_sushi"]

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

print('Training done! Next evaluate from the "val" folder')
