from detectron2.engine import DefaultPredictor
import os
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg

import glob
def get_eval_images(img_dir):
    return glob.glob(os.path.join(img_dir, '*.JPG'))
    
sushi_metadata = MetadataCatalog.get("sushi_train").set(thing_classes=["blk_sushi", "red_sushi"])
print('sushi_metadata', sushi_metadata)

cfg = get_cfg()
cfg.merge_from_file("/home/auro/detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.65   # set the testing threshold for this model
cfg.DATASETS.TEST = ("sushi_val", )
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # has two class ["blk_sushi", "red_sushi"]

predictor = DefaultPredictor(cfg)

from detectron2.utils.visualizer import ColorMode
for img_name  in get_eval_images('../data/small_images/val'):
    img = cv2.imread(img_name)
    outputs = predictor(img)
    v = Visualizer(img[:, :, ::-1],
                   metadata=sushi_metadata,
                   scale=1.5, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    cv2.imshow('window', v.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
