import numpy as np
import imageio.v2 as imageio
import os
import json
import sys 
from ..utils.utils import segment_single_image
import cv2
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:     %(message)s'
    )
def load_all_data(hparams,split,use_npz):
    if(use_npz):
        data = np.load(hparams["data_path"])
        images = data["images"]
        camtoworlds= data["poses"]
        H = data["h"]
        W = data["w"]
        return images, camtoworlds ,H, W

    if(hparams["colmap_generated"]):
        if(split=='all' or split=='trainval'):
            with open(
                os.path.join(hparams["data_path"], "transforms_colmap.json"), "r"
            ) as fp:
                meta = json.load(fp)
        else :
            with open(
                os.path.join(hparams["data_path"], f"transforms_{split}_colmap.json"), "r"
            ) as fp:
                meta = json.load(fp)
    else :
        if(split=='all' or split=='trainval'):
            with open(
                os.path.join(hparams["data_path"], "transforms.json"), "r"
            ) as fp:
                meta = json.load(fp)
        else :
            with open(
                os.path.join(hparams["data_path"], f"transforms_{split}.json"), "r"
            ) as fp:
                meta = json.load(fp)
    images = []
    camtoworlds = []
    ext=meta["frames"][0]["file_path"].split(".")[1]
    if (ext=='png' or (not hparams["colmap_generated"])):#already masked
        for i in range(len(meta["frames"])):
            frame = meta["frames"][i]
            fname = os.path.join(hparams["data_path"], frame["file_path"] +".png")
            rgba = imageio.imread(fname)
            camtoworlds.append(frame["transform_matrix"])
            images.append(rgba)
    else : #needs masking
        try:
            from detectron2.config import get_hparams
            from detectron2 import model_zoo
            from detectron2.engine import DefaultPredictor
        except ModuleNotFoundError:
            print("Detectron2 is not installed.")
            import subprocess
            package = 'git+https://github.com/facebookresearch/detectron2.git'
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            from detectron2.config import get_hparams
            from detectron2 import model_zoo
            from detectron2.engine import DefaultPredictor
        config = get_hparams()
        # Add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        config.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo.
        config.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        predictor = DefaultPredictor(config)
        category2id = json.load(open(os.path.join("", "utils/category2id.json"), "r"))
        mask_category=["car"]
        mask_ids = [category2id[c] for c in mask_category]
        cnt=0
        for i in range(len(meta["frames"])):
            frame = meta["frames"][i]
            fname = os.path.join(hparams["data_path"], frame["file_path"] )
            rgba,found = segment_single_image(predictor,fname,mask_ids) 
            if not found  :
                continue
            cnt+=1
            camtoworlds.append(frame["transform_matrix"])
            images.append(rgba)
            if hparams["saving.save_mask"] :
                cv2.imwrite(os.path.join(hparams["saving.masked_folder"], f"masked_{fname.split('.')[0]}.png"), rgba)
        logging.info(f"nb of images masked:{cnt} ")
    images = np.stack(images, axis=0)
    camtoworlds = np.stack(camtoworlds, axis=0)
    H,W = images.shape[1:3]
    return images, camtoworlds ,H, W