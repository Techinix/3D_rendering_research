import torch
import humanize,psutil,GPUtil
import numpy as np
import random
import logging
from PIL import Image
import json
import cv2
import os
import sys
import tqdm
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def save_checkpoint(state, filename):
    logging.info('--> Saving checkpoint')
    torch.save(state, filename)
    logging.info("Checkpoint saved successfully !")
def load_checkpoint(checkpoint, model,estimator,opt,scheduler):
    logging.info('--> Loading checkpoint')
    model.load_state_dict(checkpoint["model"])
    estimator.load_state_dict(checkpoint["estimator"])
    opt.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    logging.info("Checkpoint loaded successfully !")


def display_imgs():
   pass
def get_imgs_path():
   pass
def mem_report(c):
  print("\n Generating MEMORY Report after Executing: ",c,"\n")
  print("CPU RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ))
  
  GPUs = GPUtil.getGPUs()
  for i, gpu in enumerate(GPUs):
    print('GPU {:d} ... Mem Free: {:.0f}MB / {:.0f}MB | Utilization {:3.0f}%'.format(i, gpu.memoryFree, gpu.memoryTotal, gpu.memoryUtil*100))
  
def to_png(img):
      alpha = np.sum(img, axis=-1) > 0
      alpha = np.uint8(alpha * 255)
      res = np.dstack((img, alpha))
      return res  



def segment_images(mask_category=["car"],scene="car"):
                      
    try:
        import detectron2
    except ModuleNotFoundError:
              
      try:
        import torch
      except ModuleNotFoundError:
        print("PyTorch is not installed. For automatic masking, install PyTorch from https://pytorch.org/")
        sys.exit(1)

      input("Detectron2 is not installed. Press enter to install it.")
      import subprocess
      package = 'git+https://github.com/facebookresearch/detectron2.git'
      subprocess.check_call([sys.executable, "-m", "pip", "install", package])
      from detectron2.config import get_cfg
      from detectron2 import model_zoo
      from detectron2.engine import DefaultPredictor

      IMAGE_FOLDER = f"/content/drive/MyDrive/nerf_scenes/{scene}/images"
      OUTPUT_FOLDER = f"/content/drive/MyDrive/nerf_scenes/{scene}/output_png_compressed"


      category2id = json.load(open(os.path.join("", "category2id.json"), "r"))
      mask_ids = [category2id[c] for c in mask_category]

      cfg = get_cfg()
      # Add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
      cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
      cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
      # Find a model from detectron2's model zoo.
      cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
      predictor = DefaultPredictor(cfg)

      for filename in tqdm(os.listdir(IMAGE_FOLDER), desc="Masking images", unit="images"):
          basename, ext = os.path.splitext(filename)
          ext = ext.lower()

          # Only consider image files
          if ext != ".jpg" and ext != ".jpeg" and ext != ".png" and ext != ".exr" and ext != ".bmp":
              continue

          img = cv2.imread(os.path.join(IMAGE_FOLDER, filename))
          outputs = predictor(img)
          output_mask = np.zeros((img.shape[0], img.shape[1]))
          best_surface=-1
          best_mask=[]
          found=False
          for i in range(len(outputs['instances'])):
              if outputs['instances'][i].pred_classes.cpu().numpy()[0] in mask_ids:
                  pred_mask = outputs['instances'][i].pred_masks.cpu().numpy()[0]
                  mask_surface=pred_mask.sum()
                  if(mask_surface>best_surface):
                      best_surface=mask_surface
                      best_mask=pred_mask
                      found=True
          output_mask = best_mask
          if (not found) :
              continue
          else :
              for i in range(img.shape[2]):
                  img[:,:,i]=img[:,:,i]*output_mask

              img_png=to_png(img)
              cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"masked_{basename}.png"), img_png)


def segment_single_image(predictor,img,mask_ids):
                      
      
      outputs = predictor(img)
      output_mask = np.zeros((img.shape[0], img.shape[1]))
      best_surface=-1
      best_mask=[]
      found=False
      for i in range(len(outputs['instances'])):
          if outputs['instances'][i].pred_classes.cpu().numpy()[0] in mask_ids:
              pred_mask = outputs['instances'][i].pred_masks.cpu().numpy()[0]
              mask_surface=pred_mask.sum()
              if(mask_surface>best_surface):
                  best_surface=mask_surface
                  best_mask=pred_mask
                  found=True
      output_mask = best_mask
      if (not found) :
          return img,False
      else :
          for i in range(img.shape[2]):
              img[:,:,i]=img[:,:,i]*output_mask

          img_png=to_png(img)
          return img_png,True