from torch.utils.data import Dataset
import torch 
import torch.nn.functional as F
import numpy as np
import json
import collections
from data.make_dataset import load_all_data
import sys
import os
from utils.colmap2nerf import closest_point_2_lines 
from tqdm import tqdm
class Nerf_Dataset(Dataset):
    def __init__(self,cfg,split,device):
        self.cfg=cfg
        self.num_rays=self.cfg.num_rays
        self.split=split
        self.training= self.split in ['train','trainval'] 
        self.device=device
        self.gen_poses = self.cfg.gen_poses
        if((not self.training) and (self.gen_poses)):
            self.n_poses = self.cfg.n_poses
            self.poses= torch.from_numpy(self.generate_poses(self.n_poses)).to(torch.float32).to(self.device)
        self.imgs,self.c2ws,self.H,self.W=load_all_data(self.cfg,self.cfg.data_path,self.split)
        self.imgs = torch.from_numpy(self.imgs).to(torch.uint8)
        self.c2ws = torch.from_numpy(self.c2ws).to(torch.float32)
        self.cam_intrinsics,self.K=self.load_camera_intrinsics(self.H,self.W)  
        self.Rays = collections.namedtuple("Rays", ("origins", "viewdirs"))
        self.imgs=self.imgs.to(self.device)
        self.c2ws=self.c2ws.to(self.device)
        self.K=self.K.to(self.device)
        
        self.n_poses = self.cfg.n_poses
    def __len__(self):
        if((not self.training) and (self.gen_poses)):
            return self.n_poses
        return len(self.imgs)
    def update_num_rays(self,n):
        self.num_rays= n
    def get_data(self,idx):
        
        if((not self.training) and (self.gen_poses)):
            image_id = [idx]
            x, y = torch.meshgrid(
                torch.arange(self.W,device=self.imgs.device),
                torch.arange(self.H,device=self.imgs.device),
                indexing="xy",
            )
            x = x.flatten()
            y = y.flatten()
            num_rays= self.W * self.H    
  
            c2w = self.poses[image_id]

            camera_dirs  = F.pad(
                torch.stack(
                    [
                        (x-self.K[0,2])/self.K[0,0],
                        -1*(y-self.K[1,2])/self.K[1,1]
                    ],
                    dim=-1
                ),
                (0,1),
                value=-1
            )
            #querying ray origins and directions from c2w and cam_intrinsics
            ray_directions = (camera_dirs[:,None,:]*c2w[:,:3,:3]).sum(dim=-1)
            ray_origins= torch.broadcast_to(c2w[:,:3,-1],ray_directions.shape)

            #getting viewdirs normalized
            ray_viewdirs = ray_directions / torch.linalg.norm(ray_directions, dim=-1, keepdims=True)

            ray_origins = torch.reshape(ray_origins, (self.H,self.W, 3))
            ray_viewdirs = torch.reshape(ray_viewdirs, (self.H,self.W, 3))

            rays = self.Rays(origins=ray_origins, viewdirs=ray_viewdirs)
            return {
                "rays":rays, # [h,w,3] or [num_rays,3]
                "height":self.H,
                "width":self.W
                }
        else :
            num_rays = self.num_rays
            if(self.training):
                if(self.cfg.batch_over_images):
                    image_id = torch.randint(0,len(self.imgs),size=(num_rays,),device=self.imgs.device)
                else :
                    image_id = [idx] * num_rays
            
                x= torch.randint(0,self.W,size=(num_rays,),device=self.imgs.device)
                y= torch.randint(0,self.H,size=(num_rays,),device=self.imgs.device)
            else :
                image_id = [idx]
                x, y = torch.meshgrid(
                    torch.arange(self.W,device=self.imgs.device),
                    torch.arange(self.H,device=self.imgs.device),
                    indexing="xy",
                )
                x = x.flatten()
                y = y.flatten()
                num_rays= self.W * self.H    
            rgba = self.imgs[image_id,y,x]/255.0

            c2w = self.c2ws[image_id]
            camera_dirs  = F.pad(
                torch.stack(
                    [
                        (x-self.K[0,2])/self.K[0,0],
                        -1*(y-self.K[1,2])/self.K[1,1]
                    ],
                    dim=-1
                ),
                (0,1),
                value=-1
            )
            #querying ray origins and directions from c2w and cam_intrinsics
            ray_directions = (camera_dirs[:,None,:]*c2w[:,:3,:3]).sum(dim=-1)
            ray_origins= torch.broadcast_to(c2w[:,:3,-1],ray_directions.shape)

            #getting viewdirs normalized
            ray_viewdirs = ray_directions / torch.linalg.norm(ray_directions, dim=-1, keepdims=True)


            if(self.training):
                ray_origins = torch.reshape(ray_origins, (num_rays, 3))
                ray_viewdirs = torch.reshape(ray_viewdirs, (num_rays, 3))
                rgba = torch.reshape(rgba, (num_rays, 4))
            else :
                ray_origins = torch.reshape(ray_origins, (self.H,self.W, 3))
                ray_viewdirs = torch.reshape(ray_viewdirs, (self.H,self.W, 3))
                rgba = torch.reshape(rgba, (self.H,self.W, 4))
            rays = self.Rays(origins=ray_origins, viewdirs=ray_viewdirs)
            return {
                "rgba":rgba, # [h,w,4] or [num_rays,4]
                "rays":rays, # [h,w,3] or [num_rays,3]
                "height":self.H,
                "width":self.W
                }

    def preprocess(self,data):
    
        if ((not self.training) and (self.gen_poses)) :
            rays= data["rays"]
            if(self.cfg.color_bkgd=='white'):
                color_bkgd=torch.ones(3,device=self.imgs.device)
            return {
                "rays" : rays,
                "color_bkgd" : color_bkgd,
                "height":data["height"],
                "width":data["width"]
            }
        else :
            rgba, rays = data["rgba"],data["rays"]
            #splitting rgba into rgb pixels and opacity alpha
            pixels,alpha=torch.split(rgba, [3, 1], dim=-1)
            
            if(self.cfg.color_bkgd=='white'):
                color_bkgd=torch.ones(3,device=self.imgs.device)
            #if the opacity is low then fill with backrgound image

            pixels = pixels*alpha + color_bkgd*(1.0-alpha)
        
            return {
                "pixels":pixels,
                "rays":rays,
                "color_bkgd":color_bkgd,
                "height":data["height"],
                "width":data["width"]}
        
    @torch.no_grad()
    def __getitem__(self,idx):
        data=self.get_data(idx)
        data=self.preprocess(data)
        return data
    
    def load_camera_intrinsics(self,h,w):
        #transforms_path = self.cfg.data_path 
        if(self.split=='all' or self.split=='trainval'):
            f = open(os.path.join(self.cfg.data_path,"transforms.json"))
        else :
            f = open(os.path.join(self.cfg.data_path,f"transforms_{self.split}.json"))

        
        data = json.load(f)
        camera_intrinsics={}
        if(self.cfg.colmap):
            camera_intrinsics["camera_angle_x"]=data["camera_angle_x"]
            camera_intrinsics["camera_angle_y"]=data["camera_angle_y"]
            camera_intrinsics["fl_x"]=data["fl_x"]/ self.cfg.downscale
            camera_intrinsics["fl_y"]=data["fl_y"]/ self.cfg.downscale
            camera_intrinsics["k1"]=data["k1"]
            camera_intrinsics["k2"]=data["k2"]
            camera_intrinsics["p1"]=data["p1"]
            camera_intrinsics["p2"]=data["p2"]
            camera_intrinsics["cx"]=data["cx"]/ self.cfg.downscale
            camera_intrinsics["cy"]=data["cy"]/ self.cfg.downscale
            camera_intrinsics["h"]=int(data["h"]/self.cfg.downscale)
            camera_intrinsics["w"]=int(data["w"]/self.cfg.downscale)
        else :
            camera_angle_x = float(data["camera_angle_x"])
            focal = 0.5 * w / np.tan(0.5 * camera_angle_x)
            camera_intrinsics["fl_x"]=focal/ self.cfg.downscale
            camera_intrinsics["fl_y"]=focal/ self.cfg.downscale
            camera_intrinsics["cx"]=w/(2* self.cfg.downscale)
            camera_intrinsics["cy"]=h/(2* self.cfg.downscale)
        f.close()
        
        K = torch.tensor(
            [
                [camera_intrinsics["fl_x"], 0, camera_intrinsics["cx"]],
                [0, camera_intrinsics["fl_y"], camera_intrinsics["cy"]],
                [0, 0, 1],
            ],
            dtype=torch.float32,
        )  # (3, 3)
        return camera_intrinsics,K

    def generate_poses(self,n_poses):
        print("[INFO] computing center of attention , radius and angle of view...")
        totw = 0.0 # total weight
        totp = np.array([0.0, 0.0, 0.0]) # xyz of the center of attention == total position
        if(self.split=='all' or self.split=='trainval'):
            with open(
                os.path.join(self.cfg.data_path,"transforms.json"), "r"
            ) as f:
                meta = json.load(f)
        else :
            with open(
                os.path.join(self.cfg.data_path,f"transforms_{self.split}.json"), "r"
            ) as f:
                meta = json.load(f)

        frames = meta["frames"]
        radius = 0
        phi =0
        for f in frames:
            trans = np.array(f["transform_matrix"])[0:3,-1]
            r = np.sqrt(trans[0]**2+trans[1]**2+trans[2]**2)
            phi += np.arccos(trans[2]/r)
            radius+=r
            mf = np.array(f["transform_matrix"])[0:3,:]
            for g in frames:
                mg = np.array(g["transform_matrix"])[0:3,:]
                p, weight = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
                if weight > 0.01:
                    totp += p * weight 
                    totw += weight
                    
        totp /= totw

        radius/=len(frames)
        phi/=len(frames) 
        print({"center of attention":totp,
               "radius":radius,
               "phi(°)":np.rad2deg(phi),
               })
        #nerf_synthetic optimal_values are radius=4 / phi = np.deg2rad(50)

        poses = []
        theta_angles = torch.linspace(0, 361, n_poses)
        for i in theta_angles :
            theta = np.deg2rad(i)
            center = np.array([
                radius * np.sin(phi) * np.cos(theta),
                radius * np.sin(phi) * np.sin(theta),
                radius * np.cos(phi),
            ]) + totp
            # look at
            def normalize(v):
                return v / (np.linalg.norm(v) + 1e-10)
                #return v
            forward_v = normalize(center)
            up_v = np.array([0, 0, 1])
            right_v = normalize(np.cross(forward_v, up_v))
            up_v = normalize(np.cross(right_v, forward_v))
            # make pose
            pose = np.eye(4)
            pose[:3, :3] = np.stack((right_v, up_v, forward_v), axis=-1)
            pose[:3, 3] = center
            poses.append(pose)
        
        poses = np.stack(poses, axis=0)
        return poses
    
    
                