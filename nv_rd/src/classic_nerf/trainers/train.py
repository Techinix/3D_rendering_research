import numpy as np
import torch
import logging
from datetime import datetime
import torch.nn.functional as F
import time
import nerfacc
import imageio.v2 as imageio
from models import occ_grid
from utils.utils import seed_everything,load_checkpoint,save_checkpoint
import os 
from visualization.visualize import generate_video
from torch.utils.tensorboard import SummaryWriter


import piq
from lpips import LPIPS
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:     %(message)s'
    )


from PIL import Image




def train_loop(device,cfg,config,model_name):
    '''Train TinyNeRF model
    '''
    """
        model : tiny nerf model
        optimizer : optimizer
        data : dataloader
        near_thresh,far_thresh : Near and far clipping thresholds for depth values
        num_encoding_functions : Number of functions used in the positional encoding (Be sure to update the model if this number changes).
        depth_samples_per_ray :  Number of depth samples along each ray.
        chunksize :   this isn't batchsize in the conventional sense.
          This only specifies the number of rays to be queried in one go. 
          Backprop still happensonly after all rays from the current "bundle" are queried and rendered).
          Use chunksize of about 4096 to fit in ~1.4 GB of GPU memory.
        lr : learning rate
        max_steps : number of iterations
        display_every : display poses every n iter
        model_name : model_name
    """
    logging.info("Prepare data and initialize model...")
 
    # Seed RNG, for repeatability
    seed_everything(cfg.seed)

    model = config["model"].to(device)
    num_gpus = torch.cuda.device_count()
    print("num_gpus : ",num_gpus)
    """model = torch.nn.parallel.DataParallel(model, device_ids=list(range(num_gpus)), dim=0)"""

    opt = config["optimizer"](model.parameters(), lr=cfg.lr) 
    """scheduler = torch.optim.lr_scheduler.MultiStepLR(
        opt,
        milestones=[
            cfg.max_steps // 2,
            cfg.max_steps * 3 // 4,
            cfg.max_steps * 5 // 6,
            cfg.max_steps * 9 // 10,
        ],
        gamma=0.33,
    )"""

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, cfg.max_steps+1, eta_min=0, last_epoch=- 1, verbose=False)

    
    train_dataset = config["train_dataloader"]
    test_dataset = config["test_dataloader"]
    loss_fn = config["loss_fn"]
    
    
    
    # training parameters
    target_sample_batch_size = 1 << 16
    # scene parameters
    side_x=1 # 5anfouret il karhba ( moderate)
    side_y=2.6 # car_side (must be big)
    side_z=0.8 # height (moderate)
    side_x,side_y,side_z=2,2,2
    aabb = torch.tensor([-side_x, -side_y, -side_z, side_x, side_y, side_z], device=device)
    
    # render parameters

    render_step_size = 5e-3
    grid_resolution = 128
    grid_nlvl = 1
    estimator = nerfacc.OccGridEstimator(roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl).to(device)


    #Tensorboard for logging results
    current_dir = os.getcwd()

    # Create the path for the "reports" folder
    reports_path = os.path.join(os.path.abspath(os.path.join(current_dir, os.pardir)), "reports")

    time.time()
    now = datetime.now() # current date and time
    date_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    #results_path = os.path.join(reports_path,os.path.join("",f"classic_nerf_occ_grid/runs/exp_{date_string}"))
    results_path = os.path.join("/kaggle/working",f"classic_nerf_occ_grid/runs/exp_{date_string}")
    if(not os.path.exists(results_path)):
        os.makedirs(results_path)
    writer = SummaryWriter(results_path)
    """
    Train-Eval-Repeat!
    """
    
    # Lists to log metrics etc.
    psnrs = []
    losses = []
    lpipss = []
    ssims = []
    lpips_net = LPIPS(net="vgg").to(device)
    perm_fn = lambda x : x.permute(2,0,1)[None,...]
    lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2)  * 2 - 1
    lpips_metric = lambda x, y: lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()
    ssim_metric= lambda x,y  : piq.ssim(perm_fn(x),perm_fn(y))
    save_step = cfg.max_steps/5

    if(not cfg.load_ckp_path==None ):
        checkpoint=torch.load(cfg.load_ckp_path)
        print(
            f"model name is : {checkpoint['name']}",
            f"checkpoint trained for : {checkpoint['step']}",
            #f"training took : {checkpoint['training_time']}",
            )
        load_checkpoint(checkpoint, model,estimator,opt,scheduler)
        
    if(cfg.train) :
        
        print(f"Start training model for {cfg.max_steps} epochs...")
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        start_time = time.time()
      
        
        for e in range(1, cfg.max_steps+1):
            model.train()
            estimator.train()
            i=torch.randint(0, len(train_dataset), (1,)).item()
            
            data=train_dataset[i]
            
            pixels=data["pixels"]
            rays=data["rays"]
            render_bkgd = data["color_bkgd"]

            
            def occ_eval_fn(x):
                density = model.query_density(x)
                return density * render_step_size
            
            estimator.update_every_n_steps(
                step=e,
                occ_eval_fn=occ_eval_fn,
                occ_thre=1e-2
            )
           
            rgb,acc,depth, n_rendering_samples = occ_grid.render_image_with_occgrid(
                model,
                estimator,
                rays,
                #rendering_optins
                render_step_size=render_step_size,
                render_bkgd=render_bkgd,
            )
            if(n_rendering_samples==0):
                continue

            if(target_sample_batch_size > 0):
                # dynamic batch size for rays to keep sample batch size constant.
                num_rays = len(pixels)
                num_rays = int(
                    num_rays * (target_sample_batch_size / float(n_rendering_samples))
                )
                train_dataset.update_num_rays(num_rays)
            
            loss =  F.smooth_l1_loss(rgb,pixels)


            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()
            # Display images/plots/stats
            if e % cfg.disp_every==0: 
                loss = torch.nn.functional.mse_loss(rgb, pixels)
                psnr = -10. * torch.log10(loss) 
                writer.add_scalar('Loss/train', loss, e)
                writer.add_scalar('Psnr/train', psnr, e)
                losses.append(loss)
                psnrs.append(psnr)

            if e % save_step == 0 : 
                end_time = time.time()
                elapsed_time=end_time-start_time
                print(
                    f"Epoch n° {e}"
                    f"Elapsed Time  = {end_time-start_time} | Current Learning Rate = {scheduler.get_last_lr()[0]}"
                    f"loss={loss:.5f} | psnr={psnr:.2f} "
                    f"n_rendering_samples={n_rendering_samples:d} | num_rays={len(pixels):d} | "
                    f"max_depth={depth.max():.3f} | "
                )
                # save model using state_dict
                
                now = datetime.now() # current date and time
                date_string = now.strftime("%Y-%m-%d_%H-%M-%S")
                checkpoint = {
                    "step": e,
                    "training_time":elapsed_time,
                    "optimizer": opt.state_dict(),
                    "model": model.state_dict(),
                    "estimator" : estimator.state_dict(),
                    "scheduler" : scheduler.state_dict(),
                    "name" : cfg.model_name+str(e)+"_aabbxyz_"+str(side_x)+"_"+str(side_y)+"_"+str(side_z)+"_"+date_string,
                }

                save_checkpoint(checkpoint,cfg.save_ckp_path+ f"{checkpoint['name']}")
        end_time = time.time()
        logging.info(f"Model trained successfully ! it took {end_time-start_time}")
        avg_loss = sum(losses)/(len(losses))
        avg_psnr = sum(psnrs)/(len(psnrs))
       
        print(
            f"Training Time  = {end_time-start_time} | "
            f"loss={avg_loss:.5f} | psnr={avg_psnr:.2f} "
        )           
    if(cfg.run_test):
        print(f"Start testing phase: ")
        start_test=time.time()
        model.eval()
        estimator.eval()
        with torch.no_grad():
            for i in range(len(test_dataset)):
                start=time.time()
                data = test_dataset[i]
                rays = data["rays"]
                if(not cfg.gen_poses):
                    pixels = data["pixels"] 
                h=data["height"]
                w=data["width"]
                render_bkgd= data["color_bkgd"]
                rgb,acc,depth, n_rendering_samples = occ_grid.render_image_with_occgrid(
                    model,
                    estimator,
                    rays,
                    #rendering_optins
                    render_step_size=render_step_size,
                    render_bkgd=render_bkgd
                    #test options
                )
                if i % cfg.disp_every==0: 
                    end=time.time()
                    elapsed_time=end-start
                    if(not cfg.gen_poses):
                        loss = torch.nn.functional.mse_loss(rgb, pixels)
                        psnr = -10. * torch.log10(loss)
                        lpips = lpips_metric(rgb, pixels).item()
                        #ssim = ssim_metric(rgb, pixels)
                        writer.add_scalar('Loss/test', loss, i)
                        writer.add_scalar('Psnr/test', psnr, i)
                        writer.add_scalar('lpips/test', lpips, i)
                        #writer.add_scalar('ssim/test', ssim, i)

                        print(
                            f"elapsed_time={elapsed_time}s | step={i} | "
                            f"loss={loss:.5f} | psnr={psnr:.2f} | lpips={lpips:.2f} " #|ssim={ssim:.2f} |
                            f"n_rendering_samples={n_rendering_samples:d} | num_rays={len(pixels):d} | "
                            f"max_depth={depth.max():.3f} | "
                        )
                        
                    else :
                        print(
                            f"gen pose n° {i} |"
                            f"elapsed_time={elapsed_time}s | step={i} | "
                            f"n_rendering_samples={n_rendering_samples:d} | "
                            f"max_depth={depth.max():.3f} | "
                        )
                    rgb = torch.reshape(rgb, (h, w, 3))
                    imageio.imwrite(
                                    cfg.preds_folder+f"rgb_{i}.jpg",
                                    (rgb.cpu().numpy() * 255).astype(np.uint8),
                                )
        end_test=time.time()
        logging.info(f"Model tested successfully ! it took {end_test-start_test}")
    if(cfg.gen_vid):
        generate_video("/kaggle/working/out_sai/","out")
