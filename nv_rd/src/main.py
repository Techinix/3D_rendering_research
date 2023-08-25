import argparse
import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:     %(message)s'
    )
def run_approach(choice,data,use_npz):
    if choice == 1 :
        from .classic_nerf.run_classic_nerf import ClassicNerf
        approach = ClassicNerf(data=data,use_npz=use_npz)
    elif choice == 2 :
        from .instant_ngp.run_instant_ngp_occ import InstantNgpOcc
        approach = InstantNgpOcc(data=data,use_npz=use_npz)
    elif choice == 3 :
        from .instant_ngp.run_instant_ngp_prop import InstantNgpProp
        approach = InstantNgpProp(data=data,use_npz=use_npz)
    elif choice == 4 :
        from .mip_nerf.run_mip_nerf import MipNerf
        approach = MipNerf(data=data,use_npz=use_npz)
        

    approach.train()

parser = argparse.ArgumentParser()
parser.add_argument("--choice", help="choose which approach to go with ", type=int, required=True,choices=[1,2,3,4])
parser.add_argument("--data", help="input data path", type=str, required=True)
parser.add_argument("--use_npz",action="store_true",help="input npz data path")




if __name__=="__main__":
    hparams = parser.parse_args()
    logging.info(f"path u specified is {hparams.data} while current working dir is {os.getcwd()}")
    if(not os.path.exists(hparams.data)):
        logging.info("Please specify a correct data path(npz) ...")
        sys.exit()
    
    run_approach(choice=hparams.choice,data=hparams.data,use_npz=hparams.use_npz)
