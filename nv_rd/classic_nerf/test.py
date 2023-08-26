import argparse
import configs.config as cfg

def Argsparser():
    parser=argparse.ArgumentParser(prog="Execution of Nerf Benchmarking",
                                   description="An implementation of the Tiny Neural Radiance Field Nerf Approach\
                                     to estimate 3D shapes from 2D images",
                                    epilog="for further question, contact X@gmail.com",
                                   )
    
    parser.add_argument("--config", help="Path to config file.", required=False, default='./configs/config.yaml')
    parser.add_argument("opts", nargs=argparse.REMAINDER,
                        help="Modify hparams. Example: train.py resume out_dir TRAIN.BATCH_SIZE 2")

    return parser



hparams = cfg.parse_args(Argsparser())


print(hparams,hparams.run_testing)
