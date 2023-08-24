import torch
import numpy as np
import sys
from sai_tools import S3, MongoDB
import os
import traceback
import imageio
from skimage.transform import resize
from PIL import Image
from io import BytesIO
import logging

from train import train_loop
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:     %(message)s'
    )

class Trainer_Config():
    def __init__(self,
                 cfg,
                 config,
                 region_name="us-east-2", 
                 access_key=None, 
                 secret_key=None, 
                 height=None, 
                 width=None, 
                 focal_length=None, 
                 bucket=None,
                 data_folder=None, 
                 model=None,
                 optimizer=None,
                 ) -> None:
        self.cfg = cfg
        self.config = config
        self.height = height
        self.width = width
        self.focal_length = focal_length
        self.near_thresh = cfg.near_thresh
        self.num_encoding_functions = cfg.n_enc
        self.far_thresh = cfg.far_thresh
        self.Nc = cfg.Nc
        self.Nf= cfg.Nf
        self.bucket = bucket
        self.data_folder = data_folder
        self.region_name = region_name
        self.access_key = access_key
        self.secret_key = secret_key
        self.chunksize = cfg.chunk_size
        self.lr = cfg.lr
        self.num_iters = cfg.max_steps
        self.display_every = cfg.disp_every
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cached_s3 = None

        if self.access_key and self.secret_key:
            self.cached_s3 = S3(access_key=self.access_key, secret_key=self.secret_key)
        
        if self.data_folder:
            if not os.path.exists(self.data_folder):
                # If folder doesn't exist, create it
                os.makedirs(self.data_folder)


    def create_npz(self, db, collection, user_id, car_id, *args, **kwargs):
        """Given user_id and car_id get all images and their corresponding poses and focal from MongoDB and S3, then create an npz file and push it to S3,
        the url for npz file in S3 is saved in MongoDB,
        Args:
        Returns:
        """
        dataset_path = "_".join([user_id, car_id])
        npz_name = "_".join([user_id, car_id, ".npz"])

        mdb = MongoDB()
        query = {"user_id": user_id, "car_id": car_id}
        doc = mdb.get_doc(db=db, collection=collection, query=query)
        
        if doc is None:
            logging.info(f"{db} database has no document that match the query {query}")
            sys.exit(1)

        bucket = doc["bucket"]
        focal = doc["focal"]
        img_list = doc["img_list"]
        all_img_arrays = []
        all_img_poses_arrays = []
        
        logging.info("Constructing numpy array for all images, poses and focal...")

        for img_dict in img_list:
            img_name = img_dict["img_path"].split("/")[-1]
            self.cached_s3.download_file(bucket=bucket, key="/".join([dataset_path, "datasets", img_name]), dest_filename="/".join([self.data_folder, img_name]))
            all_img_arrays.append(
                np.array(resize(imageio.imread("/".join([self.data_folder, img_name])), (self.height, self.width)), dtype=np.float32)
            )
            #################################################################################################
            # poses are hardcoded at the moment so we wont get the values from MongoDB
            # all_img_poses_arrays.append(np.array(img_dict["pose"], dtype=np.float32))
            hard_coded_poses = [[-9.3054223e-01, 1.1707554e-01, -3.4696460e-01, -1.3986591e+00],
                                [-3.6618456e-01, -2.9751042e-01, 8.8170075e-01, 3.5542498e+00],
                                [7.4505806e-09, 9.4751304e-01, 3.1971723e-01, 1.2888215e+00],
                                [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00]]
            
            np_array = np.array(hard_coded_poses, dtype=np.float32)
            all_img_poses_arrays.append(np_array, dtype=np.float32)
            #################################################################################################


        stacked_all_img_array = np.stack(all_img_arrays, axis=0)
        stacked_all_poses_array = np.stack(all_img_poses_arrays, axis=0)

        logging.info('Creating npz file...')

        vals_to_save = {
            "images": stacked_all_img_array,
            "poses": stacked_all_poses_array,
            "focal": focal
        }
        np.savez("/".join([self.data_folder, npz_name]), **vals_to_save)

        logging.info("Pushing npz to S3...")

        self.cached_s3.upload_file(src_filename="/".join([self.data_folder, npz_name]), bucket=self.bucket, dest_filename="/".join([dataset_path, "datasets", npz_name]))

        logging.info("Updating MongoDB...")
        filter = {"user_id": user_id, "car_id": car_id}
        mdb.update_doc(db=db, collection=collection, filter=filter, key="preprocessed", value=True)
        mdb.update_doc(db=db, collection=collection, filter=filter, key="npz_path", value="/".join([dataset_path, "datasets", npz_name]))


    def get_npz(self, user_id, car_id, *args, **kwargs):
        """
        """
        dataset_path = "_".join([user_id, car_id])
        npz_name = "_".join([user_id, car_id, ".npz"])
        # download dataset from S3 to specific path if not existing
        if not self.cached_s3:
            self.cached_s3 = S3(access_key=self.access_key, secret_key=self.secret_key)

        if not os.path.exists("/".join([self.data_folder, npz_name])):
            logging.info("Downloading npz from S3...")
            self.cached_s3.download_file(bucket=self.bucket, key="/".join([dataset_path, "datasets", npz_name]), dest_filename="/".join([self.data_folder, npz_name]))
        else:
            logging.info(f"File {npz_name} already downloaded !")

        # Load input images, poses, and intrinsics
        data = np.load("/".join([self.data_folder, npz_name]))
        return data


    def train(self,cfg,config,upload=False,db=None,collection=None, user_id=None, car_id=None,*args, **kwargs):
        """
        """
        if(upload):
            data = self.get_npz(user_id, car_id)
            model_name = "_".join([user_id, car_id,".pt"])
            model_path = "_".join([user_id, car_id])
            mdb = MongoDB()
        # train tiny_nerf model
        train_loop(
            device=self.device, 
            cfg=cfg,
            config=config, 
            model_name="base_model")

        if(upload):
            logging.info(f"Uploading {model_name} model to S3...")
            self.cached_s3.upload_file(src_filename=model_name, bucket=self.bucket, dest_filename="/".join([model_path, "models", model_name]))

            logging.info("Updating MongoDB...")
            filter = {"user_id": user_id, "car_id": car_id}
            mdb.update_doc(db=db, collection=collection, filter=filter, key="model_trained", value=True)
            mdb.update_doc(db=db, collection=collection, filter=filter, key="model_path", value="/".join([model_path, "models", model_name]))


    def load_model(self, user_id, car_id, *args, **kwargs):
        """Download weights for existing model from S3, load it into pytorch model and return the model
        Args:
        Returns:
            model: Loaded pytorch model"""
        model_path = "_".join([user_id, car_id])
        model_name = "_".join([user_id, car_id,".pt"])
        try:
            # check if models weights already downloaded from S3
            if not os.path.exists(model_name):
                if not self.cached_s3:
                    self.cached_s3 = S3(access_key=self.access_key, secret_key=self.secret_key)
                self.cached_s3.download_file(bucket=self.bucket, key="/".join([model_path, "models", model_name]), dest_filename="/".join([self.data_folder, model_name]))
            
            model = TinyNeRFModel(num_encoding_functions=self.num_encoding_functions)
            model.to(self.device)
            model.load_state_dict(torch.load("/".join([self.data_folder, model_name])))
            model.eval()
            return model
        except Exception:
            exp = traceback.format_exc()
            return exp


    def generate_poses(self, count=1):
        """Generate list of poses that will be used by inference to generate new images
        Args:
            count (int): How many pose matrix to generate
        Returns:
            poses (list): List of poses matrix
        """
        # TODO: generate new poses using intrinsic camera matrix and focal
        
        all_img_poses_arrays = []
        hard_coded_poses = [[-9.3054223e-01, 1.1707554e-01, -3.4696460e-01, -1.3986591e+00],
                            [-3.6618456e-01, -2.9751042e-01, 8.8170075e-01, 3.5542498e+00],
                            [7.4505806e-09, 9.4751304e-01, 3.1971723e-01, 1.2888215e+00],
                            [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00]]
        np_array = np.array(hard_coded_poses, dtype=np.float32)
        all_img_poses_arrays.append(np_array, dtype=np.float32)
        return all_img_poses_arrays


    def inference(self, model, poses=None, count=1, *args, **kwargs):
        """Predict object image from a specific pose
        Args:
            model: tiny NeRF model loaded
            poses (numpy.array): List of 4x4 numpy array that holds values for specific pose
            count (int): How many poses to generate if no poses are provided
        Returns:
            (list): List of poses and predicted images [poses, rgb_predicted]
        """
        # Specify encoding function.
        encode = lambda x: positional_encoding(x, num_encoding_functions=self.num_encoding_functions)

        # If no poses is provided generate new ones
        if poses is None:
            poses = self.generate_poses(count=count)

        rgb_predicted = []
        with torch.no_grad():
            for p in poses:
                p = torch.from_numpy(p).float().to(self.device)
                model_output = run_one_iter_of_tiny_nerf(
                        height=self.height,
                        width=self.width, 
                        focal_length=self.focal_length,
                        tform_cam2world=p, 
                        near_thresh=self.near_thresh,
                        far_thresh=self.far_thresh,
                        depth_samples_per_ray=self.depth_samples_per_ray,
                        encoding_function=encode,
                        get_minibatches_function=get_minibatches, 
                        model=model
                    ).detach().numpy()
                rgb_predicted.append(model_output)

        return [poses, rgb_predicted]
        

    def save_inference(self, user_id, car_id, count, inference_result, db, collection):
        """Update MongoDB and S3 with inference result
        Args:
            user_id (str): User ID
            car_id (str): Car ID
            count (int): How many inference output were generated 
            inference_result (list): First element is list of poses array, second element is list of images array
        Returns:
            bool: True is data is saved successfully in MongoDB and S3, otherwise False

        """
        mdb = MongoDB()

        poses = inference_result[0]
        rgb_predicted = inference_result[1]

        dataset_path = "_".join([user_id, car_id])

        # list that will hold refernce of pose and image link in S3, which will be save in MongoDB
        inference_results_mongodb = []

        for i in range(count):
            img_path = "/".join([dataset_path, "inference_results", f"infer_img_{i}.jpeg"])
            img_array = rgb_predicted[i]

            image = Image.fromarray(img_array.astype('uint8')).convert('RGB')
            buffer = BytesIO()
            image.save(buffer, format='JPEG')
            buffer.seek(0)

            logging.info(f"Uploading image {img_path} to {self.bucket} bucket...")
            response = self.cached_s3.put_object(bucket=self.bucket, key=img_path, body=buffer.getvalue(), contenttype='image/jpeg')

            if response is True:
                inference_results_mongodb.append(
                    {
                        "generated_pose": poses[i].tolist(),
                        "img_path": img_path
                    }
                )

        logging.info(f"Update MongoDB...")
        filter = {"user_id": user_id, "car_id": car_id}
        mdb.update_doc(db=db, collection=collection, filter=filter, key="inference_results", value=inference_results_mongodb)
        mdb.update_doc(db=db, collection=collection, filter=filter, key="inference_ready", value=True)

        return True
