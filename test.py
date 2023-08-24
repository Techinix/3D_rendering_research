import numpy as np

def viewmatrix(z, up, pos):
        """Construct lookat view matrix."""
        vec2 = normalize(z)
        vec1_avg = up

        vec0 = normalize(np.cross(vec1_avg, vec2))
        vec1 = normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, pos], 1)
        return m

def poses_avg(poses):
        """Average poses according to the original NeRF code."""
        hwf = poses[0, :3, -1:]
        center = poses[:, :3, 3].mean(0)
        vec2 = normalize(poses[:, :3, 2].sum(0))
        up = poses[:, :3, 1].sum(0)

        c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
        return c2w

def normalize(x):
        """Normalization helper function."""
        return x / np.linalg.norm(x)
def recenter_poses(poses):
        """Recenter poses according to the original NeRF code."""
        poses_ = poses.copy()
        bottom = np.reshape([0, 0, 0, 1.], [1, 4])
        c2w = poses_avg(poses)
        c2w = np.concatenate([c2w[:3, :4], bottom], -2)
        bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
        poses = np.concatenate([poses[:, :3, :4], bottom], -2)
        poses = np.linalg.inv(c2w) @ poses
        poses_[:, :3, :4] = poses[:, :3, :4]
        poses = poses_
        return poses
def spherify_poses(poses):
        p34_to_44 = lambda p: np.concatenate([
            p,
            np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])
        ], 1)
        rays_d = poses[:, :3, 2:3]
        rays_o = poses[:, :3, 3:4]

        def min_line_dist(rays_o, rays_d):
            a_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
            b_i = -a_i @ rays_o
            pt_mindist = np.squeeze(-np.linalg.inv(
                (np.transpose(a_i, [0, 2, 1]) @ a_i).mean(0)) @ (b_i).mean(0))
            return pt_mindist

        pt_mindist = min_line_dist(rays_o, rays_d)
        center = pt_mindist
        up = (poses[:, :3, 3] - center).mean(0)
        vec0 = normalize(up)
        vec1 = normalize(np.cross([.1, .2, .3], vec0))
        vec2 = normalize(np.cross(vec0, vec1))
        pos = center
        c2w = np.stack([vec1, vec2, vec0, pos], 1)
        poses_reset = (
                np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4]))
        poses_reset = np.concatenate([
            poses_reset[:, :3, :4],
            np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)
        ], -1)
        return poses_reset
def load_colmap_data(c2w,h,w,f):
    
   
    hwf = np.array([h,w,f]).reshape([3,1])
    poses = c2w[:, :3, :4].transpose([1,2,0])
    
    poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1,1,poses.shape[-1]])], 1)
    # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)
    print("inside: ",poses)
    poses=poses.reshape([1,15])
    poses=np.concatenate([poses,np.array([[1,2]]),],axis=1)
    return poses


"""

print(poses_arr)
poses = load_colmap_data(poses_arr,800,800,1104.3)

poses = poses[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
print("before: ",poses)
poses[:2, 4, :] = np.array((800,800)).reshape([2, 1])
print("after: ",poses)
poses = np.concatenate(
            [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)

poses = np.moveaxis(poses, -1, 0).astype(np.float32)

poses = recenter_poses(poses)
#poses = spherify_poses(poses)

print("end: ",poses,poses.shape)"""


# Parameters:
# - points2D: Nx2 array; pixel coordinates
# - points3D: Nx3 array; world coordinates
# - camera: pycolmap.Camera
# Optional parameters:
# - max_error_px: float; RANSAC inlier threshold in pixels (default=12.0)
# - estimation_options: dict or pycolmap.AbsolutePoseEstimationOptions
# - refinement_options: dict or pycolmap.AbsolutePoseRefinementOptions
import numpy as np


c2w = np.array([
        [
          -0.6049851196617858,
          -0.5817819971632365,
          0.5436199891970724,
          2.168909148688849
        ],
        [
          0.796147277424188,
          -0.43175588620522015,
          0.423953237027713,
          1.7053538531609098
        ],
        [
          -0.011937231502134346,
          0.6892869880774034,
          0.7243900542121131,
          2.897055566534161
        ],
        [
          0.0,
          0.0,
          0.0,
          1.0
        ]
      ])

c2w_mats = np.array([
        [
          -0.6049851196617858,
          -0.5817819971632365,
          0.5436199891970724,
          2.168909148688849
        ],
        [
          0.796147277424188,
          -0.43175588620522015,
          0.423953237027713,
          1.7053538531609098
        ],
        [
          -0.011937231502134346,
          0.6892869880774034,
          0.7243900542121131,
          2.897055566534161
        ],
        [
          0.0,
          0.0,
          0.0,
          1.0
        ]
      ])
#pycolmap
close_depth=0
inf_depth=10
save_arr=[]
hwf = np.array([800,800,140]).reshape([3,1])
poses = c2w_mats[:3, :4].transpose([0,1])
poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1,1,poses.shape[-1]])], 1)
print(poses.shape)
#print(poses)

#poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)
print(poses)
save_arr.append(np.concatenate([poses.ravel(), np.array([close_depth, inf_depth])], 0))
save_arr = np.array(save_arr)
h ,w , f = save_arr[0][4],save_arr[0][9],save_arr[0][14]
c2w_pycolmap = np.concatenate([[save_arr[0][0:4]],[save_arr[0][5:9]],[save_arr[0][10:14]]],axis=0)
c2w_pycolmap=np.concatenate([c2w_pycolmap, np.array([[0,0,0,1]])], 0)



#colmap
c2w[0:3, 2] *= -1 # flip the y and z axis
c2w[0:3, 1] *= -1
c2w = c2w[[1, 0, 2, 3],:] # swap y and z
c2w[2, :] *= -1 # flip whole world upside down
    
print("pycolmap output is: ",c2w_pycolmap)
print("colmap output is: ",c2w)
