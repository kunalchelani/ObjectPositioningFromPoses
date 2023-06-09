import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import hloc.utils.read_write_model as rwm
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pdb
import os
from typing import Union, Tuple, List

def get_Rt_from_colmap_images_file(colmap_images_file: Path, 
                                   inlier_names: list = None):
    
    if ".bin" in str(colmap_images_file):
        images = rwm.read_images_binary(colmap_images_file)
    else:
        images = rwm.read_images_text(colmap_images_file)
    
    Rs = {}
    ts = {}
    
    for id in images:
        imname = images[id].name
        qvec = images[id].qvec
        tvec = images[id].tvec
        qvec_n = [qvec[1], qvec[2], qvec[3], qvec[0]]
        Rs[imname] = R.from_quat(qvec_n).as_matrix()
        ts[imname] = tvec

    if inlier_names is not None:
        Rs = {name:Rs[name] for name in inlier_names if name in Rs.keys()}
        ts = {name:ts[name] for name in inlier_names if name in ts.keys()}
    
    return Rs, ts

def get_Rt_from_hloc_poses_file(hloc_poses_fname: Path,
                                inlier_names: list = None):
    
    Rs = {}
    ts = {}
    f = open(hloc_poses_fname, 'r')
    
    for line in f: 
        imname, *data = line.strip().split(' ')
        qvec = np.array(data[:4], dtype = np.float32)
        tvec = np.array(data[4:], dtype = np.float32)
        qvec_n = [qvec[1], qvec[2], qvec[3], qvec[0]]
        Rs[imname] = R.from_quat(qvec_n).as_matrix()
        ts[imname] = tvec

    if inlier_names is not None:
        Rs = {name:Rs[name] for name in inlier_names if name in Rs.keys()}
        ts = {name:ts[name] for name in inlier_names if name in ts.keys()}

    return Rs, ts

def single_cam_alignment(R_s, t_s, R_c, t_c, scale):

    R_est = R.from_matrix(R_s).inv() * R.from_matrix(R_c)
    t_est = R.from_matrix(R_s).inv().apply(scale * t_c - t_s)

    return R_est.as_matrix(), t_est

def get_rot_residual(R_est, R_s, R_c):
    
    return (R.from_matrix(R_c) * R.from_matrix(R_est).inv() * R.from_matrix(R_s).inv()).magnitude() * 180/(np.pi)


def get_cam_dist_residual(R_est, t_est, scale, R_s, t_s, R_c, t_c):
    
    r_n = R.from_matrix(R_c) * R.from_matrix(R_est).inv()
    t_n = scale * t_c - (R.from_matrix(R_c) * R.from_matrix(R_est).inv()).apply(t_est)
    cc_diff_vector = np.linalg.norm(r_n.inv().apply(t_n) - R.from_matrix(R_s).inv().apply(t_s))
    return cc_diff_vector


def exp(w):
    '''
    Input
    w: np.array([3,])
    '''
    theta = np.linalg.norm(w)
    v = w/theta
    [v1, v2, v3] = [v[0], v[1], v[2]]
    v_cross = np.array([[0, -v3, v2], [v3, 0, -v1], [-v2,  v1, 0]]).reshape([3,3])
    v_cross_sq = np.array([[-1*(v2**2 + v3**2), v1*v2, v1*v3], [v1*v2, -1*(v1**2 + v3**2), v2*v3], [v1*v3,  (v2*v3), -1*(v1**2 + v2**2)]]).reshape([3,3])
    
    return R.from_matrix(np.eye(3) + np.sin(theta)*v_cross + (1-np.cos(theta))*v_cross_sq)

def log(Rot):
    '''
    Input
    Rot: scipy.spatial.transform.Rotation
    '''
    
    Rmat = R.as_matrix(Rot)
    Y = 0.5*(Rmat - Rmat.transpose())
    y = np.array([Y[2,1],Y[0,2],Y[1,0]])
    y_norm = np.linalg.norm(y)
    if y_norm == 0:
        return np.zeros([3,])
    else:
        return np.arcsin(y_norm) * y/y_norm

def avg_SO3(Rots, epsilon, max_num_iters = 1000):
    '''
    Input
    Rots: list[scipy.spatial.transform.Rotation] - list of rotations to average
    epsilon: float - convergence threshold
    max_num_iters: int - maximum number of iterations
    
    Output
    R_est: scipy.spatial.transform.Rotation
    '''
    
    iter = 0
    R_est = Rots[0]
    
    while iter < max_num_iters:
        iter += 1
        w = np.sum(np.array([log(R_est.inv() * Rots[i]) for i in range(len(Rots))]), axis = 0) / (len(Rots))
        print(w)
        if np.linalg.norm(w) < epsilon:
            print("Iteration : {} | w_norm : {}".format(iter, np.linalg.norm(w)))
            return R_est
        else:
            R_est = R_est * exp(w)
        print("Iteration : {} | w_norm : {}".format(iter, np.linalg.norm(w)))
    return R_est


def align_using_best_single_cam(poses_set1, 
                                poses_set2, 
                                scale, 
                                rot_thresh,
                                cc_thresh,
                                query_imnames = None,
                                ):
    
    Rs_1, ts_1 = poses_set1
    Rs_2, ts_2 = poses_set2 

    best_inliers = []
    max_inliers = 0
    
    if query_imnames is not None:
        keys = query_imnames
    else:
        keys = list(set(Rs_1.keys()).intersection(set(Rs_2.keys())))
    

    for key in tqdm(keys):
        R_est, t_est = single_cam_alignment(Rs_1[key], ts_1[key], Rs_2[key], ts_2[key], scale)
        r_res = {key:get_rot_residual(R_est, Rs_1[key], Rs_2[key]) for key in keys}
        cc_res = {key:get_cam_dist_residual(R_est, t_est, scale, Rs_1[key], ts_1[key], Rs_2[key], ts_2[key]) for key in keys}
        
        inliers = [key for key in keys if ((r_res[key] < rot_thresh) and (cc_res[key] < cc_thresh))]
        
        if len(inliers) > max_inliers:
            best_inliers = inliers
            best_ests = R_est, t_est
            max_inliers = len(inliers)
    
    return best_ests, best_inliers


def align_poses(attack_poses_path: Union[str, Path],
                local_poses_path: Union[str, Path],
                rot_thresh: float,
                cc_thresh: float,
                scale: float,
                average_inliers: bool = True,
                query_imnames_dir: Union[str, Path] = None,
                inliers_save_path: Union[str, Path] = None,
                ):
    '''
    Align the set of poses using the best single camera alignment and 
    optionally averaging the inliers from the best alignment
    '''
    
    if isinstance(attack_poses_path, str):
        attack_poses_path = Path(attack_poses_path)
    if isinstance(local_poses_path, str):
        local_poses_path = Path(local_poses_path)
    
    assert attack_poses_path.exists(), f"attack_poses_path {attack_poses_path} does not exist"
    assert local_poses_path.exists(), f"local_poses_path {local_poses_path} does not exist"
    
    # Get the two sets of poses - local ones and from the server
    Rs_attack, ts_attack = get_Rt_from_hloc_poses_file(attack_poses_path)
    Rs_client, ts_client = get_Rt_from_colmap_images_file(local_poses_path)

    # If a specific list of qury images are to be used for alignment, get them
    if query_imnames_dir is not None:
        query_imnames = list(os.listdir(query_imnames_dir))
    else:
        query_imnames = None

    # Align the two sets of poses
    [r,t], inliers = align_using_best_single_cam(poses_set1 = [Rs_attack, ts_attack],
                                                poses_set2 =  [Rs_client, ts_client],
                                                scale = scale,
                                                rot_thresh = rot_thresh, 
                                                cc_thresh = cc_thresh,
                                                query_imnames = query_imnames,
                                                )

    
    R_final = R.from_matrix(r).as_quat()
    t_final = t.reshape(-1, 3)
    ## Average transformation calculation
    
    if average_inliers:
        Rf_ins = []
        t_averaged = np.zeros(3,)
        for i in inliers:
            Ri, ti = single_cam_alignment(Rs_attack[i], ts_attack[i], Rs_client[i], ts_client[i], scale)
            Rf_ins.append(R.from_matrix(Ri))
            t_averaged += ti.reshape(3,)

        R_averaged = avg_SO3(Rf_ins, epsilon=0.001, max_num_iters=100)
        t_averaged /= len(inliers)

        R_final = R_averaged.as_quat()
        t_final = t_averaged.reshape(-1, 3)
    
    # Save inliers if needed
    if inliers_save_path is not None:
        with open(inliers_save_path, 'w') as f:
            for item in inliers:
                f.write("%s\n" % item)
    
    return R_final, t_final 
    

def position_object(object_model_path: Union[str, Path],
                    transformed_object_model_path: Union[str, Path],
                    scale: float,
                    rotation: Union[np.array, R],
                    translation: np.array,
                    ):
    """
    Position the object in the scene using the rotation and translation
    """
    
    # Load the object
    
    assert object_model_path.exists(), f"Object model path {object_model_path} does not exist"
    
    if isinstance(rotation, np.ndarray):
        rotation = R.from_quat(rotation)
    
    pcd_orig = o3d.io.read_point_cloud(str(object_model_path))
    pts = np.asarray(pcd_orig.points)
    
    pts_transformed = scale * rotation.apply(pts) +  translation.reshape(-1, 3)
    pcd_transformed = o3d.geometry.PointCloud()
    pcd_transformed.points = o3d.utility.Vector3dVector(pts_transformed)
    pcd_transformed.colors = pcd_orig.colors
        
    print(f"Saving transformed ply file to {transformed_object_model_path}")
    o3d.io.write_point_cloud(str(transformed_object_model_path), pcd_transformed)

def transform_colmap_poses(colmap_images_file, transformed_poses_file, s,r, t):

    images = rwm.read_images_text(colmap_images_file)
    images_new = {}
        

    for id in images:
        imid = images[id].id
        name = images[id].name
        point3D_ids = images[id].point3D_ids
        tvec = images[id].tvec
        xys = images[id].xys
        camera_id = images[id].camera_id

        qvec_old = images[id].qvec
        qvec_old_n = np.array([qvec_old[1],qvec_old[2],qvec_old[3],qvec_old[0]], dtype = np.float32)
        r_old= R.from_quat(qvec_old_n)
        qvec = (r_old * r.inv()).as_quat()
        qvec = np.array([qvec[3], qvec[0], qvec[1], qvec[2]], dtype=np.float32)
        
        tvec_old = images[id].tvec
        tvec = s * tvec_old - (r_old * r.inv()).apply(t)

        images_new[id] = rwm.Image(id = imid, name = name, point3D_ids = point3D_ids,
                        camera_id = camera_id, tvec = tvec.reshape(3,), xys = xys, qvec = qvec)
    
    rwm.write_images_text(images_new, transformed_poses_file)