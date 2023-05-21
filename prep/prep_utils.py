import os
import numpy as np

from pathlib import Path

from typing import Union, Optional, Dict

from hloc.utils.base_model import dynamic_load
from hloc.utils.parsers import names_to_pair, parse_retrieval
from hloc.utils.io import list_h5_names
from hloc.utils.read_write_model import *
from hloc import triangulation
import matplotlib.pyplot as plt
from scipy.spatial.transform.rotation import Rotation as R
from hloc import extract_features, match_features, pairs_from_covisibility, pairs_from_retrieval, pairs_from_poses
import open3d as o3d

# Feature Configs
superpoint_feature_conf  = extract_features.confs['superpoint_inloc']
sift_feature_conf = extract_features.confs['sift']
r2d2_feature_conf = extract_features.confs['r2d2']
local_feature_confs = {'superpoint' : superpoint_feature_conf , 'sift' : sift_feature_conf, 'r2d2' : r2d2_feature_conf}

global_feature_conf = extract_features.confs['netvlad']

# Matcher configs 
superglue_conf = match_features.confs['superglue']
nn_mutual_conf = match_features.confs['NN-mutual']
nn_ratio_conf = match_features.confs['NN-ratio']

matcher_confs = {'superpoint' : superglue_conf, 'r2d2': nn_mutual_conf, 'sift' : nn_ratio_conf}
global_feature_name = global_feature_conf['model']['name']


import os
import subprocess

def build_colmap_model(images_path, recons_out_path, image_list = None):

    if not os.path.isdir(recons_out_path):
        os.makedirs(recons_out_path, exist_ok=True)
        
    db_path = recons_out_path + "/database.db"

    db_create_command = "colmap database_creator --database_path {}".format(db_path)
    model_convert_command = "colmap model_converter --input_path {} --output_path {} --output_type txt".format(recons_out_path+"/0", recons_out_path+"/0")

    if image_list is not None:
        feat_extract_command = "colmap feature_extractor --database_path {} --image_path {} --image_list_path {} --SiftExtraction.use_gpu 1 --SiftExtraction.gpu_index -1".format(db_path, images_path, image_list)
        exhaustive_match_command = "colmap exhaustive_matcher --database_path {} --SiftMatching.use_gpu 1 --SiftMatching.gpu_index -1".format(db_path)
        map_command = "colmap mapper --database_path {}  --output_path {} --image_path {} --image_list_path {} --Mapper.ba_global_max_refinements 5 --Mapper.ba_global_max_num_iterations 70".format(db_path, recons_out_path, images_path, image_list)
        
    else: 
        feat_extract_command = "colmap feature_extractor --database_path {} --image_path {} --SiftExtraction.use_gpu 1 --SiftExtraction.gpu_index -1".format(db_path, images_path)
        exhaustive_match_command = "colmap exhaustive_matcher --database_path {} --SiftMatching.use_gpu 1 --SiftMatching.gpu_index -1".format(db_path)
        map_command = "colmap mapper --database_path {}  --output_path {} --image_path {} --Mapper.ba_global_max_refinements 5 --Mapper.ba_global_max_num_iterations 70".format(db_path, recons_out_path, images_path)

    print("Creating database")
    subprocess.run(db_create_command.split(" "), stderr=subprocess.STDOUT)
    print("Done !")

    print("Running feature extraction")
    subprocess.run(feat_extract_command.split(" "), stderr=subprocess.STDOUT)
    print("Done !")

    print("Running exhaustive matching")
    subprocess.run(exhaustive_match_command.split(" " ), stderr=subprocess.STDOUT)
    print("Done !")

    print("Running mapping")
    subprocess.run(map_command.split(" "), stderr=subprocess.STDOUT)
    print("Done !")

    print("Running model conversion")
    subprocess.run(model_convert_command.split(" "), stderr=subprocess.STDOUT)
    print("Done !")

def export_colmap_pts_as_ply(colmap_pts_path, out_ply_path, pct_pts = 100):
 
    print("Reading colmap points")
    if ".txt" in colmap_pts_path:
        pts = read_points3D_text(colmap_pts_path)
    elif "bin" in colmap_pts_path:
        pts = read_points3D_binary(colmap_pts_path)

    print("Done!")
    pts_xyz = []
    pts_rgb = []
    for pt in pts:
        pts_xyz.append(pts[pt].xyz)
        pts_rgb.append(pts[pt].rgb)

    num_pts_total = len(pts_xyz)
    use_inds = np.random.choice(range(num_pts_total), size = int(pct_pts/100 * num_pts_total), replace = False)

    pts_xyz = np.vstack(pts_xyz)
    pts_rgb = np.vstack(pts_rgb)

    pts_xyz = pts_xyz[use_inds, :]
    pts_rgb = pts_rgb[use_inds, :]

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts_xyz)
    pc.colors = o3d.utility.Vector3dVector(pts_rgb/255.0)

    print("Writing points to ply")
    o3d.io.write_point_cloud(out_ply_path, pc)
    print("Done!")


def interpolate_pose_at_ts(query_ts, traj):
    # traj can be a dict with key as floating point ts, value as rot, trans 6-vec
    
    traj_ts_sorted = sorted(list(traj.keys()))
    # print(traj_ts_sorted)
    # print(query_ts)
    query_index = np.searchsorted(np.array(traj_ts_sorted), query_ts)
    
    if query_index == 0:
        return traj[traj_ts_sorted[0]]

    if query_index == len(traj_ts_sorted):
        return traj[traj_ts_sorted[-1]]
    
    else:
        dist_low = abs(traj_ts_sorted[query_index - 1] - query_ts)
        dist_high = abs(traj_ts_sorted[query_index] - query_ts)
        
        a = dist_low / (dist_low + dist_high)
        
        return (1-a) * (traj[traj_ts_sorted[query_index - 1]]) +  a * (traj[traj_ts_sorted[query_index]])


# This funciton starts from a colmap model and a feature, and uses hloc to build a model using the corresponding feature without changing the camera poses.

def perform_hloc_sfm(images_dir:  Path, 
                    base_dir_sfm: Path,
                    poses_path: Path,
                    feature: str,
                    camera_intrinsics_params: np.ndarray,
                    colmap_camera_model: str,
                    num_matched_pairs_covis: int = 30,
                    force_extract_local: bool = False,
                    force_extract_global: bool = False,
                    force_extract_pairs_from_covis: bool = False,
                    force_match_features: bool = False,
                    perform_server_sfm: bool=False,
                    export_ply: bool = False,
                    ) -> None :

    # Declare feature and matcher configs
    local_feature_conf = local_feature_confs[feature]
    matcher_conf = matcher_confs[feature]

    local_feature_name = local_feature_conf['model']['name']
    matcher_name = matcher_conf['model']['name']

    # Make dir if not exists
    if not base_dir_sfm.is_dir():
        os.makedirs(base_dir_sfm, exist_ok=True)

    local_feature_dir = base_dir_sfm / f"{local_feature_name}/"
    global_feature_dir = base_dir_sfm / f"{global_feature_name}/"

    local_feature_path = local_feature_dir / f"{local_feature_conf['output']}.h5"
    global_feature_path = global_feature_dir / f"{global_feature_conf['output']}.h5"

    pairs_path = local_feature_dir / f"pairs_covisibility_{num_matched_pairs_covis}.txt"
    matches_path = local_feature_dir / f"matches_{matcher_name}.h5"

    sfm_dir = local_feature_dir / "sfm" 

    # -------------------------------- Server sfm --------------------------------- #
    
    # Skip if already done and not forced

    if not local_feature_path.exists() or force_extract_local:
        local_feature_path = extract_features.main(local_feature_conf, images_dir, feature_path = local_feature_path)
    else:
        print("Skipping local feature extraction")

    if not global_feature_path.exists() or force_extract_global:
        global_feature_path = extract_features.main(global_feature_conf, images_dir, feature_path = global_feature_path)
    else:
        print("Skipping global feature extraction")

    if not pairs_path.exists() or force_extract_pairs_from_covis:
        pairs_from_poses.get_pairs_from_poses_txt_file(poses_path, pairs_path, num_matched_pairs_covis)
    else:
        print("Skipping covis pair finding")

    if not matches_path.exists() or force_match_features:
        matches_path = match_features.main(matcher_conf, pairs_path, local_feature_path, matches = matches_path)
    else:
        print("Skipping matching")

    if perform_server_sfm:
        if sfm_dir.is_dir():
            os.remove(sfm_dir / "cameras.*")
            os.remove(sfm_dir / "points3D.*")
            os.remove(sfm_dir / "images.*")
            os.remove(sfm_dir / "database.*")
            os.remove(sfm_dir / "sparse.*")
            
        print("--------------- Started Triangulation ------------------")
        triangulation.triangulate_using_poses_and_intrinsics(images_dir = images_dir,
                                            sfm_dir = sfm_dir, 
                                            poses_path = poses_path, 
                                            pairs_path = pairs_path, 
                                            local_features_path = local_feature_path, 
                                            matches_path = matches_path, 
                                            camera_intrinsics_params = camera_intrinsics_params, 
                                            colmap_camera_model = colmap_camera_model,
                                            colmap_path='colmap', 
                                            skip_geometric_verification=False, 
                                            min_match_score=None, 
                                            is_poses_dir=False)
    else:
        print("Skipping sfm")

    # Export ply file if needed

    if export_ply:
        colmap_pts_path = sfm_dir / "points3D.bin"
        out_ply_path = sfm_dir / "sparse.ply"
        export_colmap_pts_as_ply(str(colmap_pts_path), str(out_ply_path))