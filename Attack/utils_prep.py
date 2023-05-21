from hloc import extract_features, match_features, pairs_from_poses, pairs_from_covisibility, triangulation
from hloc.utils.read_write_model import *
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