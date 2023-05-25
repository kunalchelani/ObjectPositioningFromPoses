from hloc import extract_features, match_features
from hloc.utils.read_write_model import *
import open3d as o3d

# Feature Configs
superpoint_inloc_feature_conf  = extract_features.confs['superpoint_inloc']
superpoint_inloc_feature_conf["name"] = 'superpoint_inloc'

superpoint_aachen_feature_conf = extract_features.confs['superpoint_aachen']
superpoint_aachen_feature_conf["name"] = 'superpoint_aachen'

sift_feature_conf = extract_features.confs['sift']
r2d2_feature_conf = extract_features.confs['r2d2']
local_feature_confs = {'superpoint_inloc' : superpoint_inloc_feature_conf, 'superpoint_aachen': superpoint_aachen_feature_conf, 'sift' : sift_feature_conf, 'r2d2' : r2d2_feature_conf}

global_feature_conf = extract_features.confs['netvlad']

# Matcher configs 
superglue_conf = match_features.confs['superglue']
nn_mutual_conf = match_features.confs['NN-mutual']
nn_ratio_conf = match_features.confs['NN-ratio']

matcher_confs = {'superpoint_inloc' : superglue_conf, 'superpoint_aachen' : superglue_conf, 'r2d2': nn_mutual_conf, 'sift' : nn_ratio_conf}
global_feature_name = global_feature_conf['model']['name']

# Defining some utility funcitons

def get_intrisics_dict_from_colmap_model(colmap_model_path):
    images_path_txt = colmap_model_path / "images.txt"
    cameras_path_txt = colmap_model_path / "cameras.txt"

    cameras = read_cameras_text(cameras_path_txt)
    images = read_images_text(images_path_txt)

    intrisics_dict = {}

    for im_id in images.keys():
        cam = cameras[images[im_id].camera_id]
        intrisics_dict[images[im_id].name] = cam
    
    return intrisics_dict


def write_query_intrinsics_file(intrinsics_dict, 
                                    query_fname_with_intrinsics,
                                    query_fnames=None):

    if query_fnames is None:
        query_fnames = sorted(list(intrinsics_dict.keys())) 
    
    f_intrinsics = open(query_fname_with_intrinsics, 'w')
    key_def = list(intrinsics_dict.keys())[0]

    for fname in query_fnames:
        if fname in intrinsics_dict.keys():
            cam_params = intrinsics_dict[fname]
        else: 
            cam_params = intrinsics_dict[key_def]
    
        cam_params_str = " {} {} {} ".format(cam_params.model, cam_params.width, cam_params.height)
        cam_params_str += ' '.join([str(p) for p in cam_params.params])

        f_intrinsics.write(fname +  cam_params_str + "\n")
    
    f_intrinsics.close()
    return query_fnames


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