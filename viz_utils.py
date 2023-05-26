import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import hloc.utils.read_write_model as rwm
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pdb
import os
from typing import Union
from Attack import Server, Client
from Attack import utils_attack


def get_cam_o3d(R, t, line_len = 0.1, color = np.array([1,0,0])):

    cam_corners = np.zeros([5,3])
    cam_corners[0, :] = np.array([-1*line_len, line_len, 3*line_len])
    cam_corners[1, :] = np.array([-1*line_len, -1*line_len, 3*line_len])
    cam_corners[2, :] = np.array([line_len, -1*line_len, 3*line_len])
    cam_corners[3, :] = np.array([line_len, line_len, 3*line_len])
    cam_corners[4, :] = np.array([0, 0, 0])

    world_pts = np.dot(R.transpose(), cam_corners.transpose() - t.reshape(-1,1)).transpose()
    lines = [[0,1], [1,2], [2,3], [3,0], [0,4], [1,4], [2,4], [3,4]]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(world_pts)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(np.tile(color, (len(lines),1)))

    return line_set


def get_cam_lines_from_poses(Rs, ts, cam_color, line_len = 0.1):

    cam_lines = []
    for R_curr, t_curr in zip(list(Rs.values()), list(ts.values())):
        cam = get_cam_o3d(R_curr, t_curr, line_len, cam_color)
        cam_lines.append(cam)

    return cam_lines

def visualize_attack(server_poses_path: Union[str, Path] = None,
                     local_poses_path: Union[str, Path] = None,
                     object_model_path: Union[str, Path] = None,
                     transformed_object_model_path: Union[str, Path] = None,
                     inliers_path: Union[str, Path] = None,
                     server_model_path: Union[str, Path] = None,
                     show_server_poses:bool = True,
                     highlight_inlier_poses:bool = True,
                     show_inlier_poses_only:bool = False,
                     show_transformed_local_poses:bool = False,
                     show_transformed_object:bool = False,
                     show_server_map:bool = True,
                     server_map_unicolor:bool = False,
                     object_model_unicolor:bool = False,
                     remove_cams_beyond = 15,
                     num_retrived_db_images = 30
                     ):

    geometries = []
    
    if show_server_poses:
        
        assert server_poses_path.exists(), "Server poses path does not exist"
        
        cam_color = np.array([1,0,0])
        cam_line_len = 0.1
        
        Rs, ts = utils_attack.get_Rt_from_hloc_poses_file(server_poses_path, inlier_names = None)
        cam_lines = get_cam_lines_from_poses(Rs, ts, cam_color, line_len = cam_line_len)
        
        # Skip cameras beyond a threshold distance from the origin
        if not show_inlier_poses_only:
            for cam_line in cam_lines:
                if np.linalg.norm(np.mean(np.asarray(cam_line.points), axis = 0)) < remove_cams_beyond:
                    geometries.append(cam_line)
        
        if highlight_inlier_poses or show_inlier_poses_only:

            assert inliers_path is not None, "Please provide inliers path"
            
            if isinstance(inliers_path, str):
                inliers_path = Path(inliers_path)

            assert inliers_path.exists(), "Inliers path does not exist"

            with open(inliers_path, "r") as f:
                inliers = f.readlines()
            
            Rs_attack_inliers = {i:Rs[i] for i in inliers}
            ts_attack_inliers = {i:ts[i] for i in inliers}
            
            # Get the indices of the inliers and select the corresponding cam lines
            cam_lines_in = get_cam_lines_from_poses(Rs_attack_inliers, ts_attack_inliers, np.array([0,1,0]))
            
            print(f"Number of inliers : {len(cam_lines_in)}")

            for cam_line in cam_lines_in:
                if np.linalg.norm(np.mean(np.asarray(cam_line.points), axis = 0)) < 20:
                    geometries.append(cam_line)
            
    if show_server_map:
        
        assert server_model_path is not None, "Please provide server model path"
        
        if isinstance(server_model_path, str):
            server_model_path = Path(server_model_path)

        assert server_model_path.exists(), "Server model path does not exist"

        if server_model_path.suffix == ".ply":
            server_map = o3d.io.read_point_cloud(server_model_path.as_posix())
        
        elif server_model_path.suffix == ".obj":
            server_map = o3d.io.read_triangle_mesh(server_model_path.as_posix(), True)
        
        else:
            raise NotImplementedError("Server model path should be either .ply or .obj")
        
        if server_map_unicolor:
                server_map.paint_uniform_color(np.array([1,0,0]))
                
        geometries.append(server_map)

    if show_transformed_object:
    
        assert transformed_object_model_path is not None, "Please provide transformed object model path"
        
        if isinstance(transformed_object_model_path, str):
            transformed_object_model_path = Path(transformed_object_model_path)

        assert transformed_object_model_path.exists(), "Transformed object model path does not exist"
            
        
        if transformed_object_model_path.suffix == ".ply":
            
            transformed_object = o3d.io.read_point_cloud(transformed_object_model_path.as_posix())
            
        elif transformed_object_model_path.suffix == ".obj":
            transformed_object = o3d.io.read_triangle_mesh(transformed_object_model_path.as_posix(), True)
        
        else:
            raise NotImplementedError("Transformed object model path should be either .ply or .obj")
        
        if object_model_unicolor:
                transformed_object.paint_uniform_color(np.array([0,1,0]))
    
        geometries.append(transformed_object)
            
    o3d.visualization.draw_geometries(geometries)