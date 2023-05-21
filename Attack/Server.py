# Author : Kunal Chlelani
# Email : kunalchelani@gmail.com

import os
import sys
from pathlib import Path
from typing import Union
from utils_prep import *

# Define the Class for a Server of a client-server based localization system

class Server:

    def __init__(self,
                 colmap_dir: Union(str, Path),
                 images_dir: Union(str, Path),
                 base_dir_db: Union(str, Path) = "../data/server/db",
                 base_dir_attack:Union(str, Path) = "../data/server/attack",
                 feature: str = "Superpoint_inloc",
                 num_matched_pairs_covis_db: int = 30,
                 num_matched_pairs_covis_localization: int = 30,
                 thresh_ransac_pnp: float = 12) -> None:
        
        if isinstance(colmap_dir, str):
            colmap_dir = Path(colmap_dir)
        self.colmap_dir = colmap_dir

        if isinstance(images_dir, str):
            images_dir = Path(images_dir)
        self.images_dir = images_dir

        if isinstance(base_dir_db, str):
            base_dir_db = Path(base_dir_db)
        self.base_dir_db = base_dir_db

        if isinstance(base_dir_attack, str):
            base_dir_attack = Path(base_dir_attack)
        self.base_dir_attack = base_dir_attack

        self.feature = feature
        
        self.num_matched_pairs_covis_db = num_matched_pairs_covis_db
        self.num_matched_pairs_covis_localization = num_matched_pairs_covis_localization
        self.thresh_ransac_pnp = thresh_ransac_pnp

        self.local_feature_conf = local_feature_confs[feature]
        self.matcher_conf = matcher_confs[feature]

        self.local_feature_name = self.local_feature_conf['model']['name']
        self.db_matcher_name = self.matcher_conf['model']['name']

        self.local_feature_dir = self.base_dir_sfm / f"{self.local_feature_name}/"
        self.global_feature_dir = self.base_dir_sfm / f"{self.global_feature_name}/"

        self.local_feature_path = self.local_feature_dir / f"{self.local_feature_conf['output']}.h5"
        self.global_feature_path = self.global_feature_dir / f"{global_feature_conf['output']}.h5"

        self.pairs_path = self.local_feature_dir / f"pairs_covisibility_{num_matched_pairs_covis_db}.txt"
        self.matches_path = self.local_feature_dir / f"matches_{self.db_matcher_name}.h5"

        self.db_sfm_dir = self.base_dir_db / f"{self.local_feature_name}/" / "sfm"

    def prep(self,
             poses_path: Union(str, Path),
             force_extract_local: bool = False,
             force_extract_global: bool = False,
             force_extract_pairs_from_covis:bool  = False,
             force_match_features: bool = False,
             perform_server_sfm: bool = False,
             export_ply: bool  = False) -> None:

        # Skip if already done and not forced

        if not self.local_feature_path.exists() or force_extract_local:
            local_feature_path = extract_features.main(self.local_feature_conf, self.images_dir, feature_path = self.local_feature_path)
        else:
            print("Skipping local feature extraction")

        if not self.global_feature_path.exists() or force_extract_global:
            self.global_feature_path = extract_features.main(global_feature_conf, self.images_dir, feature_path = self.global_feature_path)
        else:
            print("Skipping global feature extraction")

        if not self.pairs_path.exists() or force_extract_pairs_from_covis:
            pairs_path = pairs_from_covisibility.main(self.colmap_dir, self.images_dir, self.pairs_path, self.num_matched_pairs_covis_db)
        else:
            print("Skipping covis pair finding")

        if not matches_path.exists() or force_match_features:
            matches_path = match_features.main(self.matcher_conf, pairs_path, local_feature_path, matches = matches_path)
        else:
            print("Skipping matching")

        sfm_dir = self.db_sfm_dir
        
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
