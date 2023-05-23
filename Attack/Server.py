# Author : Kunal Chlelani
# Email : kunalchelani@gmail.com

import os
import sys
from pathlib import Path
from typing import Union
from hloc import extract_features, match_features, pairs_from_covisibility, triangulation, pairs_from_retrieval, localize_sfm
from . import utils_prep
# Define the Class for a Server of a client-server based localization system

class Server:

    def __init__(self,
                 colmap_dir: Union[str, Path],
                 images_dir: Union[str, Path],
                 base_dir_db: Union[str, Path] = "data/server/db",
                 base_dir_attack: Union[str, Path] = "data/server/attack",
                 feature: str = "Superpoint_inloc",
                 num_matched_pairs_covis_db: int = 30,
                 num_matched_pairs_covis_localization: int = 30,
                 thresh_ransac_pnp: float = 12) -> None:

        parent_dir = Path(__file__).parent.parent

        if isinstance(colmap_dir, str):
            colmap_dir = Path(colmap_dir)
        self.colmap_dir = parent_dir / colmap_dir

        if isinstance(images_dir, str):
            images_dir = Path(images_dir)
        self.images_dir = images_dir
        self.images_dir = parent_dir / images_dir

        if isinstance(base_dir_db, str):
            base_dir_db = Path(base_dir_db)
        self.base_dir_db =  base_dir_db
        self.base_dir_db = parent_dir / base_dir_db

        if isinstance(base_dir_attack, str):
            base_dir_attack = Path(base_dir_attack)
        self.base_dir_attack = base_dir_attack
        self.base_dir_attack = parent_dir / base_dir_attack

        self.feature = feature
        
        self.num_matched_pairs_covis_db = num_matched_pairs_covis_db
        self.num_matched_pairs_covis_localization = num_matched_pairs_covis_localization
        self.thresh_ransac_pnp = thresh_ransac_pnp

        self.local_feature_conf = utils_prep.local_feature_confs[feature]
        self.matcher_conf = utils_prep.matcher_confs[feature]
        self.global_feature_conf = utils_prep.global_feature_conf

        self.local_feature_name = self.local_feature_conf['model']['name']
        self.matcher_name = self.matcher_conf['model']['name']
        self.global_feature_name = self.global_feature_conf['model']['name']

        self.local_feature_dir = self.base_dir_db / f"{self.local_feature_name}/"
        self.global_feature_dir = self.base_dir_db / f"{self.global_feature_name}/"

        self.local_feature_path = self.local_feature_dir / f"{self.local_feature_conf['output']}.h5"
        self.global_feature_path = self.global_feature_dir / f"{utils_prep.global_feature_conf['output']}.h5"

        self.pairs_path = self.base_dir_db / f"pairs_covisibility_{num_matched_pairs_covis_db}.txt"
        self.matches_path = self.local_feature_dir / f"matches_{self.matcher_name}.h5"

        self.db_sfm_dir = self.base_dir_db / f"{self.local_feature_name}/" / "sfm"

    def prep(self,
             force_extract_local: bool = False,
             force_extract_global: bool = False,
             force_extract_pairs_from_covis:bool  = False,
             force_match_features: bool = False,
             perform_server_sfm: bool = False,
             export_ply: bool  = False) -> None:

        # Skip if already done and not forced

        if not self.local_feature_path.exists() or force_extract_local:
            _ = extract_features.main(self.local_feature_conf, self.images_dir, feature_path = self.local_feature_path)
        else:
            print("Skipping local feature extraction")

        if not self.global_feature_path.exists() or force_extract_global:
            _ = extract_features.main(self.global_feature_conf, self.images_dir, feature_path = self.global_feature_path)
        else:
            print("Skipping global feature extraction")

        if not self.pairs_path.exists() or force_extract_pairs_from_covis:
            _ = pairs_from_covisibility.main(self.colmap_dir, self.pairs_path, self.num_matched_pairs_covis_db)
        else:
            print("Skipping covis pair finding")

        if not self.matches_path.exists() or force_match_features:
            _ = match_features.main(self.matcher_conf, self.pairs_path, self.local_feature_path, matches = self.matches_path)
        else:
            print("Skipping matching")

        sfm_dir = self.db_sfm_dir
        
        if perform_server_sfm:
            if sfm_dir.is_dir():
                # List the files in the directory
                files = os.listdir(sfm_dir)
                # If cameras.bin, images.bin and points3D.bin are present, skip sfm
                if "cameras.bin" in files and "images.bin" in files and "points3D.bin" in files:
                    print("Skipping sfm")
                    return    
                
            print("--------------- Started Triangulation ------------------")
            triangulation.main(
                sfm_dir = sfm_dir,
                reference_sfm_model= self.colmap_dir,
                image_dir=self.images_dir,
                pairs=self.pairs_path, 
                features=self.local_feature_path,
                matches=self.matches_path,
                colmap_path='colmap')

        else:
            print("Skipping sfm")

        # Export ply file if needed

        if export_ply:
            colmap_pts_path = sfm_dir / "points3D.bin"
            out_ply_path = sfm_dir / "sparse.ply"
            utils_prep.export_colmap_pts_as_ply(str(colmap_pts_path), str(out_ply_path))


    def localize(self,
                 client_name: str,
                 query_images_with_intrinsics: Union[Path, str] = None,
                 client_local_features_path: Union[Path, str] = None,
                 client_global_features_path: Union[Path, str] = None,
                 client_images_dir: Union[Path, str] = None,
                 num_retrived_db_images: int  = 30,
                 force_extract_local_features: bool = False,
                 force_extract_global_features: bool = False,
                 force_retrival: bool = False,
                 force_match_client_server: bool = False,
                 ):

        # Extract local features from query images
        
        base_path_attack_feat = self.base_dir_attack / client_name / f"{self.feature}"
        if not base_path_attack_feat.exists():
            os.makedirs(base_path_attack_feat, exist_ok=True)
        
        client_server_retrieved_pairs_path = self.base_dir_attack / client_name / f"pairs_retrieved_{num_retrived_db_images}.txt"

        client_server_matches_path = base_path_attack_feat  / f"matches_{self.matcher_name}_{num_retrived_db_images}.h5"

        localized_poses_file_path = base_path_attack_feat / f"poses_{self.matcher_name}_{num_retrived_db_images}_{self.thresh_ransac_pnp}.txt"

        # Get number of query images by reading the intrinsics file

        assert query_images_with_intrinsics is not None, "Please provide the path to the query images with intrinsics file"

        query_fnames = []
        with open(query_images_with_intrinsics, 'r') as f:
            for line in f.readlines():
                query_fnames.append(line.split(" ")[0])
        

        print(f"Localizing {len(query_fnames)} images provided against the server using {self.feature} local feature the RANSAC thresh: {self.thresh_ransac_pnp}")
        
        # -------------------------- Client feature extraction ----------------------------- #

        if force_extract_local_features or (not client_local_features_path.exists()):
            print(f"Extracting client features to be saved at {client_local_features_path}")
            
            client_local_features_path = extract_features.main(conf = self.local_feature_conf,
                                                                image_dir = Path(client_images_dir), 
                                                                feature_path = client_local_features_path,
                                                                )

        if force_extract_global_features or (not client_global_features_path.exists()):

            client_global_features_path = extract_features.main(conf = self.global_feature_conf,
                                                                image_dir = Path(client_images_dir),
                                                                feature_path = client_global_features_path,
                                                                )
            
        # -------------------------- Get retrived images and match ------------------------- #
    
        if force_retrival or (not client_server_retrieved_pairs_path.exists()):

            pairs_from_retrieval.main(descriptors = client_global_features_path, 
                                        output = client_server_retrieved_pairs_path, 
                                        num_matched = num_retrived_db_images, 
                                        query_list = query_fnames, 
                                        db_model = self.db_sfm_dir, 
                                        db_descriptors = self.global_feature_path,
                                        )

        if force_match_client_server or (not client_server_matches_path.exists()):

            client_server_matches_path = match_features.main(conf= self.matcher_conf, 
                                                            pairs = client_server_retrieved_pairs_path, 
                                                            features = client_local_features_path, 
                                                            matches = client_server_matches_path, 
                                                            features_ref = self.local_feature_path,
                                                            force_match = force_match_client_server,
                                                            )
        
        # -------------------------------- Localization ------------------------------------ #
        
        if not os.path.exists(localized_poses_file_path):
            localize_sfm.main(
                reference_sfm = self.db_sfm_dir,
                queries = query_images_with_intrinsics,
                retrieval = client_server_retrieved_pairs_path,
                features = client_local_features_path,
                matches = client_server_matches_path,
                results = localized_poses_file_path,
                covisibility_clustering=True,
                ransac_thresh = self.thresh_ransac_pnp,
            )

        
        return self.base_dir_attack / client_name