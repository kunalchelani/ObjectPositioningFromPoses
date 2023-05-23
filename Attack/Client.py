# Author : Kunal Chlelani
# Email : kunalchelani@gmail.com

import os
import sys
from pathlib import Path
from typing import Union
from hloc import extract_features
from . import utils_prep

# A client class that is used to attack the server

class Client:

    def __init__(self,
                 base_dir: Union[str, Path],
                 colmap_dir: Union[str, Path],
                 images_dir: Union[str, Path],
                 feature: str,
                 ) -> None:

        # Create base dir if it does not exist
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        if not isinstance(base_dir, Path):
            base_dir = Path(base_dir)

        if not os.path.exists(colmap_dir):
            raise ValueError(f"colmap_dir : {colmap_dir} does not exist")
        if not isinstance(images_dir, Path):
            images_dir = Path(images_dir)
        
        if not os.path.exists(images_dir):
            raise ValueError(f"images_dir : {images_dir} does not exist")
        if not isinstance(images_dir, Path):
            images_dir = Path(images_dir)

        self.feature = feature
        self.local_feature_conf = utils_prep.local_feature_confs[feature]
        self.global_feature_conf = utils_prep.global_feature_conf

        self.query_images_with_intrinsics_file_path = base_dir / "query_images_with_intrinsics.txt"
        self.local_features_path = base_dir / feature / self.local_feature_conf['output']
        self.global_feature_path = base_dir / self.global_feature_conf['output']


    def prep_for_localization(self, 
                              query_fnames: list = None,
                              force_extract_local_features: bool = False,
                              force_extract_global_features: bool = False,
                              intrinsics_dict: dict = None):
        
        # Extract local features
        # -------------------------- Client query files preparation --------------------------#
        if intrinsics_dict is None:
            intrinsics_dict = utils_prep.get_intrisics_dict_from_colmap_model(Path(self.colmap_dir))

        query_fnames = utils_prep.write_query_intrinsics_file(intrinsics_dict = intrinsics_dict,
                                    query_fname_with_intrinsics = self.query_images_with_intrinsics_file_path, 
                                    query_fnames = query_fnames)

        # -------------------------- Client feature extraction ---------------------------- #

        if force_extract_local_features or (not self.local_features_path.exists()):
            print(f"Extracting client features to be saved at {self.local_features_path}")
            
            _ = extract_features.main(conf = self.local_feature_conf,
                                                                image_dir = self.images_dir, 
                                                                feature_path = self.local_features_path,
                                                                )

        if force_extract_global_features or (not self.global_feature_path.exists()):

            _ = extract_features.main(conf = self.global_feature_conf,
                                                                image_dir = self.images_dir,
                                                                feature_path = self.global_features_path,
                                                                )

        pass

    def visualize_cams_server_map():
        pass

    def align_local_and_server_poses():
        pass

    def visualize_aligned_model_server_map():
        pass