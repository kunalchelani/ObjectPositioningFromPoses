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
                 name,
                 base_dir: Union[str, Path],
                 colmap_dir: Union[str, Path],
                 images_dir: Union[str, Path],
                 feature: str,
                 ) -> None:

        print(f"Initialising client {name}")
        
        parent_dir = Path(__file__).parent.parent
        
        # Create base dir if it does not exist
        if not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)
        if not isinstance(base_dir, Path):
            base_dir = Path(base_dir)
        self.base_dir = parent_dir / base_dir
        
        if not os.path.exists(colmap_dir):
            raise ValueError(f"colmap_dir : {colmap_dir} does not exist")
        if not isinstance(colmap_dir, Path):
            colmap_dir = Path(colmap_dir)
        self.colmap_dir = parent_dir / colmap_dir
        
        if not os.path.exists(images_dir):
            raise ValueError(f"images_dir : {images_dir} does not exist")
        if not isinstance(images_dir, Path):
            images_dir = Path(images_dir)
        self.images_dir = parent_dir / images_dir
        
        self.name = name
        self.feature = feature
        self.local_feature_conf = utils_prep.local_feature_confs[feature]
        self.global_feature_conf = utils_prep.global_feature_conf

        self.query_images_with_intrinsics_file_path = base_dir / "query_images_with_intrinsics.txt"
        self.local_feature_path = base_dir / feature / self.local_feature_conf['output']
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

        self.query_fnames = utils_prep.write_query_intrinsics_file(intrinsics_dict = intrinsics_dict,
                                    query_fname_with_intrinsics = self.query_images_with_intrinsics_file_path, 
                                    query_fnames = query_fnames)

        # -------------------------- Client feature extraction ---------------------------- #

        if force_extract_local_features or (not self.local_feature_path.exists()):
            print(f"Extracting local features to be saved at {self.local_feature_path}")
            
            _ = extract_features.main(conf = self.local_feature_conf,
                                                                image_dir = self.images_dir, 
                                                                feature_path = self.local_feature_path,
                                                                )

        if force_extract_global_features or (not self.global_feature_path.exists()):
            print(f"Extracting global features to be saved at {self.global_feature_path}")
            _ = extract_features.main(conf = self.global_feature_conf,
                                                                image_dir = self.images_dir,
                                                                feature_path = self.global_feature_path,
                                                                )

        pass

    def visualize_cams_server_map():
        pass

    def align_local_and_server_poses():
        pass

    def visualize_aligned_model_server_map():
        pass