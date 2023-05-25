import Attack.Server as S
import Attack.Client as C
import viz_utils

def prepare_server():
    server = S.Server(name = "scene0040_00",
                    colmap_dir = "data/server/scene0040_00/colmap",
                    images_dir = "data/server/scene0040_00/images",
                    base_dir_db = "data/server/scene0040_00/db",
                    base_dir_query = "data/server/scene0040_00/query",
                    feature = "superpoint_inloc",
                    num_matched_pairs_covis_db = 30,
                    num_matched_pairs_covis_localization = 30,
                    thresh_ransac_pnp = 12)
    
    server.prep(perform_server_sfm=True)
    return server

def prep_client():
    print("Preparing client")
    client = C.Client(name = "computer_desk",
                      base_dir = "data/client/computer_desk/",
                      colmap_dir = "data/client/computer_desk/colmap",
                      images_dir = "data/client/computer_desk/images",
                      feature="superpoint_inloc")
    
    client.prep_for_localization()
    return client

if __name__ == "__main__":
    
    feature = 'superpoint_aachen'
    num_retrived_db_images = 30

    office_server = S.Server(name = "scene0040_00",
                    colmap_dir = "data/server/scene0040_00/colmap",
                    images_dir = "data/server/scene0040_00/images",
                    base_dir_db = "data/server/scene0040_00/db",
                    base_dir_query = "data/server/scene0040_00/query",
                    feature = feature,
                    num_matched_pairs_covis_db = 30,
                    num_matched_pairs_covis_localization = 30,
                    thresh_ransac_pnp = 12)
    
    office_server.prep(perform_server_sfm=True)


    print("Preparing client")
    monitor_desk_client = C.Client(name = "computer_desk",
                      base_dir = "data/client/computer_desk/",
                      colmap_dir = "data/client/computer_desk/colmap",
                      images_dir = "data/client/computer_desk/images",
                      feature=feature)
    
    monitor_desk_client.prep_for_localization()
    
    print("\nLocalizing\n")
    office_server.localize(client = monitor_desk_client,
                              num_retrived_db_images=num_retrived_db_images,
                              force_extract_local_features=False,
                              force_extract_global_features=False,
                              force_match_features=False,
                              force_retrival = False,
                              )

    print("\nAligning local poses with server returned poses and applying the transformation to the object model\n")
    
    # Assuming that the server and client are scaled in the same way, we can use the relative scale in the sfm models to align the poses
    rotation, translation = monitor_desk_client.align_local_server_poses(server = office_server,
                                                                                     rot_thresh=45,
                                                                                     cc_thresh=0.5,
                                                                                     scale=0.1,
                                                                                     num_retrived_db_images=num_retrived_db_images,
                                                                                     average_inliers=True,
                                                                                     query_imnames_dir=None,
                                                                                     )    
    
    # Transform the object model using the obtained transformation
    monitor_desk_client.transform_object_model(server = office_server,
                                                  scale = 0.1,
                                                  rotation = rotation,
                                                  translation = translation,
                                                  num_retrived_db_images = num_retrived_db_images)
    
    # Visualize the attack

    server_poses_path = monitor_desk_client.base_dir / office_server.name / feature / f"poses_{office_server.matcher_name}_{num_retrived_db_images}_{office_server.thresh_ransac_pnp}.txt"
    server_model_path = office_server.db_sfm_dir / "sparse.ply"

    viz_utils.visualize_attack(server_poses_path = server_poses_path,
                               local_poses_path = None,
                               object_model_path = None,
                               transformed_object_model_path = None,
                               inliers_path = None,
                               server_model_path = server_model_path,
                               show_server_poses = True,
                               highlight_inlier_poses = False,
                               show_inlier_poses_only = False,
                               show_transformed_local_poses = False,
                               show_transformed_object = False,
                               show_server_map = True,
                               server_map_unicolor = False,
                               object_model_unicolor = False,
                               num_retrived_db_images = num_retrived_db_images,
                               )