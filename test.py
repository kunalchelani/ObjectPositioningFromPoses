import Attack.Server as S
import Attack.Client as C

def prepare_server():
    server = S.Server(colmap_dir = "data/server/scene0040_00/colmap",
                    images_dir = "data/server/scene0040_00/images",
                    base_dir_db = "data/server/scene0040_00/db",
                    base_dir_attack = "data/server/scene0040_00/attack",
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
    print("\nPreparing server\n")
    server = prepare_server()
    
    print("\nPreparing client\n")
    client = prep_client()
    
    print("\nLocalizing\n")
    server.localize(client_name = client.name,
                 query_images_with_intrinsics = client.query_images_with_intrinsics_file_path,
                 client_local_features_path =  client.local_feature_path,
                 client_global_features_path = client.global_feature_path,
                 client_images_dir = client.images_dir,
                 )

