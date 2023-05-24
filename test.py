import Attack.Server as S
import Attack.Client as C

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
    print("\nPreparing server\n")
    office_server = prepare_server()
    
    print("\nPreparing client\n")
    monitor_desk_client = prep_client()
    
    print("\nLocalizing\n")
    office_server.localize(monitor_desk_client)

    print("\nAligning local poses with server returned poses and applying the transformation to the object model\n")
    
    rot, trans, inliers = monitor_desk_client.align_local_server_poses(office_server, rot_thresh=45, cc_thresh=0.5, scale=0.1)
                                    