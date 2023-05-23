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

def prep_client():
    client = C.Client(base_dir = "data/client/computer_desk/",
                      colmap_dir = "data/client/computer_desk/colmap",
                      images_dir = "data/client/computer_desk/images",
                      feature="superpoint_inloc",)

if __name__ == "__main__":
    prep_client()

