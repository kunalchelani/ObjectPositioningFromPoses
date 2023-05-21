import Attack.Server as S

def prepare_server():
    server = S.Server(colmap_dir = "../data/server/scene0040_00/colmap",
                    images_dir = "../data/server/scene0040_00/images",
                    base_dir_db = "../data/server/scene0040_00/db",
                    base_dir_attack = "../data/server/scene0040_00/attack",
                    feature = "superpoint_inloc",
                    num_matched_pairs_covis_db = 30,
                    num_matched_pairs_covis_localization = 30,
                    thresh_ransac_pnp = 12)
    
    server.prep()


if __name__ == "__main__":
    prepare_server()