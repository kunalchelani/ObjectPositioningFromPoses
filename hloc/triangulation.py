import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import h5py
import numpy as np
import subprocess
import pprint
import shutil
import os
import yaml

from .utils.read_write_model import (
        read_cameras_binary, read_images_binary, CAMERA_MODEL_NAMES, rotmat2qvec,
        write_points3D_binary, write_images_binary, read_cameras_text, read_images_text)
from .utils.database import COLMAPDatabase
from .utils.parsers import names_to_pair


def create_empty_model(reference_model, empty_model):
    logging.info('Creating an empty model.')
    empty_model.mkdir(exist_ok=True)
    shutil.copyfile(reference_model/'cameras.bin', empty_model/'cameras.bin')
    write_points3D_binary(dict(), empty_model / 'points3D.bin')
    images = read_images_binary(str(reference_model / 'images.bin'))
    images_empty = dict()
    for id_, image in images.items():
        images_empty[id_] = image._replace(
            xys=np.zeros((0, 2), float), point3D_ids=np.full(0, -1, int))
    write_images_binary(images_empty, empty_model / 'images.bin')


def create_db_from_model(model, database_path, filetype = "bin"):
    if database_path.exists():
        logging.warning('Database already exists.')
        
    if filetype == "bin":
        cameras = read_cameras_binary(str(model / 'cameras.bin'))
        images = read_images_binary(str(model / 'images.bin'))
    else:
        cameras = read_cameras_text(str(model / 'cameras.txt'))
        images = read_images_text(str(model / 'images.txt'))

    db = COLMAPDatabase.connect(database_path)
    db.create_tables()

    for i, camera in cameras.items():
        model_id = CAMERA_MODEL_NAMES[camera.model].model_id
        db.add_camera(
            model_id, camera.width, camera.height, camera.params, camera_id=i,
            prior_focal_length=True)

    for i, image in images.items():
        db.add_image(image.name, image.camera_id, image_id=i)

    db.commit()
    db.close()
    return {image.name: i for i, image in images.items()}


def import_features(image_ids, database_path, features_path):
    logging.info('Importing features into the database...')
    hfile = h5py.File(str(features_path), 'r')
    db = COLMAPDatabase.connect(database_path)

    for image_name, image_id in tqdm(image_ids.items()):
        keypoints = hfile[image_name]['keypoints'].__array__()
        keypoints += 0.5  # COLMAP origin
        db.add_keypoints(image_id, keypoints)

    hfile.close()
    db.commit()
    db.close()


def import_matches(image_ids, database_path, pairs_path, matches_path,
                   min_match_score=None, skip_geometric_verification=False, use_loftr = False):
    logging.info('Importing matches into the database...')

    with open(str(pairs_path), 'r') as f:
        pairs = [p.split() for p in f.readlines()]

    hfile = h5py.File(str(matches_path), 'r')
    db = COLMAPDatabase.connect(database_path)

    matched = set()
    for name0, name1 in tqdm(pairs):
        id0, id1 = image_ids[name0], image_ids[name1]
        if len({(id0, id1), (id1, id0)} & matched) > 0:
            continue
        pair = names_to_pair(name0, name1)
        if pair not in hfile:
            raise ValueError(
                f'Could not find pair {(name0, name1)}... '
                'Maybe you matched with a different list of pairs? '
                f'Reverse in file: {names_to_pair(name0, name1) in hfile}.')

        matches = hfile[pair]['matches0'].__array__()
        if not use_loftr:
            valid = matches > -1
            if min_match_score:
                scores = hfile[pair]['matching_scores0'].__array__()
                valid = valid & (scores > min_match_score)
            matches = np.stack([np.where(valid)[0], matches[valid]], -1)

        db.add_matches(id0, id1, matches)
        matched |= {(id0, id1), (id1, id0)}

        if skip_geometric_verification:
            print(id0, id1)
            #print(matches)
            db.add_two_view_geometry(id0, id1, matches)

    hfile.close()
    db.commit()
    db.close()


def geometric_verification(colmap_path, database_path, pairs_path):
    logging.info('Performing geometric verification of the matches...')
    cmd = [
        str(colmap_path), 'matches_importer',
        '--database_path', str(database_path),
        '--match_list_path', str(pairs_path),
        '--match_type', 'pairs',
        '--SiftMatching.max_num_trials', str(20000),
        '--SiftMatching.min_inlier_ratio', str(0.1)]
    subprocess.run(cmd, check=True)


def run_triangulation(colmap_path, model_path, database_path, image_dir,
                      empty_model):
    logging.info('Running the triangulation...')
    assert model_path.exists()

    cmd = [
        str(colmap_path), 'point_triangulator',
        '--database_path', str(database_path),
        '--image_path', str(image_dir),
        '--input_path', str(empty_model),
        '--output_path', str(model_path),
        '--Mapper.ba_refine_focal_length', '0',
        '--Mapper.ba_refine_principal_point', '0',
        '--Mapper.ba_refine_extra_params', '0']
    logging.info(' '.join(cmd))
    subprocess.run(cmd, check=True)

    stats_raw = subprocess.check_output(
        [str(colmap_path), 'model_analyzer', '--path', str(model_path)])
    stats_raw = stats_raw.decode().split("\n")
    stats = dict()
    for stat in stats_raw:
        if stat.startswith("Registered images"):
            stats['num_reg_images'] = int(stat.split()[-1])
        elif stat.startswith("Points"):
            stats['num_sparse_points'] = int(stat.split()[-1])
        elif stat.startswith("Observations"):
            stats['num_observations'] = int(stat.split()[-1])
        elif stat.startswith("Mean track length"):
            stats['mean_track_length'] = float(stat.split()[-1])
        elif stat.startswith("Mean observations per image"):
            stats['num_observations_per_image'] = float(stat.split()[-1])
        elif stat.startswith("Mean reprojection error"):
            stats['mean_reproj_error'] = float(stat.split()[-1][:-2])

    return stats

def run_triangulation_with_known_poses(colmap_path, output_model_path, database_path, image_dir,
                      input_model_path):
    logging.info('Running the triangulation...')
    assert output_model_path.exists()
    assert input_model_path.exists()

    cmd = [
        str(colmap_path), 'point_triangulator',
        '--database_path', str(database_path),
        '--image_path', str(image_dir),
        '--input_path', str(input_model_path),
        '--output_path', str(output_model_path),
        '--Mapper.ba_refine_focal_length', '0',
        '--Mapper.ba_refine_principal_point', '0',
        '--Mapper.ba_refine_extra_params', '0']
    logging.info(' '.join(cmd))
    subprocess.run(cmd, check=True)

    stats_raw = subprocess.check_output(
        [str(colmap_path), 'model_analyzer', '--path', str(output_model_path)])
    stats_raw = stats_raw.decode().split("\n")
    stats = dict()
    for stat in stats_raw:
        if stat.startswith("Registered images"):
            stats['num_reg_images'] = int(stat.split()[-1])
        elif stat.startswith("Points"):
            stats['num_sparse_points'] = int(stat.split()[-1])
        elif stat.startswith("Observations"):
            stats['num_observations'] = int(stat.split()[-1])
        elif stat.startswith("Mean track length"):
            stats['mean_track_length'] = float(stat.split()[-1])
        elif stat.startswith("Mean observations per image"):
            stats['num_observations_per_image'] = float(stat.split()[-1])
        elif stat.startswith("Mean reprojection error"):
            stats['mean_reproj_error'] = float(stat.split()[-1][:-2])

    return stats

def write_colmap_images_file_from_poses(colmap_images_file, poses, is_poses_dir = False):
    if not is_poses_dir:
        f_in = open(poses, 'r')
        f_out = open(colmap_images_file, 'w')

        idx = 1
        line = f_in.readline()
        while line != "":
            parts = line.strip().split(" ")
            imname = parts[0]
            qvec = np.array(parts[1:5], dtype = np.float32)
            tvec = np.array(parts[5:], dtype = np.float32)
            f_out.write("{} {} {} {} {} {} {} {} {} {}\n\n".format(idx, qvec[0], qvec[1], qvec[2], qvec[3], tvec[0], tvec[1], tvec[2], 1, imname))
            idx +=1
            line = f_in.readline()

    else:
        f_out = open(colmap_images_file, 'w')
        idx = 1

        for f in sorted(os.listdir(poses)):
            imname = f.split(".")[0] + ".color.jpg"
            data = np.loadtxt(poses / f)
            Rmat = data[0:3, 0:3]
            t = data[0:3, 3]

            #Convert to required format
            qvec = rotmat2qvec(Rmat.transpose())
            tvec = -1 * (Rmat.transpose() @ t.reshape(3,1)).reshape(3,)
            f_out.write("{} {} {} {} {} {} {} {} {} {}\n\n".format(idx, qvec[0], qvec[1], qvec[2], qvec[3], tvec[0], tvec[1], tvec[2], 1, imname))
            idx +=1

    f_out.close()

def triangulate_from_poses_without_ref_colmap(sfm_dir, poses, image_dir, pairs, features, matches, intrinsics,\
        colmap_path='colmap', skip_geometric_verification=False, min_match_score=None, is_poses_dir = False):

    # 1. Write cameras.txt from intrinsics
    # 2. Create empty points3D.txt file
    # 3. Create Images.txt file poses_file
    # 4. Write the database and input into it each image and its poses, cameras
    # 5. Import matches and keypoints into the database
    # 6. Triangulate

    assert poses.exists(), poses
    assert features.exists(), features
    assert pairs.exists(), pairs
    assert matches.exists(), matches

    sfm_dir.mkdir(parents=True, exist_ok=True)
    database_path = sfm_dir / "database.db"

    #  Write cameras.txt from intrinsics
    colmap_cameras_file = sfm_dir / "cameras.txt" 
    f_cam = open(colmap_cameras_file, 'w')
    f_cam.write("# Camera list with one line of data per camera:\n# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n# Number of cameras: 1\n")

    if intrinsics.suffixes[-1] == ".txt":
        intrinsics_ = np.loadtxt(intrinsics)
        fxy = intrinsics_[0,0]
        cx = intrinsics_[0,2]
        cy = intrinsics_[1,2]
        width = 1296
        height = 968 
        f_cam.write("1 SIMPLE_PINHOLE {} {} {} {} {}".format(width, height, fxy, cx, cy))
        f_cam.close()
    elif intrinsics.suffixes[-1] == ".yaml":
        f_yaml = open(intrinsics, 'r')
        data = yaml.load(f_yaml, Loader=yaml.FullLoader)
        intrinsics_ = data['camera_intrinsics']
        height = intrinsics_['height']
        width = intrinsics_['width']
        fx = intrinsics_['model'][0]
        fy = intrinsics_['model'][1]
        cx = intrinsics_['model'][2]
        cy = intrinsics_['model'][3]
        distortion_params = intrinsics_['distortion']
        f_cam.write("1 PINHOLE {} {} {} {} {} {}".format(width, height, fx, fy, cx, cy))
        f_yaml.close()

    f_cam.close()
    print("Written camera file.")

    # Create empty points3D.txt file
    points3D_file = sfm_dir / "points3D.txt"
    f_pts = open(points3D_file, 'w')
    f_pts.close()
    print("Written empty points file.")

    # Write Images file
    colmap_images_file = sfm_dir / "images.txt"
    write_colmap_images_file_from_poses(colmap_images_file, poses, is_poses_dir)
    print("Written Images file from known poses.")

    # Create empty database
    db_create_command = "colmap database_creator --database_path {}".format(database_path)
    print("Creating database")
    subprocess.run(db_create_command.split(" "), stderr=subprocess.STDOUT)
    print("Done !")

    # Write images into database
    images = read_images_text(colmap_images_file)
    cameras = read_cameras_text(colmap_cameras_file)
    print(cameras)

    db = COLMAPDatabase.connect(database_path)
    db.create_tables()

    for camera_id in cameras:
        camera = cameras[camera_id]
        model_id = CAMERA_MODEL_NAMES[camera.model].model_id
        db.add_camera(model_id, camera.width, camera.height, camera.params, camera_id=camera_id, prior_focal_length=True)

    for image_id in images:
        image = images[image_id]
        db.add_image(image.name, image.camera_id, image_id = image_id)
    db.commit()
    db.close()

    image_ids = {image.name:id for id, image in images.items()}
    print("Added cameras and images to db.")

    import_features(image_ids, database_path, features)
    import_matches(image_ids, database_path, pairs, matches, min_match_score, skip_geometric_verification)

    print("Imported Features and Matches to db")

    if not skip_geometric_verification:
        print("Running geometric verification.")
        geometric_verification(colmap_path, database_path, pairs)
    
    stats = run_triangulation_with_known_poses(colmap_path, sfm_dir, database_path, image_dir, sfm_dir)
    logging.info(f'Statistics:\n{pprint.pformat(stats)}')


def triangulate_using_poses_and_intrinsics(images_dir,
                                            sfm_dir, 
                                            poses_path, 
                                            pairs_path, 
                                            local_features_path, 
                                            matches_path, 
                                            camera_intrinsics_params,
                                            colmap_camera_model,
                                            colmap_path='colmap', 
                                            skip_geometric_verification=False, 
                                            min_match_score=None, 
                                            is_poses_dir=False):

    # 1. Write cameras.txt from intrinsics
    # 2. Create empty points3D.txt file
    # 3. Create Images.txt file poses_file
    # 4. Write the database and input into it each image and its poses, cameras
    # 5. Import matches and keypoints into the database
    # 6. Triangulate

    assert poses_path.exists(), poses_path
    assert local_features_path.exists(), local_features_path
    assert pairs_path.exists(), pairs_path
    assert matches_path.exists(), matches_path

    sfm_dir.mkdir(parents=True, exist_ok=True)
    database_path = sfm_dir / "database.db"

    #  Write cameras.txt from intrinsics
    colmap_cameras_file = sfm_dir / "cameras.txt" 
    f_cam = open(colmap_cameras_file, 'w')
    f_cam.write("# Camera list with one line of data per camera:\n# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n# Number of cameras: 1\n")

    width = camera_intrinsics_params['width']
    height = camera_intrinsics_params['height']
    fx = camera_intrinsics_params['fx']
    fy = camera_intrinsics_params['fy']
    cx = camera_intrinsics_params['cx']
    cy = camera_intrinsics_params['cy']
    
    if colmap_camera_model == "SIMPLE_PINHOLE":
        f_cam.write("1 SIMPLE_PINHOLE {} {} {} {} {}".format(width, height, fx, cx, cy))

    elif colmap_camera_model == "PINHOLE":
        f_cam.write("1 PINHOLE {} {} {} {} {} {}".format(width, height, fx, fy, cx, cy))
        
    f_cam.close()
    print("Written camera file.")

    # Create empty points3D.txt file
    points3D_file = sfm_dir / "points3D.txt"
    f_pts = open(points3D_file, 'w')
    f_pts.close()
    print("Written empty points file.")

    # Write Images file
    colmap_images_file = sfm_dir / "images.txt"
    write_colmap_images_file_from_poses(colmap_images_file, poses_path, is_poses_dir)
    print("Written Images file from known poses.")

    # Create empty database
    db_create_command = "colmap database_creator --database_path {}".format(database_path)
    print("Creating database")
    subprocess.run(db_create_command.split(" "), stderr=subprocess.STDOUT)
    print("Done !")

    # Write images into database
    images = read_images_text(colmap_images_file)
    cameras = read_cameras_text(colmap_cameras_file)
    print(cameras)

    db = COLMAPDatabase.connect(database_path)
    db.create_tables()

    for camera_id in cameras:
        camera = cameras[camera_id]
        model_id = CAMERA_MODEL_NAMES[camera.model].model_id
        db.add_camera(model_id, camera.width, camera.height, camera.params, camera_id=camera_id, prior_focal_length=True)

    for image_id in images:
        image = images[image_id]
        db.add_image(image.name, image.camera_id, image_id = image_id)
    db.commit()
    db.close()

    image_ids = {image.name:id for id, image in images.items()}
    print("Added cameras and images to db.")

    import_features(image_ids, database_path, local_features_path)
    import_matches(image_ids, database_path, pairs_path, matches_path, min_match_score, skip_geometric_verification)

    print("Imported Features and Matches to db")

    if not skip_geometric_verification:
        print("Running geometric verification.")
        geometric_verification(colmap_path, database_path, pairs_path)
    
    stats = run_triangulation_with_known_poses(colmap_path, sfm_dir, database_path, images_dir, sfm_dir)
    logging.info(f'Statistics:\n{pprint.pformat(stats)}')


def main(sfm_dir, reference_sfm_model, image_dir, pairs, features, matches,
         colmap_path='colmap', skip_geometric_verification=False,
         min_match_score=None, known_poses=False, input_model_path = "", use_loftr = False):

    assert reference_sfm_model.exists(), reference_sfm_model
    assert features.exists(), features
    assert pairs.exists(), pairs
    assert matches.exists(), matches

    sfm_dir.mkdir(parents=True, exist_ok=True)
    database = sfm_dir / 'database.db'
    empty_model = sfm_dir / 'empty'

    if not known_poses:
        create_empty_model(reference_sfm_model, empty_model)
        image_ids = create_db_from_model(empty_model, database)
    else:
        image_ids = create_db_from_model(input_model_path, database, filetype = "txt")

    import_features(image_ids, database, features)
    import_matches(image_ids, database, pairs, matches,
                   min_match_score, skip_geometric_verification, use_loftr=use_loftr)
    if not skip_geometric_verification:
        geometric_verification(colmap_path, database, pairs)

    if not known_poses:
        stats = run_triangulation(
            colmap_path, sfm_dir, database, image_dir, empty_model)
    else:
        stats = run_triangulation_with_known_poses(
            colmap_path, sfm_dir, database, image_dir, input_model_path)
    logging.info(f'Statistics:\n{pprint.pformat(stats)}')
    
    if not known_poses:
        shutil.rmtree(empty_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sfm_dir', type=Path, required=True)
    parser.add_argument('--reference_sfm_model', type=Path, required=True)
    parser.add_argument('--image_dir', type=Path, required=True)

    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--matches', type=Path, required=True)

    parser.add_argument('--colmap_path', type=Path, default='colmap')

    parser.add_argument('--skip_geometric_verification', action='store_true')
    parser.add_argument('--min_match_score', type=float)
    args = parser.parse_args()

    main(**args.__dict__)
