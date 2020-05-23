import os
import cv2
import sys
import json
import subprocess
import numpy as np
import glob
import copy

def edit_sfm_data(camparam_path, in_path):
    views = {}
    max_id = -1
    with open(in_path, 'r') as f:
        j = json.load(f)
        views_in = j['views']
        for v in views_in:
            key = v['key']
            pid = v['value']['polymorphic_id']
            # int_id = v['value']['ptr_wrapper']['data']['id_intrinsic']
            int_polymorphic = v['value']['ptr_wrapper']['id']
            ext_id = v['value']['ptr_wrapper']['data']['id_pose']
            views[key] = {'key': key, 'polymorphic_id': pid, 'int_polymorphic': int_polymorphic, 'ext_id': ext_id}
            max_id = max(max_id, int_polymorphic)
        f.close()
    intrinsics = []
    extrinsics = []
    
    with open(camparam_path, 'r') as f:
        j = json.load(f)
        cams = j['cam_params']
        for key in views.keys():
            v = cams[str(key)]
            print('camera {}'.format(key))
            # intrinsics
            focal_length = (v['fx'] + v['fy']) / 2.0
            principal_point = [float(v['cx']), float(v['cy'])]
            disto_k3 = [float(v['k1']), float(v['k2']), float(v['k3'])]
            disto_t2 = [float(v['p1']), float(v['p2'])]
            intrinsic = {'key': key, 'value': {'polymorphic_id': views[key]['int_polymorphic'], 'polymorphic_name': 'pinhole_radial_k3', 'ptr_wrapper': {
                'id': max_id + key + 1, 'data': {'width': 4000, 'height': 2160, 'focal_length': focal_length, 'principal_point': principal_point, 'disto_k3': disto_k3, 'disto_t2': disto_t2}
            }}}
            intrinsics.append(intrinsic)
            
            # extrinsics
            rvec = np.array(v['rvec'])
            rot, _ = cv2.Rodrigues(rvec)
            tvec = v['tvec']
            
            # tvec to camera position w.r.t. world
            tvec = -rot.T.dot(tvec)
            extrinsic = {'key': views[key]['ext_id'], 'value': {'rotation': [[rot[0, 0], rot[0, 1], rot[0, 2]], [rot[1, 0], rot[1, 1], rot[1, 2]], [rot[2, 0], rot[2, 1], rot[2, 2]]], 'center': [tvec[0], tvec[1], tvec[2]]}}
            extrinsics.append(extrinsic)
        f.close()


    in_path0 = in_path.split('.json')[0]
    with open(in_path, 'r') as f:
        j = json.load(f)
        views = j['views']
        for i in range(len(views)):
            views[i]['value']['ptr_wrapper']['data']['id_intrinsic'] = i
        j['intrinsics'] = intrinsics
        j['extrinsics'] = extrinsics
        
        out_path = in_path0 + '_newParams.json'
        with open(out_path, 'w+') as fo:
            json.dump(j, fo, indent=4)
            print('    saved to:', out_path)
            fo.close()
        f.close()
        return out_path
    return None

def print_space(n=20):
    for i in range(n):
        print('#')
def edit_sparse_points(in_path, ref_path, suffix='_custom_sparse'):
    """
    modify json from in_path using data from ref_path
    """
    out_path = in_path.split('.json')[0] + suffix + '.json'


    # load in_path json
    with open(in_path, 'r') as f:
        robust_fitting = json.load(f)
        structure = robust_fitting['structure']

    # load ref_path json
    with open(ref_path, 'r') as f:
        j = json.load(f)
        print(j.keys())
        tri = j['initialTriangulation']
        cam_ids_used = j['camIdsUsed']
        corr_pts = j['CorrPts']
    
    # reoragnize data into correct 'structure' format of in_path
    structure_new = []
    for i, t in enumerate(tri):
        dic = {'key': i, 'value': {}}
        dic['value']['X'] = t
        if t[2] < 0:
            continue
            
        observations = []
        cam_ids = cam_ids_used[i]
        for cam_idx in cam_ids:
            pts = np.array(corr_pts[cam_idx][i])
            assert(pts[0] > 0 and pts[1] > 0)
            
            obsv = {'key': cam_idx, 'value': {'id_feat': i, 'x': [pts[0], pts[1]]}}
            observations.append(obsv)
        dic['value']['observations'] = observations
        structure_new.append(dic)

    # swap 'structure' data from in_path using ref_path
    out_json = copy.deepcopy(robust_fitting)
    out_json['structure'] = structure_new


    # save
    with open(out_path, 'w+') as f:
        json.dump(out_json, f, indent=4)
        print('Input to OpenMVS:', out_path)
        return out_path
    return None

# ------------ #
# EDIT HERE. Run this code inside a folder at the same level as 'images' folder.
# ------------ #
camparam_path = r'D:\CalibrationData\CameraCalibration\2019_12_13_Lada_Capture_k1k2k3p1p2\FinalCamParams\cam_params.json'
sparse_pts_dir = r'Z:\2020_03_18_LadaAnimationWholeSeq\WholeSeq\Corrs_RThres1.5_HardRThres_1.5'
use_custom_sparse = True

#
#
#
#
#
print('-> If all 16 cameras are NOT used for reconstruction, camera indices must be modified: {}'.format(camparam_path))
print('-> Delete camera parameters that are not used.')
con = input('-> Continue? [Y/N] ')
while con != 'y' and con != 'Y' and con != 'N' and con != 'n':
    con = input('-> Invalid input: {}. Continue? [Y/N] '.format(con))
if con == 'N' or con == 'n':
    print('-> Read: {}. Exiting'.format(con))
    exit()
#
#
#
#
#
#
#
#
#
# Indicate the openMVG binary directory
OPENMVG_SFM_BIN = "D:/1_Projects/200228_OpenMVS/200229_openMVG_build/build/Windows-AMD64-Release/Release"
OPENMVS_BIN = "D:/1_Projects/200228_OpenMVS/200301_openMVS_build/bin/vc15/x64/Release"

# Indicate the openMVG camera sensor width directory
CAMERA_SENSOR_WIDTH_DIRECTORY = "D:/1_Projects/200228_OpenMVS/200229_openMVG_build/src/software/SfM" + "/../../openMVG/exif/sensor_width_database"

camera_file_params = os.path.join(CAMERA_SENSOR_WIDTH_DIRECTORY, "sensor_width_camera_database.txt")

curr_dir = os.getcwd()
input_dir = curr_dir + '/images'

# extract iamge name
print('Image dir:', input_dir)
img_paths = glob.glob(input_dir + '/*.jpg')[0]
img_name = img_paths.split('/')[-1].split('.jpg')[0].split('\\')[-1]
img_name = img_name[1:]
print('Image name:', img_name)

output_dir = curr_dir + '/output'
matches_dir = os.path.join(output_dir, "1_matches")
g_reconstruction_dir = os.path.join(output_dir, "2_reconstruction_global")
dense_dir = os.path.join(output_dir, "3_dense")

# Create the ouput/matches folder if not present
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
if not os.path.exists(matches_dir):
    os.mkdir(matches_dir)
if not os.path.exists(g_reconstruction_dir):
    os.mkdir(g_reconstruction_dir)
if not os.path.exists(dense_dir):
    os.mkdir(dense_dir)


print ("Using input dir  : ", input_dir)
print ("      output_dir : ", output_dir)
print ("===== Image listing: output={}".format(matches_dir))
pIntrisics = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_SfMInit_ImageListing"),  "-i", input_dir, "-o", matches_dir, "-d", camera_file_params, "-g", "0"])
pIntrisics.wait()

print("   * Modify sfm_data.json using our own camera parameters")
modified_path = edit_sfm_data(camparam_path, matches_dir + '/sfm_data.json')

print_space()
print ("===== Compute features: output={}".format(matches_dir))
pFeatures = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeFeatures"),  "-i", modified_path, "-o", matches_dir, "-p", "HIGH", "-n", "16", "-f", "0"] )
pFeatures.wait()

print_space()
print ("===== Compute matches: output={}".format(matches_dir))
pMatches = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeMatches"),  "-i", modified_path, "-o", matches_dir, "-g", "f", "-r", "0.8", "-f", "0"])
pMatches.wait()

# print_space()
# print ("===== 4. Do Global reconstruction: output={}".format(g_reconstruction_dir))
# pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_GlobalSfM"),  "-i", matches_dir+"/sfm_data_modified.json", "-m", matches_dir, "-o", g_reconstruction_dir, "-M", matches_dir + "/matches.f.bin"])
# pRecons.wait()

# # # # # NOT USED
# # # # # print ("5. Colorize Structure")
# # # # # pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeSfM_DataColor"),  "-i", g_reconstruction_dir+"/robust.bin", "-o", os.path.join(g_reconstruction_dir,"robust_colorized.ply")] )
# # # # # pRecons.wait()

# compute final valid structure from the known camera poses
print_space()
print ("===== Structure from Known Poses (robust triangulation)")
pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeStructureFromKnownPoses"),  "-i", modified_path, "-o", g_reconstruction_dir+"/robust_fitting.json", "-m", matches_dir])
pRecons.wait()


print("   * Modify sfm_data.json using our own camera parameters")
modified_path = edit_sfm_data(camparam_path, g_reconstruction_dir + '/robust_fitting.json')
assert(modified_path is not None)

if use_custom_sparse:
    print("   * Modify robust_fitting.json using our sparse points")
    input_sparse_path = sparse_pts_dir + '/A000{}.json'.format(img_name)
    modified_path = edit_sparse_points(modified_path, input_sparse_path)
    assert(modified_path is not None)

modified_fname = modified_path.split('.json')[0]
print('>> Input sparse points path:', modified_path)


print_space()
print("===== Convert sfm_data.bin to sfm_data.json: {}".format(g_reconstruction_dir+"/sfm_data_robust.bin"))
pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ConvertSfM_DataFormat"),  "-i", modified_path, "-o", modified_fname + '.bin', "-V", "-I", "-E", "-S", "-C"] )
pRecons.wait()

#
# MVS
#
# print_space()
# print("===== 7. Convert sfm_data.bin to sfm_data.json: {}".format(g_reconstruction_dir+"/sfm_data_robust.bin"))
# pRecons = subprocess.Popen( [os.path.join(OPENMVS_BIN, "InterfaceOpenMVG"),  "-i", matches_dir+"/sfm_data_modified.json", "-o", dense_dir+'/scene.mvs'] )
# pRecons.wait()

print_space()
print ("===== MVS: data preparation")
pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_openMVG2openMVS"),  "-i", modified_fname + '.bin', "-o", dense_dir+'/scene.mvs', "-d", dense_dir] )
pRecons.wait()

print_space()
print ("===== MVS: main")
pRecons = subprocess.Popen( [os.path.join(OPENMVS_BIN, "DensifyPointCloud"),  "-i", dense_dir+"/scene.mvs"] )
pRecons.wait()

print()
print('=== DONE ===')