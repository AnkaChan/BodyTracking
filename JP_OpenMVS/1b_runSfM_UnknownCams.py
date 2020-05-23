import json
import numpy as np
import cv2
def print_space(n=20):
	for i in range(n):
		print('#')

# Indicate the openMVG binary directory
OPENMVG_SFM_BIN = "D:/1_Projects/200228_OpenMVS/200229_openMVG_build/build/Windows-AMD64-Release/Release"
OPENMVS_BIN = "D:/1_Projects/200228_OpenMVS/200301_openMVS_build/bin/vc15/x64/Release"

# Indicate the openMVG camera sensor width directory
CAMERA_SENSOR_WIDTH_DIRECTORY = "D:/1_Projects/200228_OpenMVS/200229_openMVG_build/src/software/SfM" + "/../../openMVG/exif/sensor_width_database"

import os
import subprocess
import sys

curr_dir = os.getcwd()

input_dir = curr_dir + '/images'
output_dir = curr_dir + '/output'
matches_dir = os.path.join(output_dir, "1_matches")
g_reconstruction_dir = os.path.join(output_dir, "2_reconstruction_global")
dense_dir = os.path.join(output_dir, "3_dense")
camera_file_params = os.path.join(CAMERA_SENSOR_WIDTH_DIRECTORY, "sensor_width_camera_database.txt")

print ("Using input dir  : ", input_dir)
print ("      output_dir : ", output_dir)

# Create the ouput/matches folder if not present
if not os.path.exists(output_dir):
  os.mkdir(output_dir)
if not os.path.exists(matches_dir):
  os.mkdir(matches_dir)
if not os.path.exists(g_reconstruction_dir):
    os.mkdir(g_reconstruction_dir)
if not os.path.exists(dense_dir):
    os.mkdir(dense_dir)


print ("===== 1. Intrinsics analysis: output={}".format(matches_dir))
pIntrisics = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_SfMInit_ImageListing"),  "-i", input_dir, "-o", matches_dir, "-d", camera_file_params, "-g", "0"])
pIntrisics.wait()

print_space()
print ("===== 2. Compute features: output={}".format(matches_dir))
pFeatures = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeFeatures"),  "-i", matches_dir+"/sfm_data.json", "-o", matches_dir, "-p", "HIGH", "-n", "16", "-f", "0"] )
pFeatures.wait()

print_space()
print ("===== 3. Compute matches: output={}".format(matches_dir))
pMatches = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeMatches"),  "-i", matches_dir+"/sfm_data.json", "-o", matches_dir, "-g", "f", "-r", "0.8", "-f", "0"])
pMatches.wait()

print_space()
print ("===== 4. Do Global reconstruction: output={}".format(g_reconstruction_dir))
pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_GlobalSfM"),  "-i", matches_dir+"/sfm_data.json", "-m", matches_dir, "-o", g_reconstruction_dir, "-M", matches_dir + "/matches.f.bin"])
pRecons.wait()

print_space()
print("===== 7. Convert sfm_data.bin to sfm_data.json: {}".format(g_reconstruction_dir+"/sfm_data_robust.bin"))
pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ConvertSfM_DataFormat"),  "-i", g_reconstruction_dir+"/sfm_data.bin", "-o", g_reconstruction_dir+"/sfm_data.json", "-V", "-I", "-E", "-S", "-C"] )
pRecons.wait()

# MVS
#
print_space()
print ("===== MVS: data preparation")
pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_openMVG2openMVS"),  "-i", g_reconstruction_dir+"/sfm_data.bin", "-o", dense_dir+'/scene.mvs', "-d", dense_dir] )
pRecons.wait()

print_space()
print ("===== MVS: main")
pRecons = subprocess.Popen( [os.path.join(OPENMVS_BIN, "DensifyPointCloud"),  "-i", dense_dir+"/scene.mvs"] )
pRecons.wait()

print()
print('=== DONE ===\n')