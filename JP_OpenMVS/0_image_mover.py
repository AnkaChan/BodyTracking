import glob
import os
import shutil
from PIL import Image

def copy_pgm2jpg(in_path, to_dir):
    img_name = in_path.split('\\')[-1].split('.')[0]
    im = Image.open(in_path)
    out_path = to_dir + '\\{}.jpg'.format(img_name)
    im = im.convert("RGB")
    im.save(out_path)
    print('    saved pgm -> jpg:', out_path)

# =================== #
# 1/2. edit here
# =================== #        
# image name to move
image_names = ['03067', '04735', '06250', '06550']

for image_name in image_names:

    for i in range(0, 5-len(image_name)):
    	image_name = '0' + image_name

    print('Image name: {}'.format(image_name))

    # =================== #
    # 2/2. edit here
    # =================== #
    from_dir_root = r'Z:\2019_12_13_Lada_Capture\Converted'
    to_dir = r'D:\1_Projects\200228_OpenMVS\200318_openMVS_run_customSparse\200517_2019_12_13_Lada_Capture\\' + image_name


    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    if not os.path.exists(to_dir):
        os.mkdir(to_dir)
        print('Created output dir:', to_dir)

    to_dir += r'\\images'
    if not os.path.exists(to_dir):
        os.mkdir(to_dir)
        print('Created output dir:', to_dir)

    # get camera folders (e.g., {}\A ~ {}\P)
    folders = [f.path for f in os.scandir(from_dir_root) if f.is_dir()]

    print('Subfolders of: {}'.format(from_dir_root))
    for i, f in enumerate(folders):
        print('  [{}]. {}'.format(i, f))
        glob_path = f + '\\*{}.pgm'.format(image_name)
        paths = glob.glob(glob_path)
        
        if len(paths) != 1:
            print('  [ERROR] only 1 image should have been found: {}'.format(len(paths)))
            print('          {}'.format(glob_path))
            assert(False)
        else:
            for from_path in paths:
                copy_pgm2jpg(from_path, to_dir)

    print('=== Done ===')
