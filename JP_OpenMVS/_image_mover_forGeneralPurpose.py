import glob
import os
import shutil
from PIL import Image

cams = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
def copy_pgm2jpg(in_path, to_dir):
    img_name = in_path.split('/')[-1].split('.')[0]
    im = Image.open(in_path)
    out_path = to_dir + '/{}.png'.format(img_name)
    im = im.convert("RGB")
    im.save(out_path)
    print('    saved pgm -> png:', out_path)

# =================== #
# 1/2. edit here
# =================== #        
# image name to move
image_names = ['03052', '03990', '04917', '06950']

for image_name in image_names:

    for i in range(0, 5-len(image_name)):
    	image_name = '0' + image_name

    print('Image name: {}'.format(image_name))

    # =================== #
    # 2/2. edit here
    # =================== #
    from_dir = 'Z:/2019_12_13_Lada_Capture/Converted'
    to_dir = 'D:/Pictures/2019_12_13_Lada_Capture/RGB/grayPng/' + image_name
    dformat = 'png'
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

    for cam_idx, cam in enumerate(cams):
        from_path = from_dir + '/{}/{}{}.pgm'.format(cam, cam, image_name)
        copy_pgm2jpg(from_path, to_dir)

    print('=== Done ===')
