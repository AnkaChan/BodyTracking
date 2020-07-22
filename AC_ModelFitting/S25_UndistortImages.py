from Utility import *

if __name__ == '__main__':
    # inFolder = r'F:\WorkingCopy2\2020_06_04_SilhouetteExtraction\CleanPlate\ToRGB'
    inFolder = r'F:\WorkingCopy2\2020_06_21_TextureRendering\CleanPlatesExtracted\rgb'

    camParamF = r'F:\WorkingCopy2\2020_05_31_DifferentiableRendererRealData\CameraParams\cam_params.json'

    load_images(inFolder, True, camParamF, writeUndistorted=True)