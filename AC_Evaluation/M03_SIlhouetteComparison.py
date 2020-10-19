import sys
sys.path.append('../AC_FitPipeline')
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from os.path import join
from Utility_Rendering import *
import Config
import M05_Visualization
import cv2
import numpy as np

def renderSilhouettes(inFramesFolder, outFolder, meshExt, convertToM=False, cleanPlateFolder= r'F:\WorkingCopy2\2020_07_26_NewPipelineTestData\CleanPlateExtracted\RgbUndist',
                      inTextureMeshFile = r"..\Data\TextureMap2Color\Initial1Frame\SMPLWithSocks_tri.obj",
                      camParamF = r'F:\WorkingCopy2\2020_05_31_DifferentiableRendererRealData\CameraParams\cam_params.json'):

    cfg = Config.RenderingCfg()
    cfg.sigma = 1e-10
    cfg.blurRange = 1e-10
    cfg.imgSize = 1080
    cfg.faces_per_pixel = 5

    cfg.numCams = 16
    cfg.batchSize = 8

    os.makedirs(outFolder, exist_ok=True)
    M05_Visualization.renderConsecutiveFrames(inFramesFolder, cleanPlateFolder, inTextureMeshFile, camParamF, outFolder,
                                              frameNames=None, cfg=cfg,
                                              inMeshExt=meshExt, convertToM=convertToM, rendererType='Silhouette')

def overlaySils(refSilF, renderedSilF, outFile):
    _, refSil = load_image(refSilF)

    renderedSil = cv2.imread(renderedSilF)

    overlay = np.zeros(renderedSil.shape, dtype=renderedSil.dtype)

    overlay[:, :, 1] = refSil[:, :, 1]
    overlay[:, :, 2] = renderedSil[:, :, 2]

    # cv2.imshow('overlay', overlay)
    # cv2.waitKey()
    cv2.imwrite(outFile, overlay)

