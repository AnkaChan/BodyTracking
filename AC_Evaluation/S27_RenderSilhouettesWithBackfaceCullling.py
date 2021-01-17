import sys
sys.path.append('../AC_FitPipeline')
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from Utility_Rendering import *
import Config
import M05_Visualization
from Utility import *
import tqdm
import cv2
if __name__ == '__main__':
    # inFolder = r'F:\WorkingCopy2\2021_01_14_AnimatinoSeqs\LongSequences\LadaGround'
    # camParamF = r'F:\WorkingCopy2\2020_05_31_DifferentiableRendererRealData\CameraParams\cam_params.json'
    # inTextureMeshFile = r"F:\WorkingCopy2\2021_01_16_OpticalFlowEvaluation\TemplateMesh\AA00006141.obj"
    #
    # inFileNames = ['AA00008102']
    # ext = 'ply'
    #
    # cfg = Config.RenderingCfg()
    # cfg.sigma = 1e-10
    # cfg.blurRange = 1e-10
    # cfg.imgSize = 1080
    # cfg.faces_per_pixel = 5
    #
    # cfg.numCams = 16
    # cfg.batchSize = 8
    # # cfg.cull_BackFaces = True
    # cfg.cull_BackFaces = False
    #
    # for frameName in tqdm.tqdm(inFileNames):
    #     processRenderFolder = join(inFolder, frameName)
    #     meshFile = join(inFolder, frameName + '.' + ext)
    #     os.makedirs(processRenderFolder, exist_ok=True)
    #     M05_Visualization.renderFrame(meshFile, inTextureMeshFile, camParamF,
    #                                   processRenderFolder,
    #                                   cfg=cfg, convertToM=True, rendererType='Silhouette')

    inFile = r'G:\2021_SyntheticImages\LadaGround\A06141.png'
    img = cv2.imread(inFile, cv2.IMREAD_UNCHANGED)
    print(img.shape)
    cv2.imshow('Alpha', img[:,:,3])
    cv2.waitKey()