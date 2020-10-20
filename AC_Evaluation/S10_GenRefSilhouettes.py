from M01_BackgroundSubtraction import *
import sys
sys.path.append('../AC_FitPipeline')
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from Utility_Rendering import *
import Config
import M05_Visualization

if __name__ == '__main__':
    preprocessedFolder = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada\Preprocessed'
    finalFolder = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada\Final\Mesh'
    inputCleanPlateFolder = r'F:\WorkingCopy2\2020_07_26_NewPipelineTestData\CleanPlateExtracted\RgbUndist'

    outFolder = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada\Silhouettes'
    # silhouette renderer parameters
    cleanPlateFolder = r'F:\WorkingCopy2\2020_07_26_NewPipelineTestData\CleanPlateExtracted\RgbUndist'
    inTextureMeshFile = r"..\Data\TextureMap2Color\Initial1Frame\SMPLWithSocks_tri.obj"
    camParamF = r'F:\WorkingCopy2\2020_05_31_DifferentiableRendererRealData\CameraParams\cam_params.json'

    cfg = Config.RenderingCfg()
    cfg.sigma = 1e-10
    cfg.blurRange = 1e-10
    cfg.imgSize = 1080
    cfg.faces_per_pixel = 5

    cfg.numCams = 16
    cfg.batchSize = 8

    frames = [str(i) for i in range(10459+135, 10459 + 300)]
    # frames = [str(i) for i in range(10459 + 20, 10459 + 200)]
    diffThres = 20

    cpFiles = sortedGlob(join(inputCleanPlateFolder, '*.png'))
    cpImgs = [cv2.imread(cpF) for cpF in cpFiles]

    for frameName in tqdm.tqdm(frames):
        frameFolder = join(preprocessedFolder, frameName)

        finalMesh = join(finalFolder, 'A' + frameName + '.ply')

        outFolderFrame = join(outFolder, frameName)
        os.makedirs(outFolderFrame, exist_ok=True)

        imgFs = sortedGlob(join(frameFolder, '*.png'))

        outFolderRefSils = join(outFolderFrame, 'RefSils')
        os.makedirs(outFolderRefSils, exist_ok=True)

        for imgF, imgCP in zip(imgFs, cpImgs):
            img = cv2.imread(imgF)
            refSilhouettes = foregroundSubtractionNaiveWithNoiseRemoval(img, imgCP, diffThres, cropCenterSquare=True)
            cv2.imwrite(join(outFolderRefSils, Path(imgF).stem + '.png'), refSilhouettes)

        outFolderRenderedSils = join(outFolderFrame, 'RenderedSils')

        os.makedirs(outFolderRenderedSils, exist_ok=True)
        M05_Visualization.renderFrame(finalMesh, inTextureMeshFile, camParamF,
                                                  outFolderRenderedSils,
                                                  cfg=cfg, convertToM=False, rendererType='Silhouette')


        outFoldeFinalSils = join(outFolderFrame, 'FinalSils')
        os.makedirs(outFoldeFinalSils, exist_ok=True)

        refSilFiles = sortedGlob(join(outFolderRefSils, '*.png'))
        renderedSilFiles = sortedGlob(join(outFolderRenderedSils, '*.png'))

        for refSilF, renderedSilF in zip(refSilFiles, renderedSilFiles):
            imgRef = cv2.imread(refSilF, cv2.IMREAD_GRAYSCALE)
            imgRendered = cv2.imread(renderedSilF, cv2.IMREAD_GRAYSCALE)

            imgRendered[np.where(imgRef)] = 255

            cv2.imwrite(join(outFoldeFinalSils, Path(refSilF).stem + '.png'), imgRendered)


