import sys
sys.path.append('../AC_FitPipeline')
import os
from Utility_Rendering import *
import cv2
import numpy as np

def overlaySils(refSilF, renderedSilF, outFile):
    _, refSil = load_image(refSilF)

    renderedSil = cv2.imread(renderedSilF)

    overlay = np.zeros(renderedSil.shape, dtype=renderedSil.dtype)

    overlay[:, :, 1] = refSil[:, :, 1]
    overlay[:, :, 2] = renderedSil[:, :, 2]

    # cv2.imshow('overlay', overlay)
    # cv2.waitKey()
    cv2.imwrite(outFile, overlay)

if __name__ == '__main__':
    # refSilF = r'C:\Code\MyRepo\03_capture\Mocap-CVPR-Paper-Figures\06_SilhouetteComparison\10459\Sil_K10459.png'
    # renderedSilF = r'C:\Code\MyRepo\03_capture\Mocap-CVPR-Paper-Figures\06_SilhouetteComparison\10459\Silhouettes_ToTP\K\Rendered\10459.png'
    # renderedSilF = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada\Evaluation\Interpolated\Silhouettes\K\Rendered\InterpolatedMesh.png'
    # outFile = r'C:\Code\MyRepo\03_capture\Mocap-CVPR-Paper-Figures\06_SilhouetteComparison\10459\Overlay_Interpo.png'
    # renderedSilF = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada\Evaluation\ImageBasedFitting\Silhouettes\K\Rendered\A10459.png'
    # outFile = r'C:\Code\MyRepo\03_capture\Mocap-CVPR-Paper-Figures\06_SilhouetteComparison\10459\Overlay_Final.png'

    renderedSilF = r'C:\Code\MyRepo\03_capture\Mocap-CVPR-Paper-Figures\04_SIlhouetteOptimization\Data\Mesh\Silhouettes\P\Rendered\Fit0_Initial.png'
    refSilF = r'F:\WorkingCopy2\2020_08_27_KateyBodyModel\Silhouettes\Sihouettes_NoGlassese\18411\Undist\P18411.png'
    outFile = r'C:\Code\MyRepo\03_capture\Mocap-CVPR-Paper-Figures\04_SIlhouetteOptimization\Fit0_Initial.png'

    # renderedSilF = r'C:\Code\MyRepo\03_capture\Mocap-CVPR-Paper-Figures\04_SIlhouetteOptimization\Data\Mesh\Silhouettes\P\Rendered\PerVertex_Fit00499.png'
    # refSilF = r'F:\WorkingCopy2\2020_08_27_KateyBodyModel\Silhouettes\Sihouettes_NoGlassese\18411\Undist\P18411.png'
    # outFile = r'C:\Code\MyRepo\03_capture\Mocap-CVPR-Paper-Figures\04_SIlhouetteOptimization\PerVertex_Fit00499.png'


    overlaySils(refSilF, renderedSilF, outFile)