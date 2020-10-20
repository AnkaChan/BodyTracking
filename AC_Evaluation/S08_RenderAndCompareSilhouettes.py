from M03_SIlhouetteComparison import *

if __name__ == '__main__':
    inFramesFolder = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada\Evaluation\ToKpAndDense\All'
    refSilF = r'C:\Code\MyRepo\03_capture\Mocap-CVPR-Paper-Figures\06_SilhouetteComparison\10459\Sil_K10459.png'
    meshExt = 'obj'
    outFolder = join(inFramesFolder, 'Rendered')

    renderSilhouettes(inFramesFolder, outFolder, meshExt, convertToM=True)

    renderedSilF= r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada\Evaluation\ToKpAndDense\All\Rendered\K\Rendered\10459.png'
    refSilF = r'C:\Code\MyRepo\03_capture\Mocap-CVPR-Paper-Figures\06_SilhouetteComparison\10459\Sil_K10459.png'

    outFile = r'C:\Code\MyRepo\03_capture\Mocap-CVPR-Paper-Figures\06_SilhouetteComparison\10459\Overlay_Comparison.png'

    overlaySils(refSilF, renderedSilF, outFile)