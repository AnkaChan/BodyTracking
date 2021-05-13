from S02_RecoverDetails import *

if __name__ == '__main__':
    inChunkFile = r'F:\WorkingCopy2\2020_01_13_FinalAnimations\Katey_NewPipeline\LongSequence\LongSequence_0_2800.clean.json'
    outputFolder = r'F:\WorkingCopy2\2021_01_21_DataToSubmit\Raw3DReconstruction\Female1'

    inChunkFile=r''

    inCompleteMesh =  r'F:\WorkingCopy2\2020_01_16_KM_Edited_Meshes\LadaFinalMesh_edited.obj'


    unpackChunkData(inChunkFile, join(outputFolder, 'PointsOnly'), outputType='ply')
    fittingToVtk(join(outputFolder, 'PointsOnly'), outVTKFolder=outputFolder, extName='ply', outExtName='ply', removeUnobservedFaces=True,
                 meshWithFaces=inCompleteMesh)