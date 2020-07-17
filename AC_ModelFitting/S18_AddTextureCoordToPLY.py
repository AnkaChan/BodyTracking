from S17_VisualizeResults import *

if __name__ == '__main__':
    # inFolder = r'E:\WorkingCopy\2020_06_30_AC_ConsequtiveTexturedFitting2\Final'
    # texturedMesh = r"..\Data\TextureMap\SMPLWithSocks.obj"
    #
    # outFolderWithTextureCoord = join(inFolder, 'WithTextureCoord')
    # os.makedirs(outFolderWithTextureCoord, exist_ok=True)
    # plyFiles = glob.glob(join(inFolder, '*.ply'))
    # plyFiles.sort()
    #
    # meshWithTexture = pv.PolyData(texturedMesh)
    #
    # for plyFile in plyFiles:
    #     deformedMesh = pv.PolyData(plyFile)
    #     meshWithTexture.points = deformedMesh.points
    #     outFile = join(outFolderWithTextureCoord, os.path.basename(plyFile))
    #     meshWithTexture.save(outFile)

    inFolder = r'E:\WorkingCopy\2020_06_30_AC_ConsequtiveTexturedFitting2\FinalObj\WithTextureCoord'
    outFolder = r'E:\WorkingCopy\2020_06_30_AC_ConsequtiveTexturedFitting2\Final\WithTextureCoord'
    objFiles = glob.glob(join(inFolder, '*.obj'))

    for objFile in objFiles:
        deformedMesh = pv.PolyData(objFile)
        outFile = join(outFolder, os.path.basename(objFile) + '.ply')
        deformedMesh.save(outFile, binary=False)