import pyvista as pv
import numpy as np
import itertools, os

def searchQuadIds(qCode, codeSet):
    qvIds = [-1, -1, -1, -1]

    for iV, cCode in enumerate(codeSet):
        ids = [i for i, x in enumerate(cCode['Code']) if x == qCode]
        if len(ids) != 0:
            assert  len(ids) == 1
            qvIds[cCode['Id'][ids[0]]] = iV

    return qvIds

if __name__ == '__main__':
    inCompleteMesh = r'C:\Code\MyRepo\03_capture\BodyTracking\Data\2020_12_27_betterCoarseMesh\Mesh1487\A00003052.obj'
    inOriginalRestPoseQuadMesh = r'F:\WorkingCopy2\2020_01_16_KM_Edited_Meshes\LadaFinalMesh_edited.obj'

    outComplemeshQuad = r'../Data/2020_12_27_betterCoarseMesh/Mesh1487/Complete_QuadOnly.obj'
    outRealPtsOnlyhMeshFile = r'../Data/2020_12_27_betterCoarseMesh/Mesh1487/cleared_RealPtsOnly.ply'
    outQuadOnlyhMeshFile = r'../Data/2020_12_27_betterCoarseMesh/Mesh1487/cleared_QuadOnly.ply'
    exampleQuadFIle = r'F:\WorkingCopy2\2019_12_27_FinalLadaMesh\FinalMesh2_OnlyQuad\Mesh2_OnlyQua.obj'
    cIdFile = r'C:\Code\MyRepo\ChbCapture\04_Pipeline\GenerateModelSequenceMesh7\CID_no_meshVID.txt'

    numRealPts = 1487

    # read the complete quad mesh and struture the faces
    meshWithFaces = pv.PolyData(inOriginalRestPoseQuadMesh)
    faces = []
    fId = 0
    while fId < meshWithFaces.faces.shape[0]:
        numFVs = meshWithFaces.faces[fId]
        face = []
        fId += 1
        for i in range(numFVs):
            face.append(meshWithFaces.faces[fId])
            fId += 1

        faces.append(face)

    faceIdToPreserve = []
    pts = meshWithFaces.points

    # read the quad only mesh and structure the faces
    meshWithOnlyQuadFaces = pv.PolyData(exampleQuadFIle)
    quadFaces = []
    fId = 0
    while fId < meshWithOnlyQuadFaces.faces.shape[0]:
        numFVs = meshWithOnlyQuadFaces.faces[fId]
        face = []
        fId += 1
        for i in range(numFVs):
            face.append(meshWithOnlyQuadFaces.faces[fId])
            fId += 1

        quadFaces.append(set(face))

    faceIdToPreserve = []
    pts = meshWithOnlyQuadFaces.points

    inMesh = pv.PolyData(inCompleteMesh)

    # faces = meshWithFaces.faces.reshape(-1, 4)
    for i in range(len(faces)):
        vertsObserved = [iV < numRealPts and pts[iV][2] != -1 for iV in faces[i]]
        # if pts[faces[i][1]][2] != -1 and pts[faces[i][2]][2] != -1 and pts[faces[i][3]][2] != -1:
        if np.all(vertsObserved):
            faceIdToPreserve.append(i)
    facesToPreserve = [faces[iF] for iF in faceIdToPreserve]
    flattenFaces = []
    for face in facesToPreserve:
        flattenFaces.extend([len(face), *face])

    inMesh.faces = np.array(flattenFaces, dtype=np.int64)
    inMesh.save(outRealPtsOnlyhMeshFile)

    # write the complete mesh with only quad, mark the added face with blue
    with open(outComplemeshQuad, 'w') as fp:
        fp.write('mtllib ./' + os.path.basename(outComplemeshQuad) +'.mtl\n')
        for v in inMesh.points:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        fp.write('usemtl material_0\n')
        fp.write('vt 0.01 0.01\n') #1~4 blue
        fp.write('vt 0.01 0.49\n')
        fp.write('vt 0.49 0.49\n')
        fp.write('vt 0.49 0.01\n')

        fp.write('vt 0.51 0.51\n') #5~8 white
        fp.write('vt 0.51 0.99\n')
        fp.write('vt 0.99 0.99\n')
        fp.write('vt 0.99 0.51\n')

        for iF,f in enumerate(faces):
            fp.write('f ')
            iT = 1
            for fVId in f:
                if set(f) in quadFaces:
                    fp.write('%d/%d ' % (fVId + 1, iT))
                else:
                    fp.write('%d/%d ' % (fVId + 1, iT+4))

                iT += 1
            fp.write('\n')


    # read the mesh and retrieve the quad structure from the CID file
    inMesh.faces = pv.PolyData(exampleQuadFIle).faces
    inMesh.save(outQuadOnlyhMeshFile)