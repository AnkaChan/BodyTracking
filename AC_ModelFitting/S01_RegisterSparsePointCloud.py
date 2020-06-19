
from CGAL.CGAL_Kernel import Point_3
from CGAL.CGAL_Kernel import Triangle_3
from CGAL.CGAL_Kernel import Ray_3
from CGAL.CGAL_AABB_tree import AABB_tree_Triangle_3_soup
from numpy import cross, sum, isscalar, spacing, vstack
from numpy.core.umath_tests import inner1d

import json
import numpy as np
import pyvista as pv
import copy
from Utility import *
from os.path import join

def barycentric_coordinates_of_projection(p, q, u, v):
    """Given a point, gives projected coords of that point to a triangle
    in barycentric coordinates.
    See
        **Heidrich**, Computing the Barycentric Coordinates of a Projected Point, JGT 05
        at http://www.cs.ubc.ca/~heidrich/Papers/JGT.05.pdf
    :param p: point to project
    :param q: a vertex of the triangle to project into
    :param u,v: edges of the the triangle such that it has vertices ``q``, ``q+u``, ``q+v``
    :returns: barycentric coordinates of ``p``'s projection in triangle defined by ``q``, ``u``, ``v``
            vectorized so ``p``, ``q``, ``u``, ``v`` can all be ``3xN``
    """

    p = p.T
    q = q.T
    u = u.T
    v = v.T

    n = cross(u, v, axis=0)
    s = sum(n * n, axis=0)

    # If the triangle edges are collinear, cross-product is zero,
    # which makes "s" 0, which gives us divide by zero. So we
    # make the arbitrary choice to set s to epsv (=numpy.spacing(1)),
    # the closest thing to zero
    if isscalar(s):
        s = s if s else spacing(1)
    else:
        s[s == 0] = spacing(1)

    oneOver4ASquared = 1.0 / s
    w = p - q
    b2 = sum(cross(u, w, axis=0) * n, axis=0) * oneOver4ASquared
    b1 = sum(cross(w, v, axis=0) * n, axis=0) * oneOver4ASquared
    b = vstack((1 - b1 - b2, b1, b2))

    return b.T
#
# def VisualizeCorrs(outCorrFile, mesh1, mesh2, corrs12):
#
#     ptsVtk = vtk.vtkPoints()
#     # pts.InsertNextPoint(p1)
#     for corr in corrs12:
#         ptsVtk.InsertNextPoint(mesh1[corr[0], :])
#         ptsVtk.InsertNextPoint(mesh2[corr[1], :])
#
#     polyData = vtk.vtkPolyData()
#     polyData.SetPoints(ptsVtk)
#
#     lines = vtk.vtkCellArray()
#
#     for iL in range(corrs12.shape[0]):
#         line = vtk.vtkLine()
#
#         line.GetPointIds().SetId(0, iL*2)  # the second 0 is the index of the Origin in the vtkPoints
#         line.GetPointIds().SetId(1, iL*2 + 1)  # the second 1 is the index of P0 in the vtkPoints
#         lines.InsertNextCell(line)
#
#     polyData.SetLines(lines)
#     writer = vtk.vtkPolyDataWriter()
#     writer.SetInputData(polyData)
#     writer.SetFileName(outCorrFile)
#     writer.Update()


def pointsToTriangles(points,triangles):

    with np.errstate(all='ignore'):

        # Unpack triangle points
        p0,p1,p2 = np.asarray(triangles).swapaxes(0,1)

        # Calculate triangle edges
        e0 = p1-p0
        e1 = p2-p0
        a = inner1d(e0,e0)
        b = inner1d(e0,e1)
        c = inner1d(e1,e1)

        # Calculate determinant and denominator
        det = a*c - b*b
        invDet = 1. / det
        denom = a-2*b+c

        # Project to the edges
        p  = p0-points[:,np.newaxis]
        d = inner1d(e0,p)
        e = inner1d(e1,p)
        u = b*e - c*d
        v = b*d - a*e

        # Calculate numerators
        bd = b+d
        ce = c+e
        numer0 = (ce - bd) / denom
        numer1 = (c+e-b-d) / denom
        da = -d/a
        ec = -e/c


        # Vectorize test conditions
        m0 = u + v < det
        m1 = u < 0
        m2 = v < 0
        m3 = d < 0
        m4 = (a+d > b+e)
        m5 = ce > bd

        t0 =  m0 &  m1 &  m2 &  m3
        t1 =  m0 &  m1 &  m2 & ~m3
        t2 =  m0 &  m1 & ~m2
        t3 =  m0 & ~m1 &  m2
        t4 =  m0 & ~m1 & ~m2
        t5 = ~m0 &  m1 &  m5
        t6 = ~m0 &  m1 & ~m5
        t7 = ~m0 &  m2 &  m4
        t8 = ~m0 &  m2 & ~m4
        t9 = ~m0 & ~m1 & ~m2

        u = np.where(t0, np.clip(da, 0, 1), u)
        v = np.where(t0, 0, v)
        u = np.where(t1, 0, u)
        v = np.where(t1, 0, v)
        u = np.where(t2, 0, u)
        v = np.where(t2, np.clip(ec, 0, 1), v)
        u = np.where(t3, np.clip(da, 0, 1), u)
        v = np.where(t3, 0, v)
        u *= np.where(t4, invDet, 1)
        v *= np.where(t4, invDet, 1)
        u = np.where(t5, np.clip(numer0, 0, 1), u)
        v = np.where(t5, 1 - u, v)
        u = np.where(t6, 0, u)
        v = np.where(t6, 1, v)
        u = np.where(t7, np.clip(numer1, 0, 1), u)
        v = np.where(t7, 1-u, v)
        u = np.where(t8, 1, u)
        v = np.where(t8, 0, v)
        u = np.where(t9, np.clip(numer1, 0, 1), u)
        v = np.where(t9, 1-u, v)


        # Return closest points
        return (p0.T +  u[:, np.newaxis] * e0.T + v[:, np.newaxis] * e1.T).swapaxes(2,1)

# def pointToTriangles(points,triangles):
#     triangles = []
#
#     triangles.append(Triangle_3(a, b, c))
#     triangles.append(Triangle_3(a, b, d))
#     triangles.append(Triangle_3(a, d, c))
#
#     # constructs AABB tree
#     tree = AABB_tree_Triangle_3_soup(triangles)

def searchForClosestPoints(sourceVs, targetVs):
    closestPts = []

    for sv in sourceVs:
        minDst = 100000
        closestP = None
        # for tv in targetVs:
        #     dst = np.linalg.norm(sv - tv)
        #     if dst < minDst:
        #         minDst = dst
        #         closestP = tv

        dists = np.sum(np.square(targetVs - sv), axis=1)
        tvId = np.argmin(dists)

        closestPts.append(targetVs[tvId, :])
    return np.array(closestPts)

def searchForClosestPoints(sourceVs, targetVs, tree):
    closestPts = []
    dis = []
    for sv in sourceVs:

        d, tvId = tree.query(sv)
        closestPts.append(targetVs[tvId, :])
        dis.append(d)
    return np.array(closestPts), np.array(dis)

def toP(vNp):
    vNp = vNp.astype(np.float64)
    return Point_3(vNp[0], vNp[1], vNp[2])

def fromP(p3):
    return np.array([p3.x(), p3.y(), p3.z()])

def searchForClosestPointsOnTriangle(sourceVs, targetVs, targetFs):
    # triangles = np.array([[targetVs[f[0], :], targetVs[f[1], :], targetVs[f[2], :]] for f in targetFs])
    triangles = [Triangle_3(toP(targetVs[f[0], :]), toP(targetVs[f[1], :]), toP(targetVs[f[2], :])) for f in targetFs]
    tree = AABB_tree_Triangle_3_soup(triangles)
    closestPts = []
    for i, v in enumerate(sourceVs):
        #print("query for vertex: ", i)
        closestPts.append(fromP(tree.closest_point(toP(v))))
    return np.array(closestPts)

def searchForClosestPointsOnTriangleWithBarycentric(sourceVs, targetVs, targetFs):
    # triangles = np.array([[targetVs[f[0], :], targetVs[f[1], :], targetVs[f[2], :]] for f in targetFs])
    triangles = [Triangle_3(toP(targetVs[f[0], :]), toP(targetVs[f[1], :]), toP(targetVs[f[2], :])) for f in targetFs]
    tree = AABB_tree_Triangle_3_soup(triangles)
    closestPts = []
    trianglesId = []

    for i, v in enumerate(sourceVs):
        #print("query for vertex: ", i)
        # closestPts.append(fromP(tree.closest_point(toP(v))))
        p, id = tree.closest_point_and_primitive(toP(v))

        closestPts.append(fromP(p))
        trianglesId.append(id)

    barycentrics = []
    for tId, p in zip(trianglesId, closestPts):
        a = p
        t = targetFs[tId, :]
        tp1 = targetVs[t[0], :]
        u = targetVs[t[1], :] - tp1
        v = targetVs[t[2], :] - tp1
        c = barycentric_coordinates_of_projection(a, tp1, u, v)
        assert np.min(c) > -0.0001
        assert np.sum(c) >= 0.99999

        barycentrics.append(c[0, :])

    return np.array(closestPts), np.array(barycentrics), np.array(trianglesId)

if __name__ == '__main__':
    # inMeshFile = r'F:\WorkingCopy2\2020_05_31_DifferentiableRendererRealData\Output\RealDataSilhouette\HandHeadFix_Sig_1e-07_BR1e-07_Fpp15_NCams16ImS1080_LR0.4_LW1_NW1\FinalMesh.ply'
    # inMeshFile = r'F:\WorkingCopy2\2020_05_31_DifferentiableRendererRealData\Output\RealDataSilhouette\HandHeadFix_Sig_1e-07_BR1e-07_Fpp15_NCams16ImS1080_LR0.4_LW1_NW1\FinalMesh.obj'
    # inSparseInterpolatedMesh = r'F:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied\Deformed\SLap_SBiLap_True_TLap_0_JTW_5000_JBiLap_0_Step8_Overlap0\Deformed\A00003052.obj'
    # outInterpolatedFile = r'F:\WorkingCopy2\2020_05_31_DifferentiableRendererRealData\Output\RealDataSilhouette\HandHeadFix_Sig_1e-07_BR1e-07_Fpp15_NCams16ImS1080_LR0.4_LW1_NW1\InterpolatedWithSparse.ply'

    inMeshFile = r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\InitialFit\PersonalModel\FinalMesh.obj'
    inSparseInterpolatedMesh = r'F:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied\Deformed\SLap_SBiLap_True_TLap_0_JTW_5000_JBiLap_0_Step8_Overlap0\Deformed\A00003052.obj'
    outInterpolatedFile = r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\InitialFit\PersonalModel\InterpolatedWithSparse.ply'
    outFolder = r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\InitialFit\PersonalModel'
    skelDataFile = r'C:\Code\MyRepo\ChbCapture\06_Deformation\MeshInterpolation\06_SKelDataLadaWeightsMultiplierCorrectAnkle_1692.json'

    handIndicesFile = r'C:\Code\MyRepo\ChbCapture\06_Deformation\SMPL_Socks\HandIndices.json'
    HeadIndicesFile = r'C:\Code\MyRepo\ChbCapture\06_Deformation\SMPL_Socks\HeadIndices.json'
    softConstraintWeight = 100
    numRealCorners = 1487
    fixHandAndHead = True

    os.makedirs(outFolder, exist_ok=True)

    handIndices = json.load(open(handIndicesFile))
    headIndices = json.load(open(HeadIndicesFile))

    indicesToFix = copy.copy(handIndices)
    indicesToFix.extend(headIndices)

    deformedSMPLSH = pv.PolyData(inMeshFile)
    deformedSparseMesh = pv.PolyData(inSparseInterpolatedMesh)
    # deformedSparseMesh.points = deformedSparseMesh.points[:1487, :]

    smplshFaces = deformedSMPLSH.faces.reshape((-1, 4))[:, 1:]
    print(smplshFaces.shape)

    closestPtsNp, barys, trianglesId = searchForClosestPointsOnTriangleWithBarycentric(deformedSparseMesh.points, deformedSMPLSH.points,
                                                                                       smplshFaces)

    LNP = getLaplacian(inMeshFile)
    # LNP
    # Define fit cost to dense point cloud
    skelData = json.load(open(skelDataFile))
    coarseMeshPts = np.array(skelData['VTemplate'])
    validVertsOnRestpose = np.where(coarseMeshPts[2,:]!=-1)[0]

    obsIds = np.where(deformedSparseMesh.points[:, 2] >0)[0]

    constraintIds = np.intersect1d(obsIds, validVertsOnRestpose)
    # validTargets = targetVerts[constraintIds, :]

    intepolationMatrixNp = np.zeros((trianglesId.shape[0], deformedSMPLSH.points.shape[0]), dtype=np.float64)
    for iC in range(intepolationMatrixNp.shape[0]):
        if iC in constraintIds:
            intepolationMatrixNp[iC, smplshFaces[trianglesId[iC], 0]] = barys[iC, 0]
            intepolationMatrixNp[iC, smplshFaces[trianglesId[iC], 1]] = barys[iC, 1]
            intepolationMatrixNp[iC, smplshFaces[trianglesId[iC], 2]] = barys[iC, 2]
        else:
            trianglesId[iC] = -1

    np.save(join(outFolder, 'InterpolationMatrix.npy'), intepolationMatrixNp)

    smplshRestPoseVerts = np.array(deformedSMPLSH.points)

    registeredCorners = intepolationMatrixNp @ smplshRestPoseVerts
    registeredCornersMesh = pv.PolyData()
    registeredCornersMesh.points = registeredCorners
    registeredCornersMesh.save(join(outFolder, 'registeredCorners.ply'))


    # Deform to Sparse Point Cloud
    # only interpolate the points that is actually a corner
    constraintIds = constraintIds[np.where(constraintIds<numRealCorners)]

    targetPts = deformedSparseMesh.points[constraintIds, :]
    partialInterpolation = intepolationMatrixNp[constraintIds, :]
    displacement = targetPts - partialInterpolation @ smplshRestPoseVerts

    interpolatedPtsDisplacement = np.zeros(smplshRestPoseVerts.shape)
    nDimData = smplshRestPoseVerts.shape[0]

    fixHandMat = np.zeros((len(indicesToFix), smplshRestPoseVerts.shape[0]), dtype=np.float64)
    for iRow, handVId in enumerate(indicesToFix):
        fixHandMat[iRow, handVId] = 1

    handDisplacement = np.zeros((len(indicesToFix), 3))

    if fixHandAndHead:
        displacement = np.vstack([displacement, handDisplacement])
        partialInterpolation = np.vstack([partialInterpolation, fixHandMat])

    # displacement = 0*displacement + 10
    # We should interpolate displacement
    # Change this to soft soft constraint
    # nConstraints = constraintIds.shape[0]
    for iDim in range(3):
        x = displacement[:, iDim]
        # # Build Constraint
        D = partialInterpolation
        # e = np.zeros((nConstraints, 1))
        # for i, vId in enumerate(constraintIds):
        #     # D[i, vId] = 1
        #     e[i, 0] = x[i]
        #
        # kMat, KRes = buildKKT(LNP, D, e)
        kMat = LNP + softConstraintWeight * D.transpose() @ D
        KRes = softConstraintWeight * D.transpose() @ x
        xInterpo = np.linalg.solve(kMat, KRes)

        # print("Spatial Laplacian Energy:",  xInterpo[0:nDimX, 0].transpose() @ LNP @  xInterpo[0:nDimX, 0])
        # wI = xInterpo[0:nDimX, 0]
        # wI[nConstraints:] = 1
        # print("Spatial Laplacian Energy with noise:",  wI @ LNP @  wI)

        interpolatedPtsDisplacement[:, iDim] = xInterpo[0:nDimData]

    interpolatedVerts = smplshRestPoseVerts + interpolatedPtsDisplacement

    deformedSMPLSH.points = interpolatedVerts
    deformedSMPLSH.save(outInterpolatedFile)

    np.save(join(outFolder, 'InterpolationBarys.npy'), barys)
    np.save(join(outFolder, 'InterpolationTriId.npy'), trianglesId)
    np.save(join(outFolder, 'InterpolationDisplacement.npy'), displacement)
