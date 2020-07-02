import pyvista as pv
from scipy.spatial.transform import Rotation as R
import sys
SMPLSH_Dir = r'..\SMPL_reimp'
sys.path.insert(0, SMPLSH_Dir)
import smplsh_tf as smplsh_model
import smplsh_np

# import trimesh
from scipy.spatial import KDTree
import numpy as np
from iglhelpers import *
import pyigl as igl
import tqdm, os, json
from os.path import join
import tensorflow as tf

import vtk
from pathlib import Path
from SkelFit import Visualization

from CGAL.CGAL_Kernel import Point_3
from CGAL.CGAL_Kernel import Triangle_3
from CGAL.CGAL_Kernel import Ray_3
from CGAL.CGAL_AABB_tree import AABB_tree_Triangle_3_soup
from numpy import cross, sum, isscalar, spacing, vstack
from numpy.core.umath_tests import inner1d
from SkelFit import Data

import json

class Config:
    def __init__(s):
        s.numComputeClosest = 5
        s.numIterEachClosestSet = 500
        s.jointRegularizerWeight = 1e-6
        # jointRegularizerWeight = 0.0000001
        # jointRegularizerWeight = 0
        s.learnrate_ph = 0.05
        s.lrDecayStep = 200
        s.lrDecayRate = 0.97

        s.noBodyKeyJoint = True
        s.numBodyJoint = 22 - 9

        s.numIterFitting = 6000
        s.printStep = 500

        s.numPtsSmplshMesh = 6750
        s.indicesVertsToOptimize = list(range(s.numPtsSmplshMesh))

        s.keypointFitWeightInToDenseICP = 1
        # keypointFitWeightInToDenseICP = 0.0
        # constantBeta = False
        s.constantBeta = True
        s.betaRegularizerWeightToKP = 0
        s.manualCorrsWeightToKP = 1

        # maxDistanceToClosestPt = 30
        s.maxDistanceToClosestPt = 0.05

        # withDensePointCloud = True
        s.withDensePointCloud = False
        s.densePointCloudWeight = 1
        s.numRealCorners = 1487

        s.skeletonJointsToFix = [10, 11]


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

def VisualizeCorrs(outCorrFile, mesh1, mesh2, corrs12):

    ptsVtk = vtk.vtkPoints()
    # pts.InsertNextPoint(p1)
    for corr in corrs12:
        ptsVtk.InsertNextPoint(mesh1[corr[0], :])
        ptsVtk.InsertNextPoint(mesh2[corr[1], :])

    polyData = vtk.vtkPolyData()
    polyData.SetPoints(ptsVtk)

    lines = vtk.vtkCellArray()

    for iL in range(corrs12.shape[0]):
        line = vtk.vtkLine()

        line.GetPointIds().SetId(0, iL*2)  # the second 0 is the index of the Origin in the vtkPoints
        line.GetPointIds().SetId(1, iL*2 + 1)  # the second 1 is the index of P0 in the vtkPoints
        lines.InsertNextCell(line)

    polyData.SetLines(lines)
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(polyData)
    writer.SetFileName(outCorrFile)
    writer.Update()


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

class SMPLSH:
    def __init__(self, betas = None, pose = None, trans=None, constantBeta = False, personalShape=None, constantPersonalShape=True,
                 SMPLSHNpzFile=r'..\SMPL_reimp\SmplshModel_m.npz'):

        if constantBeta:
            self.betas = tf.constant(betas, name='betas', dtype=tf.float64)
        else:
            if betas is None:
                self.betas = tf.get_variable("betas", shape=[10], initializer=tf.initializers.zeros(), dtype=tf.float64)
            else:
                self.betas = tf.get_variable("betas", initializer=betas, dtype=tf.float64)

        if trans is None:
            self.trans = tf.get_variable("trans", shape=[3], initializer=tf.initializers.zeros(), dtype=tf.float64)
        else:
            self.trans = tf.get_variable("trans", initializer=trans, dtype=tf.float64)

        if pose is None:
            self.pose = tf.get_variable("pose", shape=[3*52], initializer=tf.initializers.zeros(), dtype=tf.float64)
        else:
            self.pose = tf.get_variable("pose", initializer=pose, dtype=tf.float64)

        if personalShape is not None:
            if constantPersonalShape:
                self.personalShape = tf.constant(personalShape, name="personalShape", dtype=tf.float64)
            else:
                self.personalShape = tf.get_variable("personalShape", initializer=personalShape, dtype=tf.float64)

        # self.smplVerts, smplFaces =  smplsh_np.SMPLSHModel(SMPLSHNpzFile)(model_path, betas, pose, trans, False)
        self.smplVerts, self.smplFaces = smplsh_model.smplsh_model(SMPLSHNpzFile, self.betas, self.pose, self.trans,
                                            personalShape = personalShape, unitMM=False, )

        self.skeletonJointsToFix = []


def toSparseFitting(dataFolder, objFile, outFolder, skelDataFile, toSparsePointCloudInterpoMatFile,
                    betaFile, personalShapeFile, OP2AdamJointMatFile, AdamGoodJointsFile, smplsh2OPRegressorMatFile,
                    smplshDataFile, cfg=Config()):
    tf.reset_default_graph()
    OP2AdamJointMat = np.load(OP2AdamJointMatFile)
    AdamGoodJoints = np.load(AdamGoodJointsFile)
    smplsh2OPRegressorMat = np.load(smplsh2OPRegressorMatFile)

    betas = np.load(betaFile)
    trans = np.array([0, 0, 0], dtype=np.float64)

    personalShape = np.load(personalShapeFile) / 1000

    smplshtf = SMPLSH(trans=trans, betas=betas, constantBeta=cfg.constantBeta, personalShape=personalShape, SMPLSHNpzFile=smplshDataFile)
    smplshtf.skeletonJointsToFixToDense = []
    # smplshtf = SMPLSH(betas, pose, trans)

    outFolderFittingParam = join(outFolder, 'FittingParams'
                                 )

    os.makedirs(outFolder, exist_ok=True)
    os.makedirs(outFolderFittingParam, exist_ok=True)

    smplFaces = np.array(smplshtf.smplFaces)

    sess = tf.Session()

    regressedJoints = tf.matmul(smplsh2OPRegressorMat, smplshtf.smplVerts)

    targetKeypointsPH = tf.placeholder(dtype=np.float64, shape=(OP2AdamJointMat.shape[0], 3), name="targetKeypointsPH")

    # Remove the cost for unobserved key points
    regressedJoints = tf.matmul(smplsh2OPRegressorMat, smplshtf.smplVerts)
    if cfg.noBodyKeyJoint:
        keypointFitCost = tf.reduce_mean(tf.square(
            tf.multiply(
                tf.nn.relu(tf.sign(targetKeypointsPH[:, 2:3])),
                tf.gather(regressedJoints, AdamGoodJoints) - targetKeypointsPH
            )[cfg.numBodyJoint:, :]
        ))
    else:
        keypointFitCost = tf.reduce_mean(tf.square(
            tf.multiply(
                tf.nn.relu(tf.sign(targetKeypointsPH[:, 2:3])),
                tf.gather(regressedJoints, AdamGoodJoints) - targetKeypointsPH
            )
        ))

    betaRegularizerCostToKp = cfg.betaRegularizerWeightToKP * tf.reduce_sum(tf.square(smplshtf.betas - betas))
    regularizerCostToKp = cfg.jointRegularizerWeight * tf.reduce_sum(tf.square(smplshtf.pose))

    # Define fit cost to dense point cloud
    skelData = json.load(open(skelDataFile))
    coarseMeshPts = np.array(skelData['VTemplate'])
    validVertsOnRestpose = np.where(coarseMeshPts[2, :] != -1)[0]

    # obsIds = np.where(targetVertsPH[:, 2] >0)[0]
    skelFixCost = 0
    for iJoint in cfg.skeletonJointsToFix:
        skelFixCost = skelFixCost + 100 * tf.reduce_mean(tf.square(smplshtf.pose[(iJoint * 3):(iJoint * 3 + 3)]))

    # closestPtsSet = tf.placeholder(dtype=np.float64, shape=targetVerts.shape, name="closestPtsSet")
    # intepolationMatrix = tf.placeholder(dtype=np.float64, shape=(coarseMeshPts.shape[0], numPtsSmplshMesh), name="intepolationMatrix")
    intepolationMatrixNp = np.load(toSparsePointCloudInterpoMatFile)
    intepolationMatrix = tf.constant(intepolationMatrixNp, dtype=np.float64, name="intepolationMatrix")
    targetVertsPH = tf.placeholder(dtype=np.float64, shape=(intepolationMatrixNp.shape[0], 3), name="targetVertsPH")

    # costICP = tf.reduce_mean(
    #     tf.square(tf.gather(targetVerts, obsIds) - tf.gather(tf.matmul(intepolationMatrix, smplVerts), obsIds)))
    constraintIdsPH = tf.placeholder(tf.int32, shape=None)
    costICPToSparse = tf.reduce_mean(
        tf.square(tf.gather(targetVertsPH - tf.matmul(intepolationMatrix, smplshtf.smplVerts), constraintIdsPH)))

    costICPToSparse = costICPToSparse + cfg.keypointFitWeightInToDenseICP * keypointFitCost + regularizerCostToKp
    costICPToSparse = costICPToSparse + skelFixCost
    # cost of fit to dense
    closestPtsSetDense = tf.placeholder(dtype=np.float64, shape=smplshtf.smplVerts.shape, name="closestPtsSet")
    fitCostToDense = tf.reduce_mean(tf.square(
        tf.multiply(
            tf.nn.relu(tf.sign(closestPtsSetDense[:, 2:3])),
            tf.gather(smplshtf.smplVerts, cfg.indicesVertsToOptimize) - tf.gather(closestPtsSetDense, cfg.indicesVertsToOptimize)
        )))

    if cfg.withDensePointCloud:
        costICPToSparse = costICPToSparse + cfg.densePointCloudWeight * fitCostToDense

    stepICPToSparse = tf.Variable(0, trainable=False)
    rateICPToSparse = tf.train.exponential_decay(cfg.learnrate_ph, stepICPToSparse, cfg.lrDecayStep, cfg.lrDecayRate)
    train_step_ICPToSparse = tf.train.AdamOptimizer(learning_rate=rateICPToSparse).minimize(costICPToSparse,
                                                                                            global_step=stepICPToSparse)
    init = tf.global_variables_initializer()

    # targetMeshFile = glob.glob(join(dataFolder, '*.obj'))[0]
    targetVerts = np.array(pv.PolyData(objFile).points, np.float64) / 1000
    # targetVerts = np.vstack([targetVerts, np.repeat([[0,0,-1]], 1692-targetVerts.shape[0], axis=0)])
    # targetVerts = targetVerts[:numRealCorners, :]
    obsIds = np.where(targetVerts[:, 2] > 0)[0]
    constraintIds = np.intersect1d(obsIds, validVertsOnRestpose)
    validTargets = targetVerts[constraintIds, :]

    inputKeypoints = join(dataFolder, r'ToRGB\Reconstruction\PointCloud.obj')
    inputDensePointCloudFile = join(dataFolder, 'scene_dense.ply')

    targetKeypointsOP = np.array(pv.PolyData(inputKeypoints).points).astype(np.float64) / 1000
    targetKeypoints = (OP2AdamJointMat @ targetKeypointsOP).astype(np.float64)
    for iKp in range(OP2AdamJointMat.shape[0]):
        relevantKpIds = np.where(OP2AdamJointMat[iKp, :])[0]
        if np.any(targetKeypointsOP[relevantKpIds, 2] < 0):
            targetKeypoints[iKp, :] = [0, 0, -1]

    if cfg.withDensePointCloud:
        densePointCloud = np.array(pv.PolyData(inputDensePointCloudFile).points).astype(np.float64) / 1000
        tree = KDTree(densePointCloud)
    sess.run(init)

    # Nonrigid ICP Procedure
    # print("Fit to sparse point clouds, initialCost:", sess.run(costICPToSparse))
    feedDict = {targetKeypointsPH: targetKeypoints, targetVertsPH: targetVerts, constraintIdsPH: constraintIds}
    for i in range(cfg.numIterFitting):
        # sess.run(costICPToSparse)
        sess.run(train_step_ICPToSparse, feed_dict=feedDict)
        if not i % cfg.printStep:
            print("Cost:", sess.run(costICPToSparse, feed_dict=feedDict),
                  "keypointFitCost:", sess.run(keypointFitCost, feed_dict=feedDict),
                  'regularizerCostToKp:', sess.run(regularizerCostToKp, feed_dict=feedDict),
                  'LearningRate:', sess.run(rateICPToSparse, feed_dict=feedDict))

    transVal = sess.run(smplshtf.trans)
    poseVal = sess.run(smplshtf.pose)
    betaVal = sess.run(smplshtf.betas)

    outParamFile = join(outFolderFittingParam, Path(dataFolder).stem + '.npz')
    np.savez(outParamFile, trans=transVal, pose=poseVal, beta=betaVal)

    outFile = join(outFolder, Path(dataFolder).stem + '.obj')
    Data.write_obj(outFile, sess.run(smplshtf.smplVerts) * 1000, smplFaces)



