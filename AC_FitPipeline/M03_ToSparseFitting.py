import pyvista as pv
from scipy.spatial.transform import Rotation as R
import sys
SMPLSH_Dir = r'..\SMPL_reimp'
sys.path.insert(0, SMPLSH_Dir)
import smplsh_tf as smplsh_model
import smplsh_np
import smplsh_torch

# import trimesh
from scipy.spatial import KDTree
import numpy as np
from iglhelpers import *
import tqdm, os, json
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

SMPLSH_Dir = r'..\SMPL_reimp'
import sys, glob
sys.path.insert(0, SMPLSH_Dir)

import smplsh_tf as smplsh_model
import smplsh_np as smplsh_np
import glob
import SkelFit.Visualization as Visualization
import pickle
import tensorflow as tf

from CGAL.CGAL_Kernel import Point_3
from CGAL.CGAL_Kernel import Triangle_3
from CGAL.CGAL_Kernel import Ray_3
from CGAL.CGAL_AABB_tree import AABB_tree_Triangle_3_soup
from numpy import cross, sum, isscalar, spacing, vstack
from numpy.core.umath_tests import inner1d
from SkelFit import Data
import numpy as np
import os
import json
import pyvista as pv
from os.path import join
from scipy.spatial import KDTree
import tqdm, copy

import torch

from Utility import getLaplacian
from matplotlib import pyplot as plt

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
vertex_ids = {
    'smplh': {
        'nose':		    332,
        'reye':		    6260,
        'leye':		    2800,
        'rear':		    4071,
        'lear':		    583,
        'rthumb':		6191,
        'rindex':		5782,
        'rmiddle':		5905,
        'rring':		6016,
        'rpinky':		6133,
        'lthumb':		2746,
        'lindex':		2319,
        'lmiddle':		2445,
        'lring':		2556,
        'lpinky':		2673,
        'LBigToe':		3216,
        'LSmallToe':	3226,
        'LHeel':		3387,
        'RBigToe':		6617,
        'RSmallToe':    6624,
        'RHeel':		6787
    },
    'smplx': {
        'nose':		    9120,
        'reye':		    9929,
        'leye':		    9448,
        'rear':		    616,
        'lear':		    6,
        'rthumb':		8079,
        'rindex':		7669,
        'rmiddle':		7794,
        'rring':		7905,
        'rpinky':		8022,
        'lthumb':		5361,
        'lindex':		4933,
        'lmiddle':		5058,
        'lring':		5169,
        'lpinky':		5286,
        'LBigToe':		5770,
        'LSmallToe':    5780,
        'LHeel':		8846,
        'RBigToe':		8463,
        'RSmallToe': 	8474,
        'RHeel':  		8635
    },
    'smplsh' : {
        "nose": 332,
        "reye": 6189,
        "leye": 2800,
        "rear": 4000,
        "lear": 583,
        "rthumb": 6120,
        "rindex": 5711,
        "rmiddle": 5834,
        "rring": 5945,
        "rpinky": 6062,
        "lthumb": 2746,
        "lindex": 2319,
        "lmiddle": 2445,
        "lring": 2556,
        "lpinky": 2673,
        "LBigToe": 3212,
        "LSmallToe": 3222,
        "LHeel": 3316,
        "RBigToe": 6747,
        "RSmallToe": 6737,
        "RHeel": 6622
        }
}
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
    def __init__(self, betas = None, pose = None, trans=None, constantBeta = False, SMPLSHNpzFile=r'SMPLSH\SmplshModel.npz'):

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

        # self.smplVerts, smplFaces =  smplsh_np.SMPLSHModel(SMPLSHNpzFile)(model_path, betas, pose, trans, False)
        self.smplVerts, self.smplJoints, self.smplFaces = smplsh_model.smplsh_model(SMPLSHNpzFile, self.betas, self.pose, self.trans, returnDeformedJoints=True)

        self.skeletonJointsToFix = []

def loadCompressedFittingParam(file, readPersonalShape=False):
    fitParam = np.load(file)
    transInit = fitParam['trans']
    poseInit = fitParam['pose']
    betaInit = fitParam['beta']

    if readPersonalShape:
        personalShape = fitParam['personalShape']
        return transInit, poseInit, betaInit, personalShape
    else:
        return transInit, poseInit, betaInit

def smpl_to_openpose(model_type='smplx', use_hands=True, use_face=True,
                     use_face_contour=False, openpose_format='coco25'):
    ''' Returns the indices of the permutation that maps OpenPose to SMPL

        Parameters
        ----------
        model_type: str, optional
            The type of SMPL-like model that is used. The default mapping
            returned is for the SMPLX model
        use_hands: bool, optional
            Flag for adding to the returned permutation the mapping for the
            hand keypoints. Defaults to True
        use_face: bool, optional
            Flag for adding to the returned permutation the mapping for the
            face keypoints. Defaults to True
        use_face_contour: bool, optional
            Flag for appending the facial contour keypoints. Defaults to False
        openpose_format: bool, optional
            The output format of OpenPose. For now only COCO-25 and COCO-19 is
            supported. Defaults to 'coco25'

    '''
    if openpose_format.lower() == 'coco25':
        if model_type == 'smpl':
            return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                             7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56, 57, 58, 59,
                                     60, 61, 62], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 63, 22, 23, 24, 64,
                                          25, 26, 27, 65, 31, 32, 33, 66, 28,
                                          29, 30, 67], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 68, 37, 38, 39, 69,
                                          40, 41, 42, 70, 46, 47, 48, 71, 43,
                                          44, 45, 72], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == 'smplx':
            body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62,
                                     63, 64, 65], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 66, 25, 26, 27,
                                          67, 28, 29, 30, 68, 34, 35, 36, 69,
                                          31, 32, 33, 70], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 71, 40, 41, 42, 72,
                                          43, 44, 45, 73, 49, 50, 51, 74, 46,
                                          47, 48, 75], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                #  end_idx = 127 + 17 * use_face_contour
                face_mapping = np.arange(76, 127 + 17 * use_face_contour,
                                         dtype=np.int32)
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
    elif openpose_format == 'coco19':
        if model_type == 'smpl':
            return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8,
                             1, 4, 7, 25, 26, 27, 28],
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 57, 22, 23, 24, 58,
                                          25, 26, 27, 59, 31, 32, 33, 60, 28,
                                          29, 30, 61], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 62, 37, 38, 39, 63,
                                          40, 41, 42, 64, 46, 47, 48, 65, 43,
                                          44, 45, 66], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == 'smplx':
            body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 60, 25, 26, 27,
                                          61, 28, 29, 30, 62, 34, 35, 36, 63,
                                          31, 32, 33, 64], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 65, 40, 41, 42, 66,
                                          43, 44, 45, 67, 49, 50, 51, 68, 46,
                                          47, 48, 69], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                face_mapping = np.arange(70, 70 + 51 +
                                         17 * use_face_contour,
                                         dtype=np.int32)
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
    else:
        raise ValueError('Unknown joint format: {}'.format(openpose_format))

class VertexJointSelector:

    def __init__(self, vertex_ids=None,
                 use_hands=True,
                 use_feet_keypoints=True, **kwargs):
        extra_joints_idxs = []

        face_keyp_idxs = np.array([
            vertex_ids['nose'],
            vertex_ids['reye'],
            vertex_ids['leye'],
            vertex_ids['rear'],
            vertex_ids['lear']], dtype=np.int64)

        extra_joints_idxs = np.concatenate([extra_joints_idxs,
                                            face_keyp_idxs])

        if use_feet_keypoints:
            feet_keyp_idxs = np.array([vertex_ids['LBigToe'],
                                       vertex_ids['LSmallToe'],
                                       vertex_ids['LHeel'],
                                       vertex_ids['RBigToe'],
                                       vertex_ids['RSmallToe'],
                                       vertex_ids['RHeel']], dtype=np.int32)

            extra_joints_idxs = np.concatenate(
                [extra_joints_idxs, feet_keyp_idxs])

        if use_hands:
            self.tip_names = ['thumb', 'index', 'middle', 'ring', 'pinky']

            tips_idxs = []
            for hand_id in ['l', 'r']:
                for tip_name in self.tip_names:
                    tips_idxs.append(vertex_ids[hand_id + tip_name])

            self.extra_joints_idxs = np.concatenate(
                [extra_joints_idxs, tips_idxs]).astype(np.int64)

    def __call__(self, vertices, joints):
        extra_joints = tf.gather(vertices, self.extra_joints_idxs, axis=1)
        joints = tf.concat([joints, extra_joints], axis=1)

        return joints

class JointMapper:
    def __init__(self, joint_maps=None):
        self.joint_maps = joint_maps

    def __call__(self, joints, **kwargs):
        if self.joint_maps is None:
            return joints
        else:
            return tf.gather(joints, self.joint_maps, axis=1)

class VertexToOpJointsConverter:
    def __init__(s, **kwargs):
        s.jSelector = VertexJointSelector(vertex_ids['smplsh'])
        jointMap = smpl_to_openpose('smplh', use_hands=True,
                                    use_face=False,
                                    use_face_contour=False, )

        s.joint_mapper = JointMapper(jointMap)

    def __call__(s, smplshVerts, smplshJoints):
        allJoints = s.jSelector(smplshVerts, smplshJoints)
        joint_mapped = s.joint_mapper(allJoints)

        return joint_mapped


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
        s.noHandAndHead = False
        # s.numBodyJoint = 22 - 9
        s.numBodyJoint = 25
        s.headJointsId = [0, 15, 16, 17, 18]
        s.withFaceKp = False

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

        s.terminateLoss = 1.0e-4

        s.outputErrs = False
        s.terminateLossStep = 1e-7
        # s.terminateLossStepAvgRange = 5


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
        # self.smplVerts, self.smplFaces = smplsh_model.smplsh_model(SMPLSHNpzFile, self.betas, self.pose, self.trans,
        #                                     personalShape = personalShape, unitMM=False, )
        self.smplVerts, self.smplJoints, self.smplFaces = smplsh_model.smplsh_model(SMPLSHNpzFile, self.betas, self.pose,
                                                                        self.trans,personalShape = personalShape, unitMM=False, returnDeformedJoints=True)


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
    obsIds = np.where(targetVerts[:cfg.numRealCorners, 2] > 0)[0]
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
        if  sess.run(costICPToSparse, feed_dict=feedDict) < cfg.terminateLoss:
            break

    transVal = sess.run(smplshtf.trans)
    poseVal = sess.run(smplshtf.pose)
    betaVal = sess.run(smplshtf.betas)

    outParamFile = join(outFolderFittingParam, Path(dataFolder).stem + '.npz')
    np.savez(outParamFile, trans=transVal, pose=poseVal, beta=betaVal)

    outFile = join(outFolder, Path(dataFolder).stem + '.obj')
    Data.write_obj(outFile, sess.run(smplshtf.smplVerts) * 1000, smplFaces)

def faceKpLosstf(smplshVerts, keypoints):
    corrs =np.array( [
        [75, 3161],  # middle chin
        [115, 3510],  # right mouth corner
        [121, 69],  # left mouth corner
        [74, 3544],
        [76, 102],

    ])
    return tf.reduce_mean(tf.multiply(
        tf.nn.relu(tf.sign(keypoints[corrs[:,0], 2:3])),
        (tf.gather(smplshVerts, corrs[:,1]) - keypoints[corrs[:,0]])**2))


def toSparseFittingNewRegressor(inputKeypoints, sparsePCObjFile, outFolder, skelDataFile, toSparsePointCloudInterpoMatFile,
                    betaFile, personalShapeFile, smplshDataFile, inputDensePointCloudFile=None, initialPoseFile=None, cfg=Config()):
    tf.reset_default_graph()

    os.makedirs(outFolder, exist_ok=True)

    targetKeypointsOP = np.array(pv.PolyData(inputKeypoints).points).astype(np.float64) / 1000

    if cfg.withDensePointCloud:
        densePointCloud = np.array(pv.PolyData(inputDensePointCloudFile).points).astype(np.float64) / 1000

    if initialPoseFile is None:
        pose = None
        if betaFile is not None:
            betas = np.load(betaFile)
        else:
            betas = None
        trans = np.array([0, 0, 0], dtype=np.float64)
    else:
        trans, pose, betas = loadCompressedFittingParam(initialPoseFile,)

    personalShape = np.load(personalShapeFile) / 1000 if personalShapeFile is not None else None

    smplshtf = SMPLSH(pose=pose, trans=trans, betas=betas, constantBeta=cfg.constantBeta, personalShape=personalShape,
                      SMPLSHNpzFile=smplshDataFile)
    smplshtf.skeletonJointsToFixToDense = []
    # smplshtf = SMPLSH(betas, pose, trans)

    smplFaces = np.array(smplshtf.smplFaces)

    jointConverter = VertexToOpJointsConverter()
    opJoints = jointConverter(smplshtf.smplVerts[None, ...], smplshtf.smplJoints[None, ...])[0, ...]
    # opJointsNp = sess.run(opJoints)
    # Data.write_obj(join(outFolder, 'SmplshRestPoseOpJoints.obj'), opJointsNp)
    # targetKeypointsOP[2, 2] = -1
    # targetKeypoints = (OP2AdamJointMat @ targetKeypointsOP).astype(np.float64)

    for iKp in range(targetKeypointsOP.shape[0]):
        if np.any(targetKeypointsOP[iKp, 2] < 0):
            targetKeypointsOP[iKp, :] = [0, 0, -1]

    # Remove the cost for unobserved key points
    bodyJoints = [i for i in range(cfg.numBodyJoint)  if i not in cfg.headJointsId]

    # Remove the cost for unobserved key points
    if cfg.noBodyKeyJoint:
        if not cfg.noHandAndHead:
            jointsNoBody = [i for i in range(opJoints.shape[0]) if i not in bodyJoints]
            keypointFitCost = tf.reduce_mean(tf.square(
                tf.gather(tf.multiply(
                    tf.nn.relu(tf.sign(targetKeypointsOP[:opJoints.shape[0], 2:3])),
                    opJoints - targetKeypointsOP[:opJoints.shape[0], :]
                ), jointsNoBody)
            ))
        else:
            keypointFitCost = 0
    else:
        keypointFitCost = tf.reduce_mean(tf.square(
            tf.multiply(
                tf.nn.relu(tf.sign(targetKeypointsOP[:opJoints.shape[0], 2:3])),
                opJoints - targetKeypointsOP[opJoints.shape[0], :]
            )
        ))

    if cfg.withFaceKp:
        keypointFitCost = keypointFitCost + faceKpLosstf(smplshtf.smplVerts, targetKeypointsOP)

    if betas is not None:
        betaRegularizerCostToKp = cfg.betaRegularizerWeightToKP * tf.reduce_sum(tf.square(smplshtf.betas - betas))
    else:
        betaRegularizerCostToKp = cfg.betaRegularizerWeightToKP * tf.reduce_sum(tf.square(smplshtf.betas))
    regularizerCostToKp = cfg.jointRegularizerWeight * tf.reduce_sum(tf.square(smplshtf.pose))

    intepolationMatrixNp = np.load(toSparsePointCloudInterpoMatFile)
    intepolationMatrix = tf.constant(intepolationMatrixNp, dtype=np.float64, name="intepolationMatrix")

    targetVerts = np.array(pv.PolyData(sparsePCObjFile).points, np.float64) / 1000

    # Define fit cost to sparse point cloud
    skelData = json.load(open(skelDataFile))
    coarseMeshPts = np.array(skelData['VTemplate'])

    # if the input is original obj point cloud
    if targetVerts.shape[0] == cfg.numRealCorners:
        targetVerts = np.concatenate([targetVerts, np.repeat([[0,0,-1]], intepolationMatrix.shape[0] - cfg.numRealCorners, axis=0)])
    validVertsOnRestpose = np.where(coarseMeshPts[2, :] != -1)[0]

    obsIds = np.where(targetVerts[:cfg.numRealCorners, 2] > 0)[0]
    constraintIds = np.intersect1d(obsIds, validVertsOnRestpose)

    costICPToSparse = tf.reduce_mean(
        tf.square(tf.gather(targetVerts - tf.matmul(intepolationMatrix, smplshtf.smplVerts), constraintIds)))

    # obsIds = np.where(targetVertsPH[:, 2] >0)[0]
    skelFixCost = 0
    for iJoint in cfg.skeletonJointsToFix:
        skelFixCost = skelFixCost + 100 * tf.reduce_mean(tf.square(smplshtf.pose[(iJoint * 3):(iJoint * 3 + 3)]))

    costICPToSparse = costICPToSparse + cfg.keypointFitWeightInToDenseICP * keypointFitCost + regularizerCostToKp + betaRegularizerCostToKp
    costICPToSparse = costICPToSparse + skelFixCost

    if cfg.withDensePointCloud:
        closestPtsSetDense = tf.placeholder(dtype=np.float64, shape=smplshtf.smplVerts.shape, name="closestPtsSet")
        fitCostToDense = tf.reduce_mean(tf.square(
            tf.multiply(
                tf.nn.relu(tf.sign(closestPtsSetDense[:, 2:3])),
                tf.gather(smplshtf.smplVerts, cfg.indicesVertsToOptimize) - tf.gather(closestPtsSetDense,
                                                                                      cfg.indicesVertsToOptimize)
            )))
        costICPToSparse = costICPToSparse + cfg.densePointCloudWeight * fitCostToDense

    stepICPToSparse = tf.Variable(0, trainable=False)
    rateICPToSparse = tf.train.exponential_decay(cfg.learnrate_ph, stepICPToSparse, cfg.lrDecayStep, cfg.lrDecayRate)
    train_step_ICPToSparse = tf.train.AdamOptimizer(learning_rate=rateICPToSparse).minimize(costICPToSparse,
                                                                                            global_step=stepICPToSparse)

    # train_step_ICPToSparse = tf.train.GradientDescentOptimizer(learning_rate=rateICPToSparse).minimize(costICPToSparse,
    #                                                                                         global_step=stepICPToSparse)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    smplshInitialVerts = sess.run(smplshtf.smplVerts)
    # smplshJoints = sess.run(smplshtf.smplJoints)
    Data.write_obj(join(outFolder, 'SmplshInitial.obj'), smplshInitialVerts * 1000, smplshtf.smplFaces)
    # Data.write_obj(join(outFolder, 'SmplshRestPoseJoints.obj'), smplshJoints)

    errs = []

    # Nonrigid ICP Procedure
    # print("Fit to sparse point clouds, initialCost:", sess.run(costICPToSparse))
    loop = tqdm.tqdm(range(cfg.numIterFitting))
    for i in loop:
        # sess.run(costICPToSparse)
        sess.run(train_step_ICPToSparse, )

        errs.append(np.abs(sess.run(costICPToSparse,)))

        # if not i % cfg.printStep:
            # print("Cost:", errs[-1],
            #       "keypointFitCost:", sess.run(keypointFitCost, ),
            #       'regularizerCostToKp:', sess.run(regularizerCostToKp, ),
            #       'LearningRate:', sess.run(rateICPToSparse, ))
        desc = "Cost:" + str(errs[-1]) + \
                  " keypointFitCost:" + str(sess.run(keypointFitCost, ) if not cfg.noHandAndHead else 0) +\
                  ' regularizerCostToKp:' + str(sess.run(regularizerCostToKp, )) +\
                  ' LearningRate:' + str(sess.run(rateICPToSparse, ))
        loop.set_description(desc)

        if sess.run(costICPToSparse,) < cfg.terminateLoss:
            print("Stop optimization because current loss is: ", sess.run(costICPToSparse,), ' less than: ', cfg.terminateLoss)
            break

        if i > 0 :
            errStep = np.abs(errs[-1] - errs[-2])
            if errStep < cfg.terminateLossStep:
                print("Stop optimization because current step is: ", errStep, ' less than: ', cfg.terminateLossStep)
                break

    transVal = sess.run(smplshtf.trans)
    poseVal = sess.run(smplshtf.pose)
    betaVal = sess.run(smplshtf.betas)

    outParamFile = join(outFolder, 'ToSparseFittingParams.npz')
    np.savez(outParamFile, trans=transVal, pose=poseVal, beta=betaVal)

    outFile = join(outFolder, 'ToSparseMesh.obj')
    Data.write_obj(outFile, sess.run(smplshtf.smplVerts) * 1000, smplFaces)

    if cfg.outputErrs:
        plt.close('all')

        fig, a_loss = plt.subplots()
        a_loss.plot(errs, linewidth=3)
        a_loss.set_yscale('log')
        # a_loss.yscale('log')
        a_loss.set_title('losses: {}'.format(errs[-1]))
        a_loss.grid()
        fig.savefig(join(outFolder,
                         'ErrCurve_' + '_LR' + str(cfg.learnrate_ph)  + '.png'),
                    dpi=256, transparent=False, bbox_inches='tight', pad_inches=0)

        json.dump(errs, open(join(outFolder, 'Errs.json'), 'w'))

def toSparseFittingKeypoints(inputKeypoints, outFolder, betaFile, personalShapeFile, smplshDataFile,
                             inputDensePointCloudFile=None, initialPoseFile=None, cfg=Config()):
    tf.reset_default_graph()

    os.makedirs(outFolder, exist_ok=True)

    if cfg.withDensePointCloud and inputDensePointCloudFile is not None:
        densePointCloud = np.array(pv.PolyData(inputDensePointCloudFile).points).astype(np.float64) / 1000
        tree = KDTree(densePointCloud)

    targetKeypointsOP = np.array(pv.PolyData(inputKeypoints).points).astype(np.float64) / 1000

    if initialPoseFile is None:
        pose = None
        if betaFile is not None:
            betas = np.load(betaFile)
        else:
            betas = None
        trans = np.array([0, 0, 0], dtype=np.float64)
    else:
        trans, pose, betas = loadCompressedFittingParam(initialPoseFile,)

    if personalShapeFile is not None:
        personalShape = np.load(personalShapeFile) / 1000
    else:
        personalShape = None


    smplshtf = SMPLSH(pose=pose, trans=trans, betas=betas, constantBeta=cfg.constantBeta, personalShape=personalShape,
                      SMPLSHNpzFile=smplshDataFile)
    smplshtf.skeletonJointsToFixToDense = []

    smplFaces = np.array(smplshtf.smplFaces)

    jointConverter = VertexToOpJointsConverter()
    opJoints = jointConverter(smplshtf.smplVerts[None, ...], smplshtf.smplJoints[None, ...])[0, ...]

    for iKp in range(targetKeypointsOP.shape[0]):
        if np.any(targetKeypointsOP[iKp, 2] < 0):
            targetKeypointsOP[iKp, :] = [0, 0, -1]

    # Remove the cost for unobserved key points
    bodyJoints = [i for i in range(cfg.numBodyJoint)  if i not in cfg.headJointsId]

    if cfg.noBodyKeyJoint:
        jointsNoBody = [i for i in range(targetKeypointsOP.shape[0]) if i not in bodyJoints]
        keypointFitCost = tf.reduce_mean(tf.square(
            tf.gather(tf.multiply(
                tf.nn.relu(tf.sign(targetKeypointsOP[:, 2:3])),
                opJoints - targetKeypointsOP
            ), jointsNoBody)
        ))
    else:
        keypointFitCost = tf.reduce_mean(tf.square(
            tf.multiply(
                tf.nn.relu(tf.sign(targetKeypointsOP[:, 2:3])),
                opJoints - targetKeypointsOP
            )
        ))

    if betas is not  None:
        betaRegularizerCostToKp = cfg.betaRegularizerWeightToKP * tf.reduce_sum(tf.square(smplshtf.betas - betas))
    else:
        betaRegularizerCostToKp = cfg.betaRegularizerWeightToKP * tf.reduce_sum(tf.square(smplshtf.betas))

    regularizerCostToKp = cfg.jointRegularizerWeight * tf.reduce_sum(tf.square(smplshtf.pose))


    # obsIds = np.where(targetVertsPH[:, 2] >0)[0]
    skelFixCost = 0
    for iJoint in cfg.skeletonJointsToFix:
        skelFixCost = skelFixCost + 100 * tf.reduce_mean(tf.square(smplshtf.pose[(iJoint * 3):(iJoint * 3 + 3)]))

    costICPToSparse =  cfg.keypointFitWeightInToDenseICP * keypointFitCost + regularizerCostToKp + betaRegularizerCostToKp
    costICPToSparse = costICPToSparse + skelFixCost

    if cfg.withDensePointCloud:
        closestPtsSetDense = tf.placeholder(dtype=np.float64, shape=smplshtf.smplVerts.shape, name="closestPtsSet")
        fitCostToDense = tf.reduce_mean(tf.square(
            tf.multiply(
                tf.nn.relu(tf.sign(closestPtsSetDense[:, 2:3])),
                tf.gather(smplshtf.smplVerts, cfg.indicesVertsToOptimize) - tf.gather(closestPtsSetDense,
                                                                                      cfg.indicesVertsToOptimize)
            )))
        costICPToSparse = costICPToSparse + cfg.densePointCloudWeight * fitCostToDense

    stepICPToSparse = tf.Variable(0, trainable=False)
    rateICPToSparse = tf.train.exponential_decay(cfg.learnrate_ph, stepICPToSparse, cfg.lrDecayStep, cfg.lrDecayRate)
    train_step_ICPToSparse = tf.train.AdamOptimizer(learning_rate=rateICPToSparse).minimize(costICPToSparse,
                                                                                            global_step=stepICPToSparse)

    # train_step_ICPToSparse = tf.train.GradientDescentOptimizer(learning_rate=rateICPToSparse).minimize(costICPToSparse,
    #                                                                                         global_step=stepICPToSparse)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    smplshInitialVerts = sess.run(smplshtf.smplVerts)
    # smplshJoints = sess.run(smplshtf.smplJoints)
    Data.write_obj(join(outFolder, 'SmplshInitial.obj'), smplshInitialVerts * 1000, smplshtf.smplFaces)
    # Data.write_obj(join(outFolder, 'SmplshRestPoseJoints.obj'), smplshJoints)

    errs = []
    keypointFitCostList = []
    regularizerCostToKpList = []
    fitCostToDenseList = []

    # Nonrigid ICP Procedure
    # print("Fit to sparse point clouds, initialCost:", sess.run(costICPToSparse))
    finished = False
    for i in range(cfg.numComputeClosest):
        if cfg.withDensePointCloud:
            verts = sess.run(smplshtf.smplVerts)
            closestPtsNp, dis = searchForClosestPoints(verts, densePointCloud, tree)
            # neglect points that are too far away
            closestPtsNp[np.where(dis > cfg.maxDistanceToClosestPt)[0], :] = [0, 0, -1]
            feedDict = {closestPtsSetDense : closestPtsNp}
        else:
            feedDict = {}

        loop = tqdm.tqdm(range(cfg.numIterFitting), position=0, leave=True)
        for i in range(cfg.numIterFitting):
            # sess.run(costICPToSparse)

            errs.append(np.abs(sess.run(costICPToSparse, feed_dict = feedDict)))

            # if not i % cfg.printStep:
                # print("Cost:", errs[-1],
                #       "keypointFitCost:", sess.run(keypointFitCost, ),
                #       'regularizerCostToKp:', sess.run(regularizerCostToKp, ),
                #       'LearningRate:', sess.run(rateICPToSparse, ))
            keypointFitCostList.append(sess.run(keypointFitCost, ))
            regularizerCostToKpList.append(sess.run(regularizerCostToKp, ))
            fitCostToDenseList.append(sess.run(fitCostToDense, feed_dict=feedDict))

            desc = "Cost:" + str(errs[-1]) + \
                      " keypointFitCost:" + str(keypointFitCostList[-1])+\
                      ' regularizerCostToKp:' + str(regularizerCostToKpList[-1]) +\
                      ' LearningRate:' + str(sess.run(rateICPToSparse)) + \
                    ' To Dense: ' + str(fitCostToDenseList[-1])

            sess.run(train_step_ICPToSparse, feed_dict = feedDict)

            loop.set_description(desc)

            if sess.run(costICPToSparse, feed_dict=feedDict) < cfg.terminateLoss:
                print("Stop optimization because current loss is: ", sess.run(costICPToSparse,), ' less than: ', cfg.terminateLoss)
                finished = True

                break

            if i > 0 :
                errStep = np.abs(errs[-1] - errs[-2])
                if errStep < cfg.terminateLossStep:
                    print("Stop optimization because current step is: ", errStep, ' less than: ', cfg.terminateLossStep)
                    finished = True
                    break
        if finished:
            break
    transVal = sess.run(smplshtf.trans)
    poseVal = sess.run(smplshtf.pose)
    betaVal = sess.run(smplshtf.betas)

    outParamFile = join(outFolder, 'ToSparseFittingParams.npz')
    np.savez(outParamFile, trans=transVal, pose=poseVal, beta=betaVal)

    outFile = join(outFolder, 'ToSparseMesh.obj')
    Data.write_obj(outFile, sess.run(smplshtf.smplVerts) * 1000, smplFaces)

    if cfg.outputErrs:
        plt.close('all')

        fig, a_loss = plt.subplots()
        a_loss.plot(errs, linewidth=1, label='TotalError')
        a_loss.plot(keypointFitCostList, linewidth=1, label='keypointFitCost')
        a_loss.plot(regularizerCostToKpList, linewidth=1, label='regularizerCost')
        a_loss.plot(fitCostToDenseList, linewidth=1, label='fitCostToDense')
        a_loss.set_yscale('log')
        # a_loss.yscale('log')
        a_loss.set_title('losses: {}'.format(errs[-1]))
        a_loss.legend()
        a_loss.grid()
        fig.savefig(join(outFolder,
                         'ErrCurve_' + '_LR' + str(cfg.learnrate_ph)  + '.png'),
                    dpi=256, transparent=False, bbox_inches='tight', pad_inches=0)

        json.dump(errs, open(join(outFolder, 'Errs.json'), 'w'))


def interpolateWithSparsePointCloudSoftly(inMeshFile, inSparseCloud, outInterpolatedFile, skelDataFile, interpoMatFile, laplacianMatFile=None, \
    handIndicesFile = r'HandIndices.json', HeadIndicesFile = 'HeadIndices.json', softConstraintWeight = 100,
    numRealCorners = 1487, fixHandAndHead = True, faces=None):
    handIndices = json.load(open(handIndicesFile))
    headIndices = json.load(open(HeadIndicesFile))

    indicesToFix = copy.copy(handIndices)
    indicesToFix.extend(headIndices)

    deformedSMPLSH = pv.PolyData(inMeshFile)

    targetMesh = pv.PolyData(inSparseCloud)
    if laplacianMatFile is None:
        LNP = getLaplacian(inMeshFile)
    else:
        LNP = np.load(laplacianMatFile)
    # LNP
    # Define fit cost to dense point cloud
    skelData = json.load(open(skelDataFile))
    coarseMeshPts = np.array(skelData['VTemplate'])
    validVertsOnRestpose = np.where(coarseMeshPts[2, :] != -1)[0]

    obsIds = np.where(targetMesh.points[:, 2] > 0)[0]

    constraintIds = np.intersect1d(obsIds, validVertsOnRestpose)
    # validTargets = targetVerts[constraintIds, :]

    intepolationMatrixNp = np.load(interpoMatFile)

    smplshRestPoseVerts = np.array(deformedSMPLSH.points)

    # Deform to Sparse Point Cloud
    # only interpolate the points that is actually a corner
    constraintIds = constraintIds[np.where(constraintIds < numRealCorners)]

    targetPts = targetMesh.points[constraintIds, :]
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

    # deformedSMPLSH.points = interpolatedVerts
    # deformedSMPLSH.save(outInterpolatedFile)

    Data.write_obj(outInterpolatedFile, interpolatedVerts, faces)

def getPersonalShapeFromInterpolation(inMeshFile, inSparseCloud, inFittingParamFile, outInterpolatedObjFile, outFittingParamFileWithPS,
    skelDataFile, interpoMatFile, laplacianMatFile=None, smplshData=r'..\SMPL_reimp\SmplshModel_m.npz',\
    handIndicesFile = r'HandIndices.json', HeadIndicesFile = 'HeadIndices.json', softConstraintWeight = 100,
    numRealCorners = 1487, fixHandAndHead = True, ):

    device = torch.device("cpu")
    smplsh = smplsh_torch.SMPLModel(device, smplshData, personalShape=None, unitMM=True)
    faces = smplsh.faces

    interpolateWithSparsePointCloudSoftly(inMeshFile, inSparseCloud, outInterpolatedObjFile, skelDataFile,
                                         interpoMatFile, laplacianMatFile=laplacianMatFile, \
                                         handIndicesFile=handIndicesFile, HeadIndicesFile=HeadIndicesFile,
                                         softConstraintWeight=softConstraintWeight,
                                         numRealCorners=numRealCorners, fixHandAndHead=fixHandAndHead,
                                         faces=faces)


    param = np.load(inFittingParamFile)
    # personalShapeFinal = param['personalShape']
    trans = param['trans'] * 1000
    pose = param['pose']
    beta = param['beta']

    pose = torch.tensor(pose, dtype=torch.float64, requires_grad=False, device=device)
    beta = torch.tensor(beta, dtype=torch.float64, requires_grad=False, device=device)
    trans = torch.tensor(trans, dtype=torch.float64,
                         requires_grad=False, device=device)

    verts = smplsh(beta, pose, trans)
    smplsh.write_obj(verts, 'SmplshDeformation.obj')


    T, pbs, v_shaped = smplsh.getTransformation(beta, pose, trans, returnPoseBlendShape=True)

    inverseTransform = np.zeros(T.shape, dtype=np.float64)

    interpolatedMesh = pv.PolyData(outInterpolatedObjFile)
    interpolatedVerts = np.array(interpolatedMesh.points)
    personalShapeFinalRestpose = np.zeros(interpolatedVerts.shape, dtype=np.float64)

    for i in range(T.shape[0]):
        inverseTransform[i, :, :] = np.linalg.inv(T[i, :, :].cpu().detach().numpy())
        pt = interpolatedVerts[i:i + 1, :].transpose()
        pt = np.vstack([pt, 1])

        ptBackToRest = inverseTransform[i, :, :] @ pt
        personalShapeFinalRestpose[i, :] = ptBackToRest[:3, 0]

    # the rest pose has also been applied with pose blend shape, we need to deduct it
    personalShapeFinalRestpose = (personalShapeFinalRestpose- pbs.cpu().detach().numpy())/1000

    # then get the pure smplsh rest pose shape
    np.savez(outFittingParamFileWithPS, trans=trans.cpu().numpy()/1000, pose=pose.cpu().numpy(), beta=beta.cpu().numpy(),
             personalShape=personalShapeFinalRestpose-v_shaped.cpu().numpy()/1000)

    # interpolatedMesh = pv.PolyData(outInterpolatedFile)

    # interpolatedMesh.points = personalShapeFinalRestpose
    # interpolatedMesh.save('DisplacementToRestpose.ply')


if __name__ == '__main__':
    inImgParentFolder = r'E:\WorkingCopy\2020_06_30_AC_ConsequtiveTexturedFitting2\Copied\Images'
    camParamFile = r'C:\Code\MyRepo\03_capture\BodyTracking\Data\CamParams\Lada_19_12_13\cam_params.json'
    completedObjFolder = r'E:\WorkingCopy\2020_06_30_AC_ConsequtiveTexturedFitting2\Copied\Deformed\SLap_SBiLap_True_TLap_0_JTW_0_JBiLap_0_Step120_Overlap0\Deformed'
    outFolder = r'E:\WorkingCopy\2020_06_30_AC_ConsequtiveTexturedFitting2\ToSparse'

    cfg = Config()
    cfg.numIterFitting = 5000
    cfg.terminateLoss = 1e-3

    startFrame=52
    # startFrame=41

    import glob

    class InputBundle:
        def __init__(s):
            # person specific
            s.skelDataFile = r'..\Data\PersonalModel_Lada\06_SKelDataLadaWeightsMultiplierCorrectAnkle_1692.json'
            s.toSparsePointCloudInterpoMatFile = r'..\Data\PersonalModel_Lada\InterpolationMatrix.npy'

            s.betaFile = r'..\Data\PersonalModel_Lada\Beta.npy'
            s.personalShapeFile = r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\InitialFit\PersonalModel\PersonalShape.npy'

            s.OP2AdamJointMatFile = r'..\Data\PersonalModel_Lada\OpenposeToSmplsh\OP2AdamJointMat.npy'
            s.AdamGoodJointsFile = r'..\Data\PersonalModel_Lada\OpenposeToSmplsh\AdamGoodJoints.npy'
            s.smplsh2OPRegressorMatFile = r'..\Data\PersonalModel_Lada\OpenposeToSmplsh\smplshRegressorNoFlatten.npy'
            s.smplshDataFile = r'..\SMPL_reimp\SmplshModel_m.npz'

    inputs = InputBundle()
    inImgFolders = glob.glob(join(inImgParentFolder, '*'))
    inObjFiles = glob.glob(join(completedObjFolder, '*.obj'))
    inImgFolders.sort()
    inObjFiles.sort()

    loop =  tqdm.tqdm(range(startFrame, len(inImgFolders)))
    for iFrame in loop:
        inImgFolder = inImgFolders[ startFrame]
        objFile = inObjFiles[iFrame]
        frameName = os.path.basename(inImgFolder)
        loop.set_description('Processing frame: ', os.path.basename(frameName))

        print('Processing frame: ', os.path.basename(frameName))

        outFolderFrame = join(outFolder, join(outFolder, frameName))
        os.makedirs(outFolderFrame, exist_ok=True)
        toSparseFitting(inImgFolder, objFile, outFolderFrame, inputs.skelDataFile, inputs.toSparsePointCloudInterpoMatFile,
                        inputs.betaFile, inputs.personalShapeFile, inputs.OP2AdamJointMatFile, inputs.AdamGoodJointsFile, inputs.smplsh2OPRegressorMatFile,
                        smplshDataFile=inputs.smplshDataFile)


