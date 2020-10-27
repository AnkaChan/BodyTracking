# from S01_Build_SMPL_Socks import *
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
import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
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

            if use_face:
                corrs = [
                    [75, 3161],  # middle chin
                    [115, 3510],  # right mouth corner
                    [121, 69],  # left mouth corner
                    [74, 3544],
                    [76, 285],

                ]


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

def faceKpLosstf(smplshVerts, keypoints):
    corrs =np.array( [
        [75, 3161],  # middle chin
        [115, 3510],  # right mouth corner
        [121, 69],  # left mouth corner
        [74, 3544],
        [76, 102],

    ])


    return tf.reduce_mean((tf.gather(smplshVerts, corrs[:,1]) - keypoints[corrs[:,0]])**2)


if __name__ == '__main__':
    corseMeshToSMPLSHCorrs = [
        [641, 5315],
        [1594, 5326],
        [719, 5015],
        [668, 5009],
        [704, 4206],
        [680, 4791],
        # [756,    6399 ], # this is bad, Right shoulder
        [696, 5250],
        [558, 1926],
        [1596, 1937],
        [585, 1914],
        [582, 1723],
        [616, 2820],
        [612, 1392],
        [549, 1862],
        [1489, 3078],
        [1235, 6416],
        [1229, 669],
        # [1192,   3077 ], # bad one
        [852, 3145],
        [1012, 3429],
        [886, 4856],
        [34, 4459],
        [1504, 1454],
        # [34,    4459  ],
        # Left knee is needed here
        [155, 1020],
        [353, 1050],
        [21, 4464],
        [410, 3203],
        [243, 6531],
        [502, 6636],
        [500, 6634],
        [514, 6580],
        [510, 6574],
        [324, 3330],
        [0, 3328],
        [327, 3246],
        [333, 3243],
        # butt,
        [896, 3117],
        [892, 6468],

    ]
    corseMeshToSMPLSHCorrs = np.array(corseMeshToSMPLSHCorrs)

    # targetMesh = r'F:\WorkingCopy2\2020_04_05_LadaRestPosePointCloud\Pointclouds\03052\AA00003052_CoarseMesh_tri.ply'
    # SMPLSHNpzFile = r'..\Data\BuildSmplsh\Output\SmplshModel_m.npz'
    # skelDataFile = r'C:\Code\MyRepo\ChbCapture\06_Deformation\MeshInterpolation\06_SKelDataLadaWeightsMultiplierCorrectAnkle_1692.json'
    #
    # # outFolder = r'SMPLSHFit\LadaOldSuit_WithOPKeypoints'
    # outFolder = r'..\Data\NewInitialFitting\InitialRegistration'
    #
    # inputKeypoints = r'F:\WorkingCopy2\2020_05_15_AC_Gray2RGBData\Copied\TPose\ToRGB\Reconstruction\PointCloud.obj'
    # inputDensePointCloudFile = r'F:\WorkingCopy2\2020_04_05_LadaRestPosePointCloud\Pointclouds\03052\scene_dense.ply'
    # inFittingParam = r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\FitToSparseCloud\FittingParams\03052.npz'

    # SMPLSHNpzFile = r'..\Data\BuildSmplsh\Output\SmplshModel_m.npz'
    # skelDataFile = r'C:\Code\MyRepo\ChbCapture\06_Deformation\MeshInterpolation\06_SKelDataLadaWeightsMultiplierCorrectAnkle_1692.json'
    #
    # # outFolder = r'SMPLSHFit\LadaOldSuit_WithOPKeypoints'
    # outFolder = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\TextureCompletionFitting\03067'
    #
    # targetMesh = r'F:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied\Deformed\SLap_SBiLap_True_TLap_0_JTW_5000_JBiLap_0_Step8_Overlap0\Deformed\A00003067.ply'
    # inputKeypoints = r'FF:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied\03067\toRGB\Reconstruction\PointCloud.obj'
    # # inputDensePointCloudFile = r'F:\WorkingCopy2\2020_04_05_LadaRestPosePointCloud\Pointclouds\03052\scene_dense.ply'
    # inputDensePointCloudFile = None
    # inFittingParam = r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\FitToSparseCloud\FittingParams\03052.npz'

    # Data for katey
    SMPLSHNpzFile = r'..\Data\BuildSmplsh_Female\Output\SmplshModel_f_noBun.npz'
    skelDataFile = r'C:\Code\MyRepo\ChbCapture\06_Deformation\MeshInterpolation\06_SKelDataKeteyWeightsMultiplierCorrectAnkle_1692.json'

    # outFolder = r'SMPLSHFit\LadaOldSuit_WithOPKeypoints'
    outFolder = r'..\Data\KateyBodyModel'

    targetMesh = r'F:\WorkingCopy2\2020_08_27_KateyBodyModel\TPose\Deformed\A00018411.obj'
    inputKeypoints = r'F:\WorkingCopy2\2020_08_27_KateyBodyModel\TPose\Keypoints\18411.obj'
    # inputDensePointCloudFile = r'F:\WorkingCopy2\2020_04_05_LadaRestPosePointCloud\Pointclouds\03052\scene_dense.ply'
    inputDensePointCloudFile = None
    inFittingParam = r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\FitToSparseCloud\FittingParams\03052.npz'

    smplsh_npModel = smplsh_np.SMPLSHModel(SMPLSHNpzFile)

    numComputeClosest = 5
    numIterEachClosestSet = 2000
    jointRegularizerWeight = 0.000001
    # jointRegularizerWeight = 0.0000001
    # jointRegularizerWeight = 0
    learnrate_ph = 0.05
    lrDecayStep = 100
    lrDecayRate = 0.97
    # numIterToKp = 3000
    # printStep = 500

    # noBodyKeyJoint = False
    noBodyKeyJoint = True
    numBodyJoint = 25
    headJointsId = [0, 15, 16, 17, 18]

    bodyJoints = [i for i in range(numBodyJoint)  if i not in headJointsId]
    keypointFitWeightInToDenseICP = 1

    withFaceKp = True

    numIterToKp = 3000
    printStep = 50

    indicesVertsToOptimize = list(range(6750))

    # keypointFitWeightInToDenseICP = 0.1
    # keypointFitWeightInToDenseICP = 0.0
    constantBeta = False
    betaRegularizerWeightToKP = 0
    manualCorrsWeightToKP = 1

    # maxDistanceToClosestPt = 30
    maxDistanceToClosestPt = 50 / 1000

    withDensePointCloud = True
    densePointCloudWeight = 1

    os.makedirs(outFolder, exist_ok=True)

    targetKeypointsOP = np.array(pv.PolyData(inputKeypoints).points).astype(np.float64) / 1000
    skeletonJointsToFix = [10, 11]

    if inputDensePointCloudFile is not None:
        densePointCloud = np.array(pv.PolyData(inputDensePointCloudFile).points).astype(np.float64) / 1000

    else:
        densePointCloud = np.zeros((1,3))

    transInit, poseInit, betaInit = loadCompressedFittingParam(inFittingParam, readPersonalShape=False)
    betas = betaInit
    trans = targetKeypointsOP[0, :]

    smplshtf = SMPLSH(trans=trans, betas=betas, constantBeta=constantBeta, SMPLSHNpzFile=SMPLSHNpzFile)
    smplshtf.skeletonJointsToFixToDense = []
    # smplshtf = SMPLSH(betas, pose, trans)

    smplFaces = np.array(smplshtf.smplFaces)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    smplshRestposeVerts = sess.run(smplshtf.smplVerts)
    smplshJoints = sess.run(smplshtf.smplJoints)
    Data.write_obj(join(outFolder, 'SmplshRestPoseMesh.obj'), smplshRestposeVerts, smplshtf.smplFaces)
    Data.write_obj(join(outFolder, 'SmplshRestPoseJoints.obj'), smplshJoints)

    jointConverter = VertexToOpJointsConverter()
    opJoints = jointConverter(smplshtf.smplVerts[None, ...], smplshtf.smplJoints[None, ...])[0,...]
    opJointsNp = sess.run(opJoints)
    Data.write_obj(join(outFolder, 'SmplshRestPoseOpJoints.obj'), opJointsNp)
    # targetKeypointsOP[2, 2] = -1
    # targetKeypoints = (OP2AdamJointMat @ targetKeypointsOP).astype(np.float64)

    for iKp in range(targetKeypointsOP.shape[0]):
        if np.any(targetKeypointsOP[iKp, 2] < 0):
            targetKeypointsOP[iKp, :] = [0, 0, -1]

    # Remove the cost for unobserved key points
    if noBodyKeyJoint:
        jointsNoBody = [i for i in range(opJoints.shape[0])  if i not in bodyJoints]
        keypointFitCost = tf.reduce_mean(tf.square(
            tf.gather(tf.multiply(
                tf.nn.relu(tf.sign(targetKeypointsOP[:opJoints.shape[0], 2:3])),
                opJoints - targetKeypointsOP[:opJoints.shape[0], :]
            ), jointsNoBody)
        ))
    else:
        keypointFitCost = tf.reduce_mean(tf.square(
            tf.multiply(
                tf.nn.relu(tf.sign(targetKeypointsOP[:opJoints.shape[0], 2:3])),
                opJoints - targetKeypointsOP[opJoints.shape[0], :]
            )
        ))

    if withFaceKp:
        keypointFitCost = keypointFitCost + faceKpLosstf(smplshtf.smplVerts, targetKeypointsOP )

    betaRegularizerCostToKp = betaRegularizerWeightToKP * tf.reduce_sum(tf.square(smplshtf.betas - betas))

    targetVerts = np.array(pv.PolyData(targetMesh).points, np.float64) / 1000
    # Define the fit cost to manually picked correspondences:
    costToManualCorrs = tf.reduce_mean(
        tf.square(
            tf.gather(smplshtf.smplVerts, corseMeshToSMPLSHCorrs[:, 1])
            - tf.gather(targetVerts, corseMeshToSMPLSHCorrs[:, 0])))

    smplshtf.skeletonJointsToFixToDense = skeletonJointsToFix
    # Define fit cost to keypoints
    skelFixCost = 0
    for iJoint in smplshtf.skeletonJointsToFixToDense:
        skelFixCost = skelFixCost + 100 * tf.reduce_mean(tf.square(smplshtf.pose[(iJoint * 3):(iJoint * 3 + 3)]))

    stepICPToKp = tf.Variable(0, trainable=False)
    rateICPToKp = tf.train.exponential_decay(learnrate_ph, stepICPToKp, lrDecayStep, lrDecayRate)

    regularizerCostToKp = jointRegularizerWeight * tf.reduce_sum(tf.square(smplshtf.pose))
    costICPToKp = keypointFitCost + regularizerCostToKp + skelFixCost + manualCorrsWeightToKP * costToManualCorrs

    if not constantBeta:
        costICPToKp = costICPToKp + betaRegularizerCostToKp

    train_step_ICPToKp = tf.train.AdamOptimizer(learning_rate=rateICPToKp).minimize(costICPToKp, global_step=stepICPToKp)

    # Define fit cost to dense point cloud
    skelData = json.load(open(skelDataFile))
    coarseMeshPts = np.array(skelData['VTemplate'])
    validVertsOnRestpose = np.where(coarseMeshPts[2,:]!=-1)[0]

    obsIds = np.where(targetVerts[:, 2] >0)[0]

    constraintIds = np.intersect1d(obsIds, validVertsOnRestpose)
    validTargets = targetVerts[constraintIds, :]

    closestPtsSet = tf.placeholder(dtype=np.float64, shape=targetVerts.shape, name="closestPtsSet")
    intepolationMatrixNp = np.zeros((constraintIds.shape[0], smplshtf.smplVerts.shape[0]), dtype=np.float64)
    intepolationMatrix = tf.placeholder(dtype=np.float64, shape=intepolationMatrixNp.shape, name="intepolationMatrix")

    # costICP = tf.reduce_mean(
    #     tf.square(tf.gather(targetVerts, obsIds) - tf.gather(tf.matmul(intepolationMatrix, smplVerts), obsIds)))

    costICPToSparse = tf.reduce_mean(
        tf.square(tf.gather(targetVerts, constraintIds) - tf.matmul(intepolationMatrix, smplshtf.smplVerts)))

    costICPToSparse = costICPToSparse + skelFixCost + keypointFitWeightInToDenseICP * keypointFitCost + regularizerCostToKp

    # cost of fit to dense
    closestPtsSetDense = tf.placeholder(dtype=np.float64, shape=smplshtf.smplVerts.shape, name="closestPtsSet")
    fitCostToDense = tf.reduce_mean(tf.square(
        tf.multiply(
            tf.nn.relu(tf.sign(closestPtsSetDense[:, 2:3])),
            tf.gather(smplshtf.smplVerts, indicesVertsToOptimize) - tf.gather(closestPtsSetDense, indicesVertsToOptimize)
        )

    ))

    if withDensePointCloud:
        costICPToSparse = costICPToSparse + densePointCloudWeight * fitCostToDense

    stepICPToSparse = tf.Variable(0, trainable=False)
    rateICPToSparse = tf.train.exponential_decay(learnrate_ph, stepICPToSparse, lrDecayStep, lrDecayRate)
    train_step_ICPToSparse = tf.train.AdamOptimizer(learning_rate=rateICPToSparse).minimize(costICPToSparse, global_step=stepICPToSparse)

    init = tf.global_variables_initializer()
    sess.run(init)

    print("Fit to key points, initialCost:", sess.run(costICPToKp))

    for i in range(numIterToKp):
        sess.run(train_step_ICPToKp)
        if not i % printStep:
            print("Cost:", sess.run(costICPToKp), 'Lrate:', sess.run(rateICPToKp), " betaRegularizerCostToKp:", sess.run(betaRegularizerCostToKp))

    optimizedPose = sess.run(smplshtf.pose)
    print('optimizedPose:', optimizedPose.reshape((-1, 3)))

    smplshFittedVerts = sess.run(smplshtf.smplVerts)
    Data.write_obj(join(outFolder, 'SmplshFittedToKeypoints.obj'), smplshFittedVerts * 1000, smplFaces)
    np.save(join(outFolder, 'OptimizedPose.npy'), optimizedPose)

    opJointsNp = sess.run(opJoints)

    Data.write_obj(join(outFolder, 'SmplshOpJoints.obj'), opJointsNp * 1000)
    Data.write_obj(join(outFolder, 'OPKeyptsTarget.obj'), targetKeypointsOP * 1000)
    Visualization.drawCorrs(opJointsNp * 1000, targetKeypointsOP * 1000, join(outFolder, 'Corrs.vtk'))

    # Nonrigid ICP Procedure
    tree = KDTree(densePointCloud)
    # print("Fit to sparse point clouds, initialCost:", sess.run(costICPToSparse))

    for i in range(numComputeClosest):
        print("************************************************************\nIteration: %d\n" % i)
        verts = sess.run(smplshtf.smplVerts)

        # We should register the target vertices to SMPL
        # closestPtsNp = searchForClosestPointsOnTriangle(verts, TMesh.points, TMesh.faces.reshape(TMesh.n_faces, -1)[:,1:4])
        closestPtsNp, barys, trianglesId = searchForClosestPointsOnTriangleWithBarycentric(validTargets, verts,
                                                                                           smplFaces)

        intepolationMatrixNp = np.zeros((trianglesId.shape[0], smplshtf.smplVerts.shape[0]), dtype=np.float64)

        for iC in range(intepolationMatrixNp.shape[0]):
            intepolationMatrixNp[iC, smplFaces[trianglesId[iC], 0]] = barys[iC, 0]
            intepolationMatrixNp[iC, smplFaces[trianglesId[iC], 1]] = barys[iC, 1]
            intepolationMatrixNp[iC, smplFaces[trianglesId[iC], 2]] = barys[iC, 2]
            # for iSV in range(intepolationMatrixNp.shape[1]):
            #     t = triVidsNp[iC, :]
            #     b = barycentricsNp[iC, :]
            #     intepolationMatrixNp[iC, t[0]] = b[0]
            #     intepolationMatrixNp[iC, t[1]] = b[1]
            #     intepolationMatrixNp[iC, t[2]] = b[2]

        if withDensePointCloud:
            closestPtsNp, dis = searchForClosestPoints(verts, densePointCloud, tree)
            # neglect points that are too far away
            closestPtsNp[np.where(dis > maxDistanceToClosestPt)[0], :] = [0, 0, -1]
            feedDict = {intepolationMatrix: intepolationMatrixNp, closestPtsSetDense : closestPtsNp}
        else:
            feedDict = {intepolationMatrix: intepolationMatrixNp}

        loop = tqdm.tqdm(range(numIterEachClosestSet))
        for j in loop:
            sess.run(train_step_ICPToSparse, feed_dict=feedDict)

            if not (j % printStep):
                costVal = sess.run(costICPToSparse, feed_dict=feedDict)
                # print("cost:", costVal, "mean_square_root(mm): ", 1000 * np.sqrt(costVal))
                loop.set_description("cost:" +  str(costVal) + " mean_square_root(mm): " + str(1000 * np.sqrt(costVal)))


    verts = sess.run(smplshtf.smplVerts)

    closestPtsNp, barys, trianglesId = searchForClosestPointsOnTriangleWithBarycentric(targetVerts, verts,
                                                                                       smplFaces)

    intepolationMatrixNp = np.zeros((trianglesId.shape[0], smplshtf.smplVerts.shape[0]), dtype=np.float64)
    for iC in range(intepolationMatrixNp.shape[0]):
        if iC in constraintIds:
            intepolationMatrixNp[iC, smplFaces[trianglesId[iC], 0]] = barys[iC, 0]
            intepolationMatrixNp[iC, smplFaces[trianglesId[iC], 1]] = barys[iC, 1]
            intepolationMatrixNp[iC, smplFaces[trianglesId[iC], 2]] = barys[iC, 2]
        else:
            trianglesId[iC] = -1

    # dists = 1000 * np.sqrt(np.sum(np.square(closestPtsNp - result), axis=1))
    #
    # meanDist = np.mean(dists)
    # print("meanDist", meanDist)

    optimizedBetas = sess.run(smplshtf.betas)
    np.save(join(outFolder, r'OptimizedBetas_ICPTriangle.npy'), optimizedBetas)

    optimizedPose = sess.run(smplshtf.pose)
    np.save(join(outFolder, r'OptimizedPoses_ICPTriangle.npy'), optimizedPose)

    optimizedTranslation = sess.run(smplshtf.trans)
    np.save(join(outFolder, r'OptimizedTranslation_ICPTriangle.npy'), optimizedTranslation)

    np.save(join(outFolder, r'InterpolationMatrix.npy'), intepolationMatrixNp)
    np.save(join(outFolder, r'InterpolationTriId.npy'), trianglesId)
    np.save(join(outFolder, r'InterpolationBarys.npy'), barys)

    Data.write_obj(join(outFolder, 'SmplshFittedToSparse.obj'), verts * 1000, smplFaces)