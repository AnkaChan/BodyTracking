from Utility import *
# convert from quad annotation to corner annotation
import json
from SuitCapture import Triangulation, Data, Camera

from shapely.geometry import Polygon
import geopandas
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def loadLabelMePolygonLabels(labelFile):
    with open(labelFile, 'r') as myfile:
        #with open('wenxian_02241.json', 'r') as myfile:
        data=myfile.read()

        # parse file
        obj = json.loads(data)

        return obj['shapes'], obj


def corners_inside(i_min, i_max, j_min, j_max, verts):
    """ returns a vector of indices of wsquares that are within the ij bounding box"""
    within_i = np.logical_and(verts[:,1] >= i_min, verts[:,1] <= i_max)
    within_j = np.logical_and(verts[:,0] >= j_min, verts[:,0] <= j_max)
    within_ij = np.logical_and(within_i, within_j)
    return np.flatnonzero(within_ij)

def angles_between_vecs_batch(v0, v1):
    """ input: v0 and v1 are Nx2 numpy arrays of N 2D vectors (batch)
        output: Nx1 array of angles (unsigned)
    """
    return np.rad2deg(np.arccos(np.clip((v0 * v1).sum(axis=1) / (np.linalg.norm(v0, axis=1) * np.linalg.norm(v1, axis=1)), -1, 1)))

def angles_quad_batch(points):
    """ input: 4 x N x 2 tensor of N 2D quads
        output: N x 4 matrix of internal quad angles (unsigned)
    """
    a0 = angles_between_vecs_batch(points[:,1,:] - points[:,0,:], points[:,3,:] - points[:,0,:])
    a1 = angles_between_vecs_batch(points[:,2,:] - points[:,1,:], points[:,0,:] - points[:,1,:])
    a2 = angles_between_vecs_batch(points[:,1,:] - points[:,2,:], points[:,3,:] - points[:,2,:])
    a3 = angles_between_vecs_batch(points[:,0,:] - points[:,3,:], points[:,2,:] - points[:,3,:])
    return np.array([a0, a1, a2, a3]).transpose()

def edgelens_quad_batch(points):
    """ input: 4 x N x 2 tensor of N 2D quads
        output: N x 4 matrix of quad edges
    """
    e1 = np.linalg.norm(points[:,1,:] - points[:,0,:], axis=1)
    e2 = np.linalg.norm(points[:,2,:] - points[:,1,:], axis=1)
    e3 = np.linalg.norm(points[:,3,:] - points[:,2,:], axis=1)
    e4 = np.linalg.norm(points[:,0,:] - points[:,3,:], axis=1)
    return np.array([e1, e2, e3, e4]).transpose()

def quad_proposals(quad_verts, max_width=60, min_area=450, min_edge_len=15, max_edge_len=55, min_angle=45,
                   max_angle=135):
    """ quad_verts ... N x 2 numpy array of 2D coordinates of N points
        task: generate a set quads ("quad proposals") connecting vertices in quad_verts
        max_width ... [2*max_width, max_width] is the size of crop window to generate initial (unordered) four-sets of points (which will be later connected into quad proposals)
        min/max_edge_length, min/max_angle ... controls which quad proposals will be discarded
        returns: a list of four-tuples (ordered), each four-tuple contains four indices into quad_verts
    """
    # four-point set proposals (unordered, just sets):
    fourtuple_props = set()
    for i in range(quad_verts.shape[0]):
        v0 = quad_verts[i, :]
        cset = set(corners_inside(v0[1], v0[1] + max_width, v0[0] - max_width, v0[0] + max_width, quad_verts))
        cset.remove(i)
        for sub3 in itertools.combinations(cset, 3):
            sub3 += (i,)
            fourtuple_props.add(frozenset(sub3))

    # comput convex hulls of the four-point sets:
    hi_list = []
    rp_list = []
    for idx, item in enumerate(fourtuple_props):
        vert_indices = list(item)
        points = quad_verts[vert_indices, :]
        hull = cv2.convexHull(points.astype(np.float32), returnPoints=False)

        if len(hull) <= 3:
            continue
        assert (len(hull) == 4)

        hull_indices = np.array(vert_indices)[hull[:, 0]]

        reorder_points = quad_verts[hull_indices, :]
        area = cv2.contourArea(reorder_points.astype(np.float32), True)
        assert (area >= 0)  # otherwise wrong orientation
        if area >= min_area:
            rp_list.append(reorder_points)
            hi_list.append(hull_indices)

            # batch filtering of quads whose angles or edges are out of the bounds:
    rp = np.array(rp_list)
    hi = np.array(hi_list)

    if rp.shape[0] == 0:
        return []

    elens = edgelens_quad_batch(rp)
    angles = angles_quad_batch(rp)

    valid_quads = np.all(np.logical_and(np.logical_and(elens >= min_edge_len, elens <= max_edge_len),
                                        np.logical_and(angles >= min_angle, angles <= max_angle)), axis=1)

    vq_idx = np.flatnonzero(valid_quads)
    hull_list = hi[vq_idx, :]

    return hull_list

def quadInCrop(quadPts, cropXYWH):
    quadPts = np.array(quadPts)
    x_max = cropXYWH[0] + cropXYWH[2]
    x_min = cropXYWH[0]

    y_max = cropXYWH[1]+ cropXYWH[3]
    y_min = cropXYWH[1]

    within_x = np.logical_and(quadPts[:,0] >= x_min, quadPts[:,0] <= x_max)
    within_y = np.logical_and(quadPts[:,1] >= y_min, quadPts[:,1] <= y_max)
    within_xy = np.logical_and(within_x, within_y)

    return np.all(within_xy)


def overlapAreaRatio(quad1, quad2):
    minX1 = min([c[0] for c in quad1])
    minY1 = min([c[1] for c in quad1])

    maxX2 = max([c[0] for c in quad2])
    maxY2 = max([c[1] for c in quad2])

    if maxX2 < minX1 or maxY2 < minY1:
        return 0

    minX2 = min([c[0] for c in quad2])
    minY2 = min([c[1] for c in quad2])

    maxX1 = max([c[0] for c in quad1])
    maxY1 = max([c[1] for c in quad1])

    if maxX1 < minX2 or maxY1 < minY2:
        return 0

    polys1 = geopandas.GeoSeries([Polygon(quad1)])
    polys2 = geopandas.GeoSeries([Polygon(quad2)])

    df1 = geopandas.GeoDataFrame({'geometry': polys1})
    df2 = geopandas.GeoDataFrame({'geometry': polys2})

    # areaSmaller = min(df1.area.tolist()[0], df2.area.tolist()[0])
    areaBigger = max(df1.area.tolist()[0], df2.area.tolist()[0])
    overlap = geopandas.overlay(df1, df2, how='intersection')

    overlapAreaList = overlap.area.tolist()

    return overlapAreaList[0] / areaBigger if len(overlapAreaList) > 0 else 0

QuadProposalCNNDir = r'QuadProposals/'
from matplotlib import pyplot as plt
import cv2
class PreprocessConfig:
    def __init__(self):
        self.CNMWeightsPath = './CornerKeys/2019_01_31_CNN_v03/CNN_2char_bn_unified.ckpt'
        self.method = 'component_search' # either 'component_search' or 'quad_proposal'

        self.qdCNNCfg = CNNQuadDetector.CNNQuadDetectorCfg()

        self.skipStep = 1
        self.skipPatternExtraction = False
        self.outputPatternRecogInfo = False
        self.outputPatternRecogInfoExt = 'png'
        self.outputPatternExtractionInfo = False
        self.outputBinaryImg = False
        self.outputCornersRecogInfo = False
        self.select = []
        self.startShift = 0
        self.useSoftMaxCotOff = False
        self.softMaxCutOffVal = 0.5
        self.extName = "pgm"
        self.outputPatternPixels = False
        self.outputCorners = False
        self.numProcess = 6
        # White Pattern Extranction Config
        self.minBlackComponentsSize = 180
        self.minWhiteComponentsSize = 15
        self.maxWhiteComponentsSize = 3000
        self.applyPatternFilter = True
        self.maxEdgeRatio = 2.8
        self.maxAngle = 0.75 * 3.1415926
        self.minAngle = 0.1 * 3.1415926
        # Erosion size control
        self.erosionSize = 2
        # Corner detection control
        self.findCornerOnBinarizedImg = True

        if self.findCornerOnBinarizedImg:
            self.minCornerDistance = 5
            self.blockSize = 5    # block size for goodFeaturesToTrack
            self.cornerQualityLevel = 0.2
            self.subPixRefineWindowSize = 5
        else:
            self.minCornerDistance = 5
            self.blockSize = 5    # block size for goodFeaturesToTrack
            self.cornerQualityLevel = 0.03
            self.subPixRefineWindowSize = 2

        self.pattern = ""
        self.patternPix = ""
        self.allcorner = ""
        self.patterncorner = ""
        self.binary = ""
        self.verbose = False

import sys
sys.path.insert(0, QuadProposalCNNDir)

from QuadProposals import CNNQuadDetector

def convertPredictionToCornerAnnotation(labelData):
    verts =[{'vert':v, 'code':[]} for v in  labelData['corners']]

    for qis, code in zip( labelData['accept_qi'],  labelData['code']):
        for order, vI in enumerate(qis):
            verts[vI]['code'].append(code.upper()+str(order+1))

    verts = [v for v in verts if len(v['code'])!=0]

    return verts

def predToCorr(predFile):
    labeler = Data.CornerLabeler()

    corners, cornerKeys, _ = Data.readProcessJsonFile(predFile)

    # corner keys to uIds & corrs without consistency check
    cornerUIds = labeler.labelCorners(cornerKeys, consistencyCheckScheme='discard',
                                                  )
    corr = labeler.cornerUIdsToCorrList(corners, cornerUIds, )

    return corr, corners, cornerKeys

def drawRecogResultsPDF(outPDFName, img, predFile, drawCorners=False):
    print("Drawing: ", outPDFName)
    predData = json.load(open(predFile))
    fig, ax = plt.subplots()
    ax.imshow(img, vmin=0, vmax=255, interpolation = 'nearest', cmap=plt.get_cmap('gray'))
    for i, p in enumerate(predData['accept_qi']):
        # if len(p.cornersCorrectOrder) == 0: continue

        x_coords = [predData['corners'][cid][0] for cid in p]
        x_coords.append(predData['corners'][p[0]][0])
        y_coords = [predData['corners'][cid][1] for cid in p]
        y_coords.append(predData['corners'][p[0]][1])

        ax.plot(x_coords, y_coords, '-', linewidth=0.05, color='red')
        ax.text(np.mean(x_coords[0:4]), np.mean(y_coords[0:4]), predData['code'][i], \
                    verticalalignment='center', horizontalalignment='center', fontsize=1, color='green')
        ax.text(x_coords[0], y_coords[0], '0', verticalalignment='top', horizontalalignment='left', fontsize=0.3, color='red')

    if drawCorners:
        corners = np.array([c for c in predData['corners']])
        ax.plot(corners[:, 0], corners[:, 1], 'x', color='green', markeredgewidth=0.06, markersize=0.3)
    ax.axis('off')
    fig.savefig(outPDFName, dpi = 1000, bbox_inches='tight', pad_inches=0)
    plt.close()
    print('Saved pdf file: ', outPDFName)


if __name__ == '__main__':
    # this is not raw annotation, this is already converted
    # inFolder = r'E:\Dropbox\mcproj\2019_12_26_LK_old_annot_imgs_quads'
    # inFolder = r'NewSuit'
    # inAnnotPredFileFolder = r'E:\Dropbox\mcproj\2019_12_26_LK_old_annot_imgs_quads\ConvertToPredData'
    # outFolder = r'output/S33_Evaluate_EndToEnd_Test'

    inFolder = r'NewSuit\Annotation\QuadAnnotaion'
    inAnnotPredFileFolder = r'NewSuit\Annotation\QuadAnnotaion\ConvertToPredData'
    outFolder = r'output/S33_Evaluate_EndToEnd_Test'

    # inFolder = r'E:\Dropbox\Mocap_Networks\cws_detector (1)\test_02'
    # inAnnotPredFileFolder = r'E:\Dropbox\Mocap_Networks\cws_detector (1)\test_02\ConvertToPredData'

    os.makedirs(outFolder, exist_ok=True)

    imgFiles = sortedGlob(join(inFolder, '*.pgm'))

    cfg = PreprocessConfig()
    # type 2
    cfg.qdCNNCfg.cornerdet_sess = 'QuadProposals/nets/28_renamed.ckpt'
    cfg.qdCNNCfg.rejector_sess = 'QuadProposals/nets_rejector/75_rejector_v04.ckpt'
    cfg.qdCNNCfg.recognizer_sess = 'QuadProposals/nets_recognizer/CNN_100_gen7_and_synth.ckpt'
    cfg.qdCNNCfg.usePreRejector = False

    # cfg.qdCNNCfg.cornerdet_sess = './_Nets/Cornerdet/20200105_13h57m_epoch_60.ckpt'
    # # rejector
    # cfg.qdCNNCfg.rejector_sess = './_Nets/Rejector/200117_rejector_321.ckpt'
    # # recognizer
    # cfg.qdCNNCfg.recognizer_sess = './_Nets/Recognizer/CNN_108_gen12_auto00001.ckpt'
    cfg.CIDFile = 'QuadProposals/CID/CID_list.txt'

    CNNModel = CNNQuadDetector.CNNQuadDetector(cfg.qdCNNCfg)
    CNNModel.restoreCNNSess()

    allErrs = []
    numberAllAnnot = 0
    numberAllDetection = 0
    numberAllExtra = 0
    # imgFiles = [imgFiles[i] for i in [0, 2, 3, 4, 5, 7, 8, 9]]

    for inFile in imgFiles:

        # inFile = r'Data\test_02\I05602.pgm'
        outPFile = join(outFolder, Path(inFile).stem + 'OutQuadProposal.txt')
        outH5File =join(outFolder, Path(inFile).stem +  r'OutQuadProposal.h5')
        predictionFile = join(outFolder, Path(inFile).stem + 'predictionFile.json')

        annotPredDataFile = join(inAnnotPredFileFolder, Path(inFile).stem + '_quads_PredData.json')
        cfg.outputPatternRecogInfo = join(outFolder, Path(inFile).stem + 'predictionFile.pdf')
        CNNQuadDetector.processSequenceQuadProposalSingleFile(inFile, outPFile, outH5File,
                                                                  qDetector=CNNModel, config=cfg,
                                                                  predictionFile=predictionFile)

        img = cv2.imread(inFile, cv2.IMREAD_GRAYSCALE)
        # drawRecogResultsPDF(join(outFolder, Path(inFile).stem +  r'_Pred.png'), img, predictionFile,)
        # drawRecogResultsPDF(join(outFolder, Path(inFile).stem +  r'_Annot.png'), img, annotPredDataFile,)

        annotPredData = json.load(open(annotPredDataFile))
        predictionData = json.load(open(predictionFile))
        predictedVerts = convertPredictionToCornerAnnotation(predictionData)
        # # print(predictedVerts)

        # codeToCoordMap = {}
        # observed = np.zeros(len(cornerAnnot))
        # for cId, c in enumerate(cornerAnnot):
        #     for code in c['code']:
        #         codeToCoordMap[code] = {'vert':c['vert'], 'Id':cId}
        #
        #
        # #evaluate predicted Verts
        # errors = []
        # notAnnotated = []
        # for predC in predictedVerts:
        #
        #     annotC = codeToCoordMap.get(predC['code'][0])
        #     if len(predC['code'])> 1 and annotC is None:
        #         annotC = codeToCoordMap.get(predC['code'][1])
        #     if annotC is not None:
        #         diff = np.array(predC['vert']) - np.array(annotC['vert'])
        #         dis = np.sqrt(diff[0]**2 + diff[1]**2)
        #         errors.append(dis)
        #         observed[annotC['Id']] = True
        #     else:
        #         notAnnotated.append(annotC)

        predCorrs, predCorners, predCornerKeys = np.array(predToCorr(predictionFile))
        annotCorrs, annotCorners, annotCornerKeys =  np.array(predToCorr(annotPredDataFile))

        predCorrs = np.array(predCorrs)
        annotCorrs = np.array(annotCorrs)

        annotCIds = np.where(annotCorrs[:,0]!=-1.0)[0]
        predCIds = np.where(predCorrs[:,0]!=-1.0)[0]

        observed = np.intersect1d(annotCIds, predCIds)
        diff = np.array(predCorrs)[observed, :] - np.array(annotCorrs)[observed, :]
        dis = np.sqrt(diff[0] ** 2 + diff[1] ** 2)
        allErrs.append(dis)

        extra = np.setdiff1d(predCIds, annotCIds)

        # correct recognition
        # enumerate predicted quads match with annotated ones to check
        predictionCorrectness = []
        extraQuads = []
        for iq, qi in enumerate(predictionData['accept_qi']):
            annotatedQuqad = False
            q = [predictionData['corners'][iV] for iV in qi]
            for code, acceptedQV in zip(annotPredData['code'], annotPredData['accept_qv']):
                if overlapAreaRatio(q, acceptedQV) > 0.95:
                    annotatedQuqad = True
                    if code.upper() == predictionData['code'][iq].upper():
                        predictionCorrectness.append(True)
                    else:
                        print("Wrong prediction: ", code.upper(),' ->', predictionData['code'][iq].upper() )
                        predictionCorrectness.append(False)
                    break
            if not annotatedQuqad:
                extraQuads.append(iq)

        print('Quad recog Accuracy: ', np.where(predictionCorrectness)[0].shape[0] / len(predictionCorrectness))
        print('Found ', np.where(predictionCorrectness)[0].shape[0], ' of ', len(annotPredData['accept_qv']), ' annotated quads.')

        # print('Max localization error', np.max(dis))
        # print('Average localization error', np.mean(dis))
        # # print('Found ',  np.where(observed)[0].shape[0], ' of ',  len(cornerAnnot), 'corners')
        # print('Found ',  len(observed), ' of ',  len(annotCIds), 'corners')
        # print('Number of extra pointsL: ', len(extra))


        allErrs.append(dis)
        numberAllAnnot += len(annotCIds)
        numberAllDetection += len(observed)
        numberAllExtra += len(extra)

    allErrs = np.concatenate(allErrs)
    print('**************************************************')
    print('Max localization error', np.max(allErrs))
    print('Average localization error', np.mean(allErrs))
    # print('Found ',  np.where(observed)[0].shape[0], ' of ',  len(cornerAnnot), 'corners')
    print('Found ', numberAllDetection, ' of ', numberAllAnnot, 'corners')
    print('Number of extra pointsL: ', numberAllExtra)



