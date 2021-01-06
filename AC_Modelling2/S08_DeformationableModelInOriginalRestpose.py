from S04_FigureOutTheDeformationModel_ import *
from SkelFit.Data import *
from SkelFit.SkeletonModel import *
if __name__ == '__main__':
    outFolder = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487'
    inSkelFile = join(outFolder, 'S01_Combined_Lada_HandHead.json')
    outSkelFile = join(outFolder, 'S02_Combined_Lada_HandHead_OriginalRestpose.json')

    inOriginalRestpose = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487\Complete_withHeadHand_XYZOnly_tri_backToRestpose.obj'
    outOriginalRestpose = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487\Complete_withHeadHand_XYZOnly_tri_backToRestpose.ply'
    inOriginalJoints = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487\CoarseJoints_restpose.obj'
    completeMeshBackToRestpose = r'C:\Code\MyRepo\03_capture\BodyTracking\Data\2020_12_27_betterCoarseMesh\Mesh1487\BackToRestpose\Complete_withHeadHand_XYZOnly_tri_backToRestpose.ply'

    headVIdsFile = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487\HeadVIds.Json'
    handVIdsFile = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487\HandVIds.json'

    headVIds = json.load(open(headVIdsFile))
    handVIds = json.load(open(handVIdsFile))

    vIdsNeedCorrsToSmplsh = headVIds + handVIds

    completeMeshBackToRestpose = pv.PolyData(completeMeshBackToRestpose)

    originalRestpose = pv.PolyData(inOriginalRestpose)
    isolatedVIds = getIsolatedVerts(originalRestpose)
    originalJoints = pv.PolyData(inOriginalJoints)

    originalRestpose.points[vIdsNeedCorrsToSmplsh, :] = completeMeshBackToRestpose.points[vIdsNeedCorrsToSmplsh, :]
    originalRestpose.points[isolatedVIds, :] = [0,0,-1]
    originalRestpose.save(outOriginalRestpose)

    skelData = json.load(open(inSkelFile))

    originalRestpose = originalRestpose.points.tolist()
    for i in range(len(originalRestpose)):
        skelData['VTemplate'][0][i] = originalRestpose[i][0]
        skelData['VTemplate'][1][i] = originalRestpose[i][1]
        skelData['VTemplate'][2][i] = originalRestpose[i][2]

    joints = originalJoints.points.tolist()
    for i in range(16):
        skelData['JointPos'][0][i] = joints[i][0]
        skelData['JointPos'][1][i] = joints[i][1]
        skelData['JointPos'][2][i] = joints[i][2]

    json.dump(skelData, open(outSkelFile, 'w'))