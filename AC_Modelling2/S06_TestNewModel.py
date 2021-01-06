from S04_FigureOutTheDeformationModel_ import *

if __name__ == '__main__':
    outFolder = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487'
    parameterFileSmplsh = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\TextureCompletionFitting\03067\PoseParam_00077.npz'

    inSmplshSkelData = join(outFolder, 'S01_Combined_Lada_HandHead.json')

    numCoarseJoints=16
    jointsIdToReduce = [10, 11, 12, 15, 20, 21, 22, 23]  # joints we removed from
    toSmplshJointsCorrs = []
    iJCoarse = 0
    for iJ in range(24):  # smpl has 24 joints
        if iJ not in jointsIdToReduce:
            toSmplshJointsCorrs.append([iJCoarse, iJ])
            iJCoarse += 1

    toSmplshJointsCorrs = toSmplshJointsCorrs + [[numCoarseJoints, 12], [numCoarseJoints + 1, 15]] + [
        [numCoarseJoints + 2 + i, 20 + i] for i in range(32)]

    toSmplshJointsCorrs = np.array(toSmplshJointsCorrs)

    parameterSmplsh = np.load(parameterFileSmplsh)
    vRestpose, J, weights, poseBlendShape, kintreeTable, parent, faces = readSkeletonData(inSmplshSkelData)
    # r, t = readSkelParams(parameterFileCoarse)
    Rs = axisAnglesToRotation(parameterSmplsh['pose'].reshape((-1, 3))[toSmplshJointsCorrs[:, 1], :])
    trans = parameterSmplsh['trans']
    # newJsSMPLSH = transformJoints(Rs, parameterSmplsh['trans'], J, parent)
    verts = deformVerts(vRestpose, Rs, trans, J, weights, kintreeTable, parent)
    # print('newJs', newJs)
    write_obj('Test.obj', verts, faces)