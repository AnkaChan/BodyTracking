from S04_FigureOutTheDeformationModel_ import *

if __name__ == '__main__':
    outFolder = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487'
    parameterFileSmplsh = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\TextureCompletionFitting\03067\PoseParam_00077.npz'

    inSkelData = join(outFolder, 'S01_Combined_Lada_HandHead.json')
    numCoarseJoints = 16
    skelData = json.load(open(inSkelData))

    skelData['HeadJointIds'] = [numCoarseJoints, numCoarseJoints+1]
    skelData['HandJointIds'] = list(range(numCoarseJoints+2,50))

    json.dump(skelData, open(inSkelData, 'w'))