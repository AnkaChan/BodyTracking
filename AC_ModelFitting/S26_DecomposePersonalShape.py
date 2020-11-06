import numpy as np

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


if __name__ == '__main__':
    inParamNpz = r'..\Data\KateyBodyModel\FitParamsWithPersonalShape.npz'

    betaFile = r'..\Data\KateyBodyModel\beta.npy'
    personalShapeFile = r'..\Data\KateyBodyModel\PersonalShape.npy'

    pose, trans, betas, personalShape = loadCompressedFittingParam(inParamNpz, True)

    np.save(betaFile, betas, )
    np.save(personalShapeFile, personalShape * 1000, )