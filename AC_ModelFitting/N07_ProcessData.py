from matplotlib import pyplot as plt
import numpy as np
from os.path import join
from glob import glob

if __name__ == '__main__':
    # inFolder = r'F:\WorkingCopy2\2020_04_20_DifferentiableRendererTest\Param_Sig0.0001_BRange0.0001_Fpp50_BodyOnlyFalse'
    # inFolders = glob(r'F:\WorkingCopy2\2020_04_20_DifferentiableRendererTest\*')
    inFolders = [
        # r'F:\WorkingCopy2\2020_04_20_DifferentiableRendererTest\Param_Sig0.0001_BRange0_Fpp20_BodyOnlyFalse'
        r'F:\WorkingCopy2\2020_04_20_DifferentiableRendererTest\Param_Sig0.0001_BRange0.0001_Fpp20_BodyOnlyFalse'
    ]
    for inFolder in inFolders:
        nBodyJoints = 22

        losses = np.load(join(inFolder, 'Losses.npy'))
        print(losses)

        plt.plot(losses)
        plt.savefig(join(inFolder, 'Loss.png'))

        # plt.waitforbuttonpress()

        poses = np.load(join(inFolder, 'Poses.npy'))
        posesTarget = np.load(join(inFolder, 'PoseTarget.npy'))
        print(poses.shape)
        print(posesTarget)

        bodyJDiff = [((p[:3*nBodyJoints] - posesTarget[:3*nBodyJoints])**2).mean() for p in poses]
        handJDiff = [((p[3*nBodyJoints:] - posesTarget[3*nBodyJoints:])**2).mean() for p in poses]

        t = np.linspace(0, poses.shape[1])
        plt.figure()
        plt.plot( bodyJDiff, label='body')
        plt.plot( handJDiff, label='hand')
        plt.legend()
        plt.savefig(join(inFolder, 'PoseDiff.png'))
        # plt.waitforbuttonpress()
