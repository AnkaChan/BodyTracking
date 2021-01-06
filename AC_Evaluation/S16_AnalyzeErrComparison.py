import numpy as np
from matplotlib import  pyplot as plt
import matplotlib
from Utility import *
import tqdm
import json

def loadALlErrs(errFolder):
    errFiles = sortedGlob(join(errFolder, '*.json'))[:100]
    allErrs = []

    log_x = False
    # log_x = True

    for errF in tqdm.tqdm(errFiles):
        # errs = json.load(open(errF))['Errs']
        errs = json.load(open(errF))
        for err in errs:
            allErrs.extend(err)

    return allErrs

if __name__ == '__main__':
    # triangulateFolder = r'F:\WorkingCopy2\2020_01_16_Lada_FinalAnimations\WholeSeq\TriangulationType1Only'
    triangulateFolderWithoutConsis = r'F:\WorkingCopy2\2020_12_22_ReconstructionEvaluation\Recon\WithoutConsis'
    triangulateFolderWithConsis = r'F:\WorkingCopy2\2020_12_22_ReconstructionEvaluation\Recon\WithConsis'
    triangulateFolderConsisRansac = r'F:\WorkingCopy2\2020_12_22_ReconstructionEvaluation\Recon\WithConsisRansac'

    outFolder = r'outout/S16_ReprojErrsAnalysis'

    # errsWithoutConsis = loadALlErrs(triangulateFolderWithoutConsis)
    # errsWithConsis = loadALlErrs(triangulateFolderWithConsis)
    # errsConsisRansac = loadALlErrs(triangulateFolderConsisRansac)

    # json.dump(errsWithoutConsis, open('errsWithoutConsis.json', 'w'))
    # json.dump(errsWithConsis, open('errsWithConsis.json', 'w'))
    # json.dump(errsConsisRansac, open('errsConsisRansac.json', 'w'))

    errsWithoutConsis = json.load( open('errsWithoutConsis.json',))
    errsWithConsis = json.load(open('errsWithConsis.json', ))
    errsConsisRansac = json.load( open('errsConsisRansac.json', ))

    reprojErrCutoff = 6

    rpjErrThres = 100
    errFiles = sortedGlob(join(triangulateFolderWithoutConsis, '*.json'))[:100]
    allErrs = []

    log_x = False
    # log_x = True

    os.makedirs(outFolder, exist_ok=True)
    for errF in tqdm.tqdm(errFiles):
        # errs = json.load(open(errF))['Errs']
        errs = json.load(open(errF))
        for err in errs:
            # allErrs.extend(err)
            largeErrIds = np.where(errs > rpjErrThres)[0]


    p_95 = np.percentile(errsConsisRansac, 95)  # return 50th percentile, e.g median.
    p_99 = np.percentile(errsConsisRansac, 99)  # return 50th percentile, e.g median.
    p_999 = np.percentile(errsConsisRansac, 99.9)  # return 50th percentile, e.g median.
    p_9999 = np.percentile(errsConsisRansac, 99.99)  # return 50th percentile, e.g median.
    print('p_95:', p_95)
    print('p_99:', p_99)
    print('p_999:', p_999)
    print('p_9999:', p_9999)

    font = {'family': 'normal',
            # 'weight': 'bold',
            'size': 20}

    matplotlib.rc('font', **font)
    # n_bins = 1000
    #
    # hist, bins, _ = plt.hist(allErrs, bins=n_bins)
    #
    # if log_x:
    #     plt.figure()
    #     logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    #     plt.hist(allErrs, bins=logbins)
    #
    # # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel('Reprojection Errors')
    # plt.ylabel('Number of Points (in pixel)')
    # plt.show()

    kwargs = dict(alpha=0.5, bins=200, stacked=True)

    plt.hist(errsWithoutConsis, **kwargs, color='b', label='Without outlier filtering')
    # plt.hist(errsWithConsis, **kwargs, color='b', label='With Consistency Check')
    plt.hist(errsConsisRansac, **kwargs, color='r', label='With outlier filtering')
    plt.yscale('log')
    plt.gca().set(title='Reprojection errors', ylabel='Errs')
    # plt.xlim(0, 75)
    plt.legend();
    plt.savefig(join(outFolder, 'ReprojErrsCompr.png'), dpi=300)
    plt.show()

    errsConsisRansacCutoff = []
    for err in errsConsisRansac:
        if err< reprojErrCutoff:
            errsConsisRansacCutoff.append(err)

    plt.figure('ReprojErrWithOutlierLessThan'+str(reprojErrCutoff))
    # plt.hist(errsConsisRansacCutoff, **kwargs, color='r', label='With outlier filtering')
    plt.hist(errsConsisRansacCutoff, **kwargs, label='With outlier filtering')
    plt.xlim(0, reprojErrCutoff)
    plt.yscale('log')
    plt.gca().set(title='Reprojection errors with outlier filtering', ylabel='Errs')
    # plt.xlim(0, 75)
    plt.legend();
    plt.savefig(join(outFolder, 'ReprojErrWithOutlierLessThan' + str(reprojErrCutoff)+'.png'), dpi=300)
    plt.show()

