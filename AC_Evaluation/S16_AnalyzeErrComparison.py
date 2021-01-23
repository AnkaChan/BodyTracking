import numpy as np
from matplotlib import  pyplot as plt
import matplotlib
from Utility import *
import tqdm
import json
from matplotlib.gridspec import GridSpec


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

    outputFolder = r'output/S16_ReprojErrsAnalysis'

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

    os.makedirs(outputFolder, exist_ok=True)
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

    font = {
            # 'family': 'normal',
            # 'weight': 'bold',
            # 'size': 20
    }

    matplotlib.rc('font', **font)
    # n_bins = 1000
    #

    #
    # # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel('Reprojection Errors')
    # plt.ylabel('Number of Points (in pixel)')
    # plt.show()

    # kwargs = dict(alpha=0.5, bins=200, stacked=True)
    kwargs = dict(alpha=0.5, stacked=True)
    # fig = plt.figure(constrained_layout=True, tight_layout=True)
    plt.rcParams["figure.figsize"] = (6, 2.5)  # (w, h)
    fig = plt.figure(constrained_layout=True, tight_layout=True)
    gs = GridSpec(1, 1, figure=fig)

    # linear
    hist, bins, _ = plt.hist(errsWithoutConsis, bins=500)

    plt.figure()
    logbins = np.logspace(np.log10(bins[0]+1e-4), np.log10(bins[-1]), len(bins))

    plt.hist(errsWithoutConsis, **kwargs, color='b', label='Without outlier filtering', rwidth=1, bins=logbins)
    # plt.hist(errsWithConsis, **kwargs, color='b', label='With Consistency Check')
    plt.hist(errsConsisRansac, **kwargs, color='r', label='With outlier filtering', rwidth=1, bins=logbins)
    plt.yscale('log')
    plt.xscale('log')
    plt.gca().set(ylabel='Bin count', xlabel='(a) Reprojection error of 3D reconstruction [pixels]')
    plt.ylim([0, 1e5])
    # plt.xlim([0, 300])
    plt.legend();
    plt.savefig(join(outputFolder, 'ReprojErrsCompr_xlog.png'), dpi=400, bbox_inches="tight")
    plt.show()

    errsConsisRansacCutoff = []
    for err in errsConsisRansac:
        if err< reprojErrCutoff:
            errsConsisRansacCutoff.append(err)

    # plt.figure('ReprojErrWithOutlierLessThan'+str(reprojErrCutoff))
    # # plt.hist(errsConsisRansacCutoff, **kwargs, color='r', label='With outlier filtering')
    # plt.hist(errsConsisRansacCutoff, **kwargs, label='With outlier filtering')
    # plt.xlim(0, reprojErrCutoff)
    # plt.yscale('log')
    # plt.gca().set(title='Reprojection errors with outlier filtering', ylabel='Errs')
    # # plt.xlim(0, 75)
    # plt.legend();
    # plt.savefig(join(outFolder, 'ReprojErrWithOutlierLessThan' + str(reprojErrCutoff)+'.png'), dpi=300)
    # plt.show()

    matplotlib.style.use('default')

    # mpl.rcParams['axes.prop_cycle'] = cycler(color='rbgcmyk')
    num_bins = 500
    plt.rcParams["figure.figsize"] = (6, 2.5)  # (w, h)
    fig = plt.figure(constrained_layout=True, tight_layout=True)
    gs = GridSpec(1, 1, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])

    # linear
    save_path = os.path.join(outputFolder, 'ReprojErrWithOutlierLessThan'+str(reprojErrCutoff) + ".png")

    ax1.hist(errsConsisRansacCutoff, bins=num_bins)
    ax1.set_yscale('log')
    ax1.set_xlim(left=0)
    ax1.set_xlabel('(b) Reprojection error  of 3D reconstruction [pixels]')
    ax1.set_ylabel('Bin count')
    # ax1.set_title(
    #     'Reprojection Errors | {} image points\n(mean={:.2f}, std={:.2f}, max={:.2f})'.format(len(reproj_errs), mean,
    #                                                                                           std, max_err))
    # ax1.legend(['bundle adjustment (6dof)'])
    plt.grid(False)
    plt.xlim([0, 6])
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    print(save_path)
    plt.show()
