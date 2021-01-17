from S24_CalibrationErrs import *
from Utility import *
import pyvista as pv
from M01_ARAPDeformation import *
from SkelFit.Data import *

from M02_ObjConverter import removeVertsFromMeshFolder

if __name__ == '__main__':
    inChunkedFile = r'F:\WorkingCopy2\2021_01_04_NewModelFitting\Inputs\Katey_CalibrationSeqs_54_6054.json'
    outChunkFileKillOutliers = r'F:\WorkingCopy2\2021_01_04_NewModelFitting\Inputs\Katey_CalibrationSeqs_54_6054_outlierFiltered.json'

    # initialErrFile = r'F:\WorkingCopy2\2021_01_09_ActorTuningVis\Evaluation\Training_Initial\SLap_SBiLap_True_TLap_1_JTW_0.5_JBiLap_0_Step1_Overlap0\Init\Errs\Errors.json'
    # finalErrFile = r'F:\WorkingCopy2\2021_01_09_ActorTuningVis\Evaluation\Training_Final\SLap_SBiLap_True_TLap_1_JTW_0.5_JBiLap_0_Step1_Overlap0\Init\Errs\Errors.json'
    # outFileName = 'FittingErrsCompr_train.png'

    initialErrFile = r'F:\WorkingCopy2\2021_01_09_ActorTuningVis\Evaluation\Test_Initial\SLap_SBiLap_True_TLap_1_JTW_0.5_JBiLap_0_Step1_Overlap0\Init\Errs\Errors.json'
    finalErrFile = r'F:\WorkingCopy2\2021_01_09_ActorTuningVis\Evaluation\Test_Final\SLap_SBiLap_True_TLap_1_JTW_0.5_JBiLap_0_Step1_Overlap0\Init\Errs\Errors.json'
    outFileName = 'FittingErrsCompr_test.png'

    # inBatchFile, errs = killOutliersChunked(inChunkedFile, finalErrFile, 100, outChunkFileKillOutliers)

    outputFolder = join('output', 'S25_EvaluateActorTuning')

    os.makedirs(outputFolder, exist_ok=True)
    intialErrs = json.load(open(initialErrFile))

    intialErrs = np.concatenate([np.array(err)[np.where(np.array(err)!=0)] for err in intialErrs])

    finalErrs = json.load(open(finalErrFile))
    finalErrs = np.concatenate([np.array(err)[np.where(np.array(err)!=0)] for err in finalErrs])

    kwargs = dict(alpha=0.5, bins=400, stacked=True)
    # fig = plt.figure(constrained_layout=True, tight_layout=True)
    plt.rcParams["figure.figsize"] = (6, 2.5)  # (w, h)
    fig = plt.figure(tight_layout=True)
    # gs = GridSpec(1, 1, figure=fig)

    # linear
    plt.hist(intialErrs, **kwargs, color='b', label='Before actor tuning', )
    # plt.hist(errsWithConsis, **kwargs, color='b', label='With Consistency Check')
    plt.hist(finalErrs, **kwargs, color='r', label='After actor tuning', )
    plt.yscale('log')
    # plt.xscale('log')
    # plt.gca().set(ylabel='Bin count', xlabel='(a) Fitting error on the training set[mm]')
    plt.gca().set(ylabel='Bin count', xlabel='(b) Fitting error on the test set[mm]')
    # plt.ylim([0, 1e5])
    plt.xlim([0, 150])
    plt.legend();
    plt.savefig(join(outputFolder, outFileName), dpi=1000, bbox_inches="tight")
    plt.show()