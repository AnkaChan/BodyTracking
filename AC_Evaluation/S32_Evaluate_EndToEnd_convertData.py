from Utility import *
# convert from quad annotation to corner annotation
import json

from SuitCapture import Camera

def convertToCornerAnnotation(labelData):
    verts =[{'corners':v, 'code':[]} for v in  labelData['verts']]

    for qis, code in zip( labelData['indices'],  labelData['codes']):
        for order, vI in enumerate(qis):
            verts[vI]['code'].append(code.upper()+str(order+1))

    verts = [v for v in verts if len(v['code'])!=0]

    return verts

def convertToPredAnnotation(labelData):
    # verts =[{'corners':v, 'code':[]} for v in  labelData['verts']]
    #
    # for qis, code in zip( labelData['indices'],  labelData['codes']):
    #     for order, vI in enumerate(qis):
    #         verts[vI]['code'].append(code.upper()+str(order+1))
    #
    # verts = [v for v in verts if len(v['code'])!=0]

    predData = {
        'corners': labelData['verts'],
        'code': labelData['codes'],
        'accept_qi': labelData['indices'],
        'accept_qv': [ [labelData['verts'][iC] for iC in qi]  for qi in labelData['indices']],
        'recog_dens4a': [[1 for i in range(25)] for j in range(len(labelData['indices']))],
        'recog_dens4b': [[1 for i in range(25)] for j in range(len(labelData['indices']))]

    }

    return predData

if __name__ == '__main__':
    # this is not raw annotation, this is already converted
    # inFolder = r'E:\Dropbox\Mocap_Networks\cws_detector (1)\test_02'
    inFolder = r'C:\Code\MyRepo\03_capture\BodyTracking\AC_Evaluation\NewSuit\Annotation\QuadAnnotaion'
    outFolder = join(inFolder,
                     'ConvertToPredData')

    os.makedirs(outFolder, exist_ok=True)
    labelFiles = sortedGlob(join(inFolder, '*.json'))

    for file in labelFiles:
        labelData = json.load(open(file))
        # print(labelData)

        predData = convertToPredAnnotation(labelData)
        # print(verts)

        json.dump(predData, open(join(outFolder, Path(file).stem+'_PredData.json'), 'w'))


