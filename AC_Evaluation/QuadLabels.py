import json
import itertools
import math
import cv2
import numpy as np
from matplotlib import pyplot as plt

import LabelMeFileIO

# if __name__ == '__main__':
#     import LabelMeFileIO
# else:
#     from . import LabelMeFileIO

def refineCorners(corners, img):
    pts = np.array(corners).astype(np.float32)
    pts = np.expand_dims(pts, axis=1)

    stop_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                     30, 0.001)
    ptsSubPix = cv2.cornerSubPix(img, pts, (2, 2), (-1, -1), stop_criteria)
    return np.squeeze(ptsSubPix).tolist()

def coordsDistance(p1, p2):
     return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

class QuadLabels:
    def __init__(self):
        self.verts = []
        self.indices = []
        self.codes = []
        self.flags = []
        self.maxTolerancePixelShift = 3

    def quarryVertId(self, qp):
        for i, p in enumerate(self.verts):
            if coordsDistance(p, qp) < self.maxTolerancePixelShift:
                return i
        return -1

    def refineCorners(self, img):
        pts = np.array(self.verts).astype(np.float32)
        pts = np.expand_dims(pts, axis=1)

        stop_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                         30, 0.001)
        ptsSubPix = cv2.cornerSubPix(img, pts, (2, 2), (-1, -1), stop_criteria)
        self.verts = np.squeeze(ptsSubPix).tolist()

    def readQuadLabelFile(self, labelFile, toUpperCode = True):
        with open(labelFile, 'r') as myfile:
            #with open('wenxian_02241.json', 'r') as myfile:
            data=myfile.read()
            # parse file
            obj = json.loads(data)

            self.verts = obj['verts']
            self.indices = obj['indices']
            self.codes = obj['codes']
            if obj.get('flags') != None:
                self.flags = obj['flags']
            if toUpperCode:
                self.codes = [c.upper() for c in self.codes]
                #for c in self.codes:
                #    print('c before:', c)
                #    c = c.upper() 
                #    print('c after:', c)

    def changeToPolygonLabels(self):
        pLabelSet = [
            {
                'label':code,
                'points':[self.verts[i] for i in indices]
                }
            for indices, code in zip(self.indices, self.codes)
        ]
        return pLabelSet

    def quadIndicesToVertsSeq(self, indices):
        return [self.verts[id] for id in indices]

    def writeAsLabelMeLabelFile(self, outLabelmeFile, imgFile):
        pLabelSet = self.changeToPolygonLabels()
        LabelMeFileIO.writeAsLabelMeLabelFileWithImg(outLabelmeFile, pLabelSet, imgFile)


def drawQuadLabel(outPDFName, qLabelSet, img):
    fig, ax = plt.subplots()
    ax.imshow(img, vmin=0, vmax=255, interpolation = 'nearest', cmap=plt.get_cmap('gray'))
    pts = np.array(qLabelSet.verts)
    for i, (indices, code) in enumerate(zip(qLabelSet.indices, qLabelSet.codes)):
        x_coords = pts[[indices[0],indices[1],indices[2],indices[3],indices[0]], 0]
        y_coords = pts[[indices[0],indices[1],indices[2],indices[3],indices[0]], 1]
        ax.plot(x_coords, y_coords, '-', linewidth=0.02, color='red')
        ax.text(np.mean(x_coords[0:4]), np.mean(y_coords[0:4]), code, \
                    verticalalignment='center', horizontalalignment='center', fontsize=1, color='green')
        ax.text(x_coords[0], y_coords[0], '0', verticalalignment='top', horizontalalignment='left', fontsize=0.3, color='red')

    fig.savefig(outPDFName, dpi = 2000)
    plt.close()
    print('Saved pdf file: ', outPDFName)

defaultColorFlagMap = {
    'None':'good',
    '[255, 0, 0, 128]':'margin'
    }

def polygonLabelsToQuad(polygonLabelSet, readFlagFromColor = False, colorFlagMap = defaultColorFlagMap):
    qLabels = QuadLabels()
    for l in polygonLabelSet:
        squareIndices = []
        for p in l['points']:
            vId = qLabels.quarryVertId(p)
            if vId != -1:
                squareIndices.append(vId)
            else:
                vId = len(qLabels.verts)
                qLabels.verts.append(p)
                squareIndices.append(vId)
        qLabels.indices.append(squareIndices)
        qLabels.codes.append(l['label'])
        if readFlagFromColor:
            lineColor = l['line_color']
            flag = colorFlagMap.get(str(lineColor))
            if flag == None:
                raise Exception('Color does not exist in color flag map: ' + str(lineColor))
            else:
                qLabels.flags.append(flag)
    return qLabels

