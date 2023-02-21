from S38_GenVideoClips import *
def preprocessImg(img, camParam, turnToRGB=False):
    # convert to Rgb
    # Undist images
    # img = cv2.imread(inImgFile, cv2.IMREAD_GRAYSCALE)
    if turnToRGB:
        imgColor = cv2.cvtColor(img, cv2.COLOR_BAYER_GB2BGR_EA )

    imgColor = cv2.resize(imgColor, outputSize)

    fx = camParam['fx']
    fy = camParam['fy']
    cx = camParam['cx']
    cy = camParam['cy']
    intrinsic_mtx = np.array([
        [fx/2, 0.0, cx/2, ],
        [0.0, fy/2, cy/2],
        [0.0, 0.0, 1],
    ])

    undistortParameter = np.array(
        [camParam['k1'], camParam['k2'], camParam['p1'], camParam['p2'],
         camParam['k3'], camParam['k4'], camParam['k5'], camParam['k6']])

    imgColorUndist = cv2.undistort(imgColor, intrinsic_mtx, undistortParameter)

    return imgColorUndist

if __name__ == '__main__':
    inImageFolder = r'X:/MocapProj/2019_12_13_Lada_Capture'
    inCamParam = r'X:/MocapProj/2019_12_13_Lada_Capture/CameraParameters/cam_params.json'

    outFolder = r'F:\WorkingCopy2\2021_05_12_ReverseRenderingAnimation\Animation'
    outputSize = (2000, 1080,)
    fps = 30
    convertRefImgToRGB = True

    camIds = [8,]
    camNames = 'ABCDEFGHIJKLMNOP'
    # frameNames = ['08102', '08034']
    frameNames = [str(frame).zfill(5) for frame in range(6141, 8141)]
    # frameNames = [str(frame).zfill(5) for frame in range(6141, 6141+20)]
    camParamsAll = json.load(open(inCamParam))

    out = cv2.VideoWriter(join(outFolder, 'Ref.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, outputSize)

    for iCam in camIds:
        camName = camNames[iCam]

        camParam = camParamsAll['cam_params'][str(iCam)]

        # inCleanPlateImg = cv2.resize(inCleanPlateImg, (2000, 1080,))

        for frame in  tqdm.tqdm(frameNames, desc='Generating for cam: '+camName):
            inRefImgF = join(inImageFolder, camName, camName + frame + '.pgm')
            imRef = preprocessImg(cv2.imread(inRefImgF, cv2.IMREAD_GRAYSCALE), camParam, convertRefImgToRGB)
            # imRef = cv2.resize(imRef, outputSize)
            # cv2.imshow('imRef', imRef,)
            # cv2.waitKey(10)
            out.write(imRef)

    out.release()