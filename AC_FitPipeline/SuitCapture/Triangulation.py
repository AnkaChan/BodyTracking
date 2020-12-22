import numpy as np
import itertools
# C++ version of multiview triangulation
# void MultiCameraReconstructor::multiCamsDLT(const VecEP3f& controlPts, const std::vector<Eigen::Matrix4f>& controlMats, Eigen::Matrix<float, 3, 1>& X)
# {
# 	int numConstraints = controlPts.size();
# 	Eigen::MatrixXf A, b;
# 	if (numConstraints >= 2)
# 	{
# 		A.resize(2 * numConstraints, 4);
#
# 		for (size_t iC = 0; iC < numConstraints; iC++)
# 		{
# 			A.row(iC * 2) = controlPts[iC](0, 0) * controlMats[iC].row(2) - controlMats[iC].row(0);
# 			A.row(iC * 2 + 1) = controlPts[iC](1, 0) * controlMats[iC].row(2) - controlMats[iC].row(1);
# 		}
#
# 		b = -A.block(0, 3, 2 * numConstraints, 1);
# 		Eigen::MatrixXf AA = A.block(0, 0, 2 * numConstraints, 3);
#
# 		X = AA.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
# 	}
# 	else
# 	{
# 		X << 0, 0, -1;
# 	}
# 	return ;
# }

def projectPoints(points, projMat):
    pts2D = []
    for iV in range(points.shape[0]):
        p3D = points[iV, :]
        p3DH = np.vstack([p3D.reshape(3, 1), 1])

        ptsP = projMat @ p3DH
        ptsP = ptsP / ptsP[2]

        # print(ptsP)
        pts2D.append(ptsP[:2, 0])

    pts2D = np.array(pts2D)

    return pts2D

def mulCamsDLT(controlPts, camProjMats):
    numConstraints = len(controlPts)
    A = np.zeros((2*numConstraints, 4))

    if numConstraints >= 2:
        for iCam in range(numConstraints):
            A[2*iCam, :] = controlPts[iCam][0] * camProjMats[iCam][2, :] - camProjMats[iCam][0, :]
            A[2*iCam + 1, :] = controlPts[iCam][1] * camProjMats[iCam][2, :] - camProjMats[iCam][1, :]
        b = -A[:, 3]

        AA = A[:, :3]
        X = np.linalg.lstsq(AA, b)[0]

        reprojErrs = []
        for iCam in range(numConstraints):
            p3DH = np.vstack([X.reshape(3, 1), 1])

            ptsP = camProjMats[iCam] @ p3DH
            ptsP = ptsP / ptsP[2]

            err = np.sqrt((controlPts[iCam][0] - ptsP[0])**2 + (controlPts[iCam][1] - ptsP[1])**2 )

            reprojErrs.append(err)

    else:
        print("Need at least 2 constraints")
        assert False

    return X, np.array(reprojErrs)

def computeReprojErrMultiCam(X, controlPts, camProjMats):
    numConstraints = len(controlPts)
    reprojErrs = []
    for iCam in range(numConstraints):
        p3DH = np.vstack([X.reshape(3, 1), 1])

        ptsP = camProjMats[iCam] @ p3DH
        ptsP = ptsP / ptsP[2]

        err = np.sqrt((controlPts[iCam][0] - ptsP[0]) ** 2 + (controlPts[iCam][1] - ptsP[1]) ** 2)

        reprojErrs.append(err)
    return reprojErrs

def mulCamsRansac(controlPts, camProjMats, computeErrOnAllCam=True):
    numConstraints = len(controlPts)

    bestErr = -1
    bestX = 1

    cams = list(range(numConstraints))
    if numConstraints >= 2:
        for camPair in itertools.product(cams, repeat = 2):
            if camPair[0] == camPair[1]:
                continue
            X, errs = mulCamsDLT([controlPts[camPair[0]], controlPts[camPair[1]]], [camProjMats[camPair[0]], camProjMats[camPair[1]]])

            if computeErrOnAllCam:
                reprojErrs = computeReprojErrMultiCam(X, controlPts, camProjMats)
                reprojErr = np.mean(reprojErrs)
            else:
                reprojErr = np.mean(errs)
            if bestErr > reprojErr or bestErr < 0:
                bestErr = reprojErr
                bestX = X
        # reprojErrs = computeReprojErrMultiCam(X, controlPts, camProjMats)
        reprojErrs = computeReprojErrMultiCam(bestX, controlPts, camProjMats)

        reprojErrsS = sorted(reprojErrs)
        q1, q3 = np.percentile(reprojErrsS, [25, 75])
        iqr = q3 - q1
        upper_bound = q3 + (1.5 * iqr)
        outlierIds = np.where(reprojErrs > upper_bound)

        controlPtsFilterer = [controlPts[iP] for iP in range(len(controlPts)) if iP not in outlierIds]
        camProjMatsFilterer = [camProjMats[iP] for iP in range(len(camProjMats)) if iP not in outlierIds]
        X, errs = mulCamsDLT(controlPtsFilterer, camProjMatsFilterer)

        return X, errs

    else:
        print("Need at least 2 constraints")
        assert False