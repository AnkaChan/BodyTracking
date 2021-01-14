import cv2
import os
import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import matplotlib
import json
from tqdm.notebook import tqdm
from os.path import join
def load_bundle_adj_output(path, num_cams):
    cam_parameters = {}
    frames = {}
    g_chb_corners = None
    with open(path, 'r') as f:
        lines = f.readlines()
        num_frames = int(lines[0].split()[0])
        line_start = num_frames + 2

        cam_line = lines[line_start]
        cam_params = cam_line.split()
        for cam_idx in range(num_cams):
            param = cam_params[cam_idx * 22:(cam_idx + 1)*22 + 1]
            ax = np.array([param[0], param[1], param[2]]).astype(np.float32)
            t = np.array([param[3], param[4], param[5]]).astype(np.float32)
            fx = float(param[6])
            fy = float(param[7])
            cx = float(param[8])
            cy = float(param[9])
            k = np.array([param[10], param[11], param[12], param[13], param[14], param[15]]).astype(np.float32)
            p = np.array([param[16], param[17], param[18], param[19], param[20], param[21]]).astype(np.float32)
            out = {}
            out['ax'] = ax
            out['t'] = t
            out['fx'] = fx
            out['fy'] = fy
            out['cx'] = cx
            out['cy'] = cy
            out['k1'] = k[0]
            out['k2'] = k[1]
            out['k3'] = k[2]
            out['k4'] = k[3]
            out['k5'] = k[4]
            out['k6'] = k[5]

            out['p1'] = p[0]
            out['p2'] = p[1]
            out['p3'] = p[2]
            out['p4'] = p[3]
            out['p5'] = p[4]
            out['p6'] = p[5]
            cam_parameters[cam_idx] = out

        wp_start = line_start + 1
        line_indices = range(wp_start, wp_start + num_frames)

        chb_lines_exist = len(lines[wp_start + num_frames].split()) > 2
        if chb_lines_exist:
            print(" - 6-DoF checkerboard orientations exist")
        for line_idx in line_indices:
            wp_line = lines[line_idx]

            v = wp_line.split()
            frame_idx = int(v[0])
            img_name = v[1]
            detections = np.zeros((num_cams, 1))
            for cam_idx in range(num_cams):
                detections[cam_idx, :] = int(v[2 + cam_idx])
            wp = np.array(v[18:]).astype(np.float).reshape((88, 3))
            frame = {}
            frame['frame_idx'] = frame_idx
            frame['img_name'] = img_name
            frame['detections'] = detections
            frame['chb_corners'] = wp

            if chb_lines_exist:
                chb_line = lines[line_idx + num_frames]
                c = chb_line.split()
                chb_rvec = np.array([float(c[18]), float(c[19]), float(c[20])])
                chb_tvec = np.array([float(c[21]), float(c[22]), float(c[23])])
                frame['chb_rvec'] = chb_rvec
                frame['chb_tvec'] = chb_tvec
            frames[int(v[0])] = frame

        chb_points_start = wp_start + num_frames*2
        chb_points_exist = False
        chb_points_exist_1 = len(lines) > chb_points_start
        if chb_points_exist_1:
            chb_points_exist_2 = len(lines[chb_points_start].split()) > 2
            chb_points_exist = chb_points_exist_2

        if chb_points_exist:
            chb = lines[chb_points_start].split()
            g_chb_corners = np.array(chb).reshape((88, 3)).astype(np.float)
            print(' - Global checkerboard points exist : {}'.format(g_chb_corners.shape))
        final_cost = float(lines[-1].split()[-1])
        f.close()
    return cam_parameters, frames, g_chb_corners, final_cost

def reproject_world_points(param, world_points, configs):
    max_k = configs['max_k']
    max_p = configs['max_p']
    radial_model = configs['radial_model']
    # world_points = np.array((88, 3))
    # max_k, max_p = [2, 6]
    image_points = []
    k = [param['k1'], param['k2'], param['k3'], param['k4'], param['k5'], param['k6']]
    p = [param['p1'], param['p2'], param['p3'], param['p4'], param['p5'], param['p6']]

    fx = param['fx']
    fy = param['fy']
    cx = param['cx']
    cy = param['cy']

    if 'rvec' in param:
        ax = np.float32(param['rvec'])
        t = np.float32(param['tvec'])
    else:
        ax = np.float32(param['ax'])
        t = np.float32(param['t'])
    R, _ = cv2.Rodrigues(ax)

    E = np.array(
        [[R[0, 0], R[0, 1], R[0, 2], t[0]], [R[1, 0], R[1, 1], R[1, 2], t[1]], [R[2, 0], R[2, 1], R[2, 2], t[2]],
         [0, 0, 0, 1]])

    img_points = np.zeros((world_points.shape[0], 2))
    for p_idx in range(world_points.shape[0]):
        wp = world_points[p_idx, :]

        cp = E.dot(np.array([wp[0], wp[1], wp[2], 1]).astype('double'))
        xp = cp[0] / cp[2]
        yp = cp[1] / cp[2]

        if radial_model == 0:
            r2 = xp * xp + yp * yp
            r2_radials = 1.0
            radial_dist = 1.0
            for ki in range(0, max_k):
                r2_radials *= r2
                radial_dist += k[ki] * r2_radials
        else:
            r2 = xp ** 2 + yp ** 2
            r4 = r2 ** 2
            r6 = r2 ** 3
            radial_dist = (1.0 + k[0] * r2 + k[1] * r4 + k[2] * r6) / (1.0 + k[3] * r2 + k[4] * r4 + k[5] * r6)

        tan_post = 1.0
        r2_tangentials = 1.0
        for pi in range(2, max_p):
            r2_tangentials *= r2
            tan_post += p[pi] * r2_tangentials

        tan_x = (p[1] * (r2 + 2.0 * xp * xp) + 2.0 * p[0] * xp * yp) * tan_post
        tan_y = (p[0] * (r2 + 2.0 * yp * yp) + 2.0 * p[1] * xp * yp) * tan_post

        xp = xp * radial_dist + tan_x
        yp = yp * radial_dist + tan_y

        x_pred = fx * xp + cx
        y_pred = fy * yp + cy

        img_points[p_idx, 0] = x_pred
        img_points[p_idx, 1] = y_pred
    return img_points

def load_bundle_adj_input(path):
    # key = image_name, value = dictionary with key=cam_idx, val=2d points list
    output = {}
    with open(path, 'r') as f:
        j = json.load(f)
        configs = j['configs']
        num_frames = configs['num_frames']
        frames = j['frames']
        if len(frames) != num_frames:
            print('[ERROR] num frames sanity check fail: {} != {}'.format(len(frames), num_frames))

        for frame in frames:
            out = {}
            img_name = frame['img_name']
            img_pts = frame['img_pts']
            for cam_idx, v in img_pts.items():
                wp = np.array(v).reshape((88, 2))
                out[int(cam_idx)] = wp
            output[img_name] = out

        f.close()
    return output

def compute_reprojection_errors(preds_in, opencvs_in, configs, mins=None, maxs=None):
    img_names_2_skip = {}
    if mins is not None:
        min_u = mins[0]
        min_v = mins[1]
        max_u = maxs[0]
        max_v = maxs[1]
        for img_name, v in preds_in.items():
            img_names_2_skip[img_name] = False
            opencvs = opencvs_in[img_name]
            for cam_idx in v.keys():
                mea = opencvs[cam_idx]

                for p_idx in range(mea.shape[0]):
                    mask = (min_u[cam_idx] < mea[p_idx, 0]) and (mea[p_idx, 0] < max_u[cam_idx]) and (
                                min_v[cam_idx] < mea[p_idx, 1]) and (mea[p_idx, 1] < max_v[cam_idx])
                    if not mask:
                        img_names_2_skip[img_name] = True
                        break
                if img_names_2_skip[img_name]:
                    print('skip: {}'.format(img_name))
                    break

    all_errs = []
    err_data = {}
    sanity_err = 0
    for img_name, v in preds_in.items():
        if img_name in img_names_2_skip:
            if img_names_2_skip[img_name]:
                print(' - skip: {}'.format(img_name))
                continue

        opencvs = opencvs_in[img_name]
        curr_errs = {}
        for cam_idx in v.keys():
            pred = v[cam_idx]
            mea = opencvs[cam_idx]

            dxdy = pred - mea
            dxdx_dydy_sum = np.sum(dxdy ** 2, axis=1)
            if configs['loss_type'] == 'huber':
                delta = configs['loss_huber_delta']
                for ei in range(dxdx_dydy_sum.shape[0]):
                    if dxdx_dydy_sum[ei] > delta:
                        sanity_err += 2 * delta * np.sqrt(dxdx_dydy_sum[ei]) - 1 * delta ** 2
                    else:
                        sanity_err += dxdx_dydy_sum[ei]
            else:
                sanity_err += np.sum(dxdx_dydy_sum)

            all_errs.extend(np.sqrt(dxdx_dydy_sum).tolist())
            curr_errs[cam_idx] = np.sqrt(dxdx_dydy_sum)
        err_data[img_name] = curr_errs

    sanity_err *= 0.5
    return all_errs, err_data, sanity_err

if __name__ == '__main__':
    dnames = ['Lada', "Katey", "Marianne"]

    dname = dnames[0]

    if dname == "Lada":
        # change here
        dataFolder = r'F:\WorkingCopy2\2021_01_07_JP_TEMP_HistogramForAnka\CalibrationData\2019_12_13_Lada_Capture_k1k2k3p1p2\BundleAdjustment'
        bundle_path = join(dataFolder, r'output\bundle_adjustment_6dof\bundleadjustment_output.txt')
        measurement_path =  join(dataFolder, r'input\image_points.json')

        # don't change
        configs = {
            'max_k': 3,
            'max_p': 2,
            'radial_model': 0,
            'loss_type': 'none',
            'loss_huber_delta': 0.0
        }
    elif dname == "Katey":
        # change here
        bundle_path = r'D:\CalibrationData\CameraCalibration\2020_01_01_KateyCapture\BundleAdjustment\output\bundle_adjustment_6dof\bundleadjustment_output.txt'
        measurement_path = r'D:\CalibrationData\CameraCalibration\2020_01_01_KateyCapture\BundleAdjustment\input\image_points.json'

        # don't change
        configs = {
            'max_k': 6,
            'max_p': 2,
            'radial_model': 1,
            'loss_type': 'none',
            'loss_huber_delta': 0.0
        }
    elif dname == "Marianne":
        # change here
        bundle_path = r'D:\CalibrationData\CameraCalibration\2019_12_24_Marianne_Capture\BundleAdjustment\output\bundle_adjustment_6dof\bundleadjustment_output.txt'
        measurement_path = r'D:\CalibrationData\CameraCalibration\2019_12_24_Marianne_Capture\BundleAdjustment\input\image_points.json'

        # don't change
        configs = {
            'max_k': 6,
            'max_p': 2,
            'radial_model': 1,
            'loss_type': 'none',
            'loss_huber_delta': 0.0
        }
    print(dname)
    print(bundle_path)
    print(measurement_path)

    outputFolder = join('output', 'S24_CalibrationErrs')
    os.makedirs(outputFolder, exist_ok=True)

    cam_params, frames, g_chb_corners, final_cost = load_bundle_adj_output(bundle_path, 16)
    print(frames[0].keys())

    print("=================================================================================")
    for cam_idx, p in cam_params.items():
        k = [p['k1'], p['k2'], p['k3'], p['k4'], p['k5'], p['k6']]
        p = [p['p1'], p['p2'], p['p3'], p['p4'], p['p5'], p['p6']]
        print('cam[{}]\n\tk={}\n\tp={}'.format(cam_idx, k, p))
        break
    print("=================================================================================")
    print('done')

    img_pts_pred = {}
    pbar = tqdm(total=len(frames.keys()))
    for i, frame in frames.items():
        pbar.update(1)
        wps = np.float32(frame["chb_corners"])
        img_name = frame['img_name']
        img_pts = {} # cam_idx, pts
        for cam_idx, d in enumerate(frame['detections']):
            detected = bool(d)
            if detected:
                param = cam_params[cam_idx]
                pts = reproject_world_points(param, wps, configs)
                img_pts[int(cam_idx)] = pts
        img_pts_pred[img_name] = img_pts

    img_pts_opencv = load_bundle_adj_input(measurement_path)
    print('done: {} frames'.format(len(img_pts_opencv.keys())))

    reproj_errs, err_data, sanity_err = compute_reprojection_errors(img_pts_pred, img_pts_opencv, configs)

    mean = sum(reproj_errs) / len(reproj_errs)
    max_err = max(reproj_errs)
    std = np.std(reproj_errs)

    num_frames = len(err_data)
    print('{} frames'.format(num_frames))
    print('- mean = {:<8.4f} [pixel/point]'.format(mean))
    print('- std  = {:<8.4f}'.format(std))
    print('- max  = {:<8.4f} [pixel/point]'.format(max_err))

    # sanity check
    # sanity_err = 0.5*np.sum(np.array(reproj_errs)**2)

    reg = 0
    if reg:
        for cam_idx, p in cam_params.items():
            # reg
            sanity_err += 0.5 * (
                        p['k1'] ** 2 + p['k2'] ** 2 + p['k3'] ** 2 + p['k4'] ** 2 + p['k5'] ** 2 + p['k6'] ** 2 + p[
                    'p1'] ** 2 + p['p2'] ** 2)

    print()
    print('Sanity check: mine({:.4f}) vs ceres({:.4f}) | error={:.2f}%'.format(sanity_err, final_cost, abs(
        final_cost - sanity_err) / final_cost * 100.0))
    print('done')

    import numpy as np
    import matplotlib.pyplot as plt
    from cycler import cycler
    from matplotlib.gridspec import GridSpec


    mean = sum(reproj_errs) / len(reproj_errs)
    max_err = max(reproj_errs)
    std = np.std(reproj_errs)

    num_frames = len(err_data)
    print('{} frames'.format(num_frames))
    print('- mean = {:<8.4f} [pixel/point]'.format(mean))
    print('- std  = {:<8.4f}'.format(std))
    print('- max  = {:<8.4f} [pixel/point]'.format(max_err))

    mpl.style.use('default')

    # mpl.rcParams['axes.prop_cycle'] = cycler(color='rbgcmyk')
    num_bins = 1000
    plt.rcParams["figure.figsize"] = (6, 2.5)  # (w, h)
    fig = plt.figure(constrained_layout=True, tight_layout=True)
    gs = GridSpec(1, 1, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])

    # linear
    save_path = os.path.join(outputFolder, dname + ".png")

    ax1.hist(reproj_errs, bins=num_bins)
    ax1.set_yscale('log')
    ax1.set_xlim(left=0)
    ax1.set_xlabel('(c) Reprojection error of camera calibration [pixels]')
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
