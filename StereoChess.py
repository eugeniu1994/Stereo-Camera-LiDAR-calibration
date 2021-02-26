
'''  CONFIDENTIAL

     Copyright (c) 2021 Eugeniu Vezeteu,
     Department of Remote Sensing and Photogrammetry,
     Finnish Geospatial Research Institute (FGI), National Land Survey of Finland (NLS)


     PERMISSION IS HEREBY LIMITED TO FGI'S INTERNAL USE ONLY. THE CODE
     MAY BE RE-LICENSED, SHARED, OR TAKEN INTO OTHER USE ONLY WITH
     A WRITTEN CONSENT FROM THE HEAD OF THE DEPARTMENT.


     The software is provided "as is", without warranty of any kind, express or
     implied, including but not limited to the warranties of merchantability,
     fitness for a particular purpose and noninfringement. In no event shall the
     authors or copyright holders be liable for any claim, damages or other
     liability, whether in an action of contract, tort or otherwise, arising from,
     out of or in connection with the software or the use or other dealings in the
     software.
'''

import numpy as np
import cv2
import glob
import pickle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
from utils import *

np.set_printoptions(suppress=True)

class StereoChess_Calibrator(object):
    def __init__(self, filepath, name = ''):
        self.term_criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 1000, 0.0001)
        self.square = 0.1 #m
        self.objp = np.zeros((10 * 7, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:10, 0:7].T.reshape(-1, 2) * self.square

        self.objpoints = []  # 3d point in real world space
        self.imgpoints_l = []  # 2d points in image plane.
        self.imgpoints_r = []  # 2d points in image plane.

        self.cal_path = filepath
        self.image_width = 1936
        self.image_height = 1216
        self.image_size = None
        self.fx = 0.25
        self.fy = 0.3

        self.position = (20, 30)
        self.see = True
        self.flipVertically = True
        self.name = name

        self.total = 0

    def readMonoData(self):
        left_outside = load_obj(name='chess_{}_left'.format(self.name))
        self.K_left = left_outside['K']
        self.D_left = left_outside['D']

        right_outside = load_obj(name='chess_{}_right'.format(self.name))
        self.K_right = right_outside['K']
        self.D_right = right_outside['D']

        print("Left K:\n", self.K_left)
        print("Left distortion : ", self.D_left.ravel())
        print("Right K:\n", self.K_right)
        print("Right distortion : ", self.D_right.ravel())

        print('Loaded mono internal calibration data')

    def readStereoData(self):
        camera_model = load_obj('{}_chess_camera_model'.format(self.name ))
        camera_model_rectify = load_obj('{}_chess_camera_model_rectify'.format(self.name))
        self.K_left = camera_model['K_left']
        self.K_right = camera_model['K_right']
        self.D_left = camera_model['D_left']
        self.D_right = camera_model['D_right']

        self.R = camera_model['R']
        self.T = camera_model['T']
        self.Q = camera_model_rectify['Q']

        self.roi_left, self.roi_right = camera_model_rectify['roi_left'],camera_model_rectify['roi_right']
        self.leftMapX, self.leftMapY = camera_model_rectify['leftMapX'], camera_model_rectify['leftMapY']
        self.rightMapX, self.rightMapY = camera_model_rectify['rightMapX'], camera_model_rectify['rightMapY']

        print('Rotation R')
        print(self.R)
        print('Translation T')
        print(self.T)
        print('Stereo data has been loaded...')

    def read_images(self, test=False):
        images_right = glob.glob(self.cal_path + '/Right/*.png')
        images_left = glob.glob(self.cal_path + '/Left/*.png')

        images_left.sort()
        images_right.sort()
        self.LeftImg, self.RightImg = [], []
        wait = 0
        for i, fname in enumerate(images_right):
            img_l = cv2.imread(images_left[i])
            img_r = cv2.imread(images_right[i])

            if self.flipVertically:
                img_l = cv2.flip(img_l, -1)
                img_r = cv2.flip(img_r, -1)

            self.LeftImg.append(img_l)
            self.RightImg.append(img_r)

            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

            if test:
                ret_l = True
                ret_r = True
            else:
                ret_l, corners_l = cv2.findChessboardCorners(gray_l, (10, 7), None)
                ret_r, corners_r = cv2.findChessboardCorners(gray_r, (10, 7), None)

            if ret_l and ret_r:
                if test == False:
                    self.objpoints.append(self.objp)
                    rt = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), self.term_criteria)
                    self.imgpoints_l.append(corners_l)
                    ret_l = cv2.drawChessboardCorners(img_l, (10, 7), corners_l, ret_l)

                    rt = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), self.term_criteria)
                    self.imgpoints_r.append(corners_r)
                    ret_r = cv2.drawChessboardCorners(img_r, (10, 7), corners_r, ret_r)

                    self.total+=len(corners_r)

                if self.see:
                    cam_right_resized = cv2.resize(img_r, None, fx=self.fx, fy=self.fy)
                    cam_left_resized = cv2.resize(img_l, None, fx=self.fx, fy=self.fy)

                    cv2.putText(cam_left_resized, "Left", self.position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255, 0))
                    cv2.putText(cam_right_resized, "Right", self.position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255, 0))

                    im_h = cv2.hconcat([cam_left_resized, cam_right_resized])
                    cv2.imshow('Stereo camera', im_h)
                    k = cv2.waitKey(wait)
                    if k % 256 == 32:  # pressed space
                        self.see = False
                        cv2.destroyAllWindows()
                    elif k & 0xFF == ord('q'):
                        wait = 50

            self.img_shape = gray_l.shape[::-1]
        cv2.destroyAllWindows()

    def stereoCalibrate(self, flags=None, save=False):
        if flags is None:
            flags = 0
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        flags |= cv2.CALIB_FIX_ASPECT_RATIO

        self.rms_stereo, self.K_left, self.D_left, self.K_right, self.D_right, self.R, self.T, self.E, self.F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_l,
            self.imgpoints_r, self.K_left, self.D_left, self.K_right, self.D_right, self.img_shape,
            criteria=self.term_criteria, flags=flags)

        print('Stereo calibraion done')
        print('rms_stereo:{}'.format(self.rms_stereo))
        print('Rotation R')
        print(self.R)
        print('Translation T')
        print(self.T)

        R1, R2, P1, P2, self.Q, self.roi_left, self.roi_right = cv2.stereoRectify(self.K_left, self.D_left, self.K_right,
                                                                        self.D_right, self.img_shape, self.R, self.T,
                                                                        flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1)

        self.leftMapX, self.leftMapY = cv2.initUndistortRectifyMap(
            self.K_left, self.D_left, R1,
            P1, self.img_shape, cv2.CV_32FC1)

        self.rightMapX, self.rightMapY = cv2.initUndistortRectifyMap(
            self.K_right, self.D_right, R2,
            P2, self.img_shape, cv2.CV_32FC1)

        if save:
            camera_model_rectify = dict([('R1', R1), ('R2', R2), ('P1', P1),
                                         ('P2', P2), ('Q', self.Q),
                                         ('roi_left', self.roi_left), ('roi_right', self.roi_right),
                                         ('leftMapX', self.leftMapX), ('leftMapY', self.leftMapY),
                                         ('rightMapX', self.rightMapX), ('rightMapY', self.rightMapY)])

            camera_model = dict([('K_left', self.K_left), ('K_right', self.K_right), ('D_left', self.D_left),
                                 ('D_right', self.D_right), ('R', self.R), ('T', self.T),
                                 ('E', self.E), ('F', self.F)])

            save_obj(camera_model,self.name+'_chess_camera_model')
            save_obj(camera_model_rectify, self.name+'_chess_camera_model_rectify')

    def depth_map_SGBM(self, imgL, imgR):
        window_size = 5
        if self.left_matcher is None:
            self.left_matcher = cv2.StereoSGBM_create(minDisparity=5,
                                                        numDisparities=256,
                                                        blockSize=window_size,
                                                    )

            self.stereo = cv2.StereoBM_create(numDisparities=256, blockSize=5)
            self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
            # FILTER Parameters
            lmbda = 80000
            sigma = 1.3

            self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self.left_matcher)
            self.wls_filter.setLambda(lmbda)
            self.wls_filter.setSigmaColor(sigma)

        displ = self.left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
        dispr = self.right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
        displ = np.int16(displ)
        dispr = np.int16(dispr)
        SGBM_disp = self.wls_filter.filter(displ, imgL, imgR, dispr)

        SGBM_disp = cv2.normalize(src=SGBM_disp, dst=SGBM_disp, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
        SGBM_disp = np.uint8(SGBM_disp)

        disparity = self.stereo.compute(imgL, imgR)
        disparity = cv2.normalize(src=disparity, dst=disparity, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
        BM_disp = np.uint8(disparity)

        return SGBM_disp, BM_disp

    def depth_and_color(self, img, left_rectified):
        points = cv2.reprojectImageTo3D(img, self.Q)

        reflect_matrix = np.identity(3)  # reflect on x axis
        reflect_matrix[0] *= -1
        points = np.matmul(points, reflect_matrix)
        colors = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2RGB)

        # filter by min disparity
        mask = img > img.min()
        out_points = points[mask]
        out_colors = colors[mask]

        idx = np.fabs(out_points[:, 0]) < 250  # 10.5 # filter by dimension
        out_points = out_points[idx]
        out_colors = out_colors.reshape(-1, 3)
        out_colors = out_colors[idx]
        write_ply('BM_2.ply', out_points, out_colors)
        print('%s saved' % 'out.ply')

        reflected_pts = np.matmul(out_points, reflect_matrix)
        projected_img, _ = cv2.projectPoints(reflected_pts, np.identity(3), np.array([0., 0., 0.]), self.K_left,
                                             self.D_left)
        projected_img = projected_img.reshape(-1, 2)

        blank_img = np.zeros(left_rectified.shape, 'uint8')
        img_colors = left_rectified[mask][idx].reshape(-1, 3)

        for i, pt in enumerate(projected_img):
            pt_x = int(pt[0])
            pt_y = int(pt[1])
            if pt_x > 0 and pt_y > 0:
                # use the BGR format to match the original image type
                col = (int(img_colors[i, 2]), int(img_colors[i, 1]), int(img_colors[i, 0]))
                cv2.circle(blank_img, (pt_x, pt_y), 1, col)

        return blank_img

    def depth(self):
        self.left_matcher = None
        wait = 0
        for i, fname in enumerate(self.LeftImg):
            leftFrame = self.LeftImg[i]
            rightFrame = self.RightImg[i]

            left_rectified = cv2.remap(leftFrame, self.leftMapX, self.leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
            right_rectified = cv2.remap(rightFrame, self.rightMapX, self.rightMapY, cv2.INTER_LINEAR,
                                        cv2.BORDER_CONSTANT)

            #out = right_rectified.copy()
            #out[:, :, 0] = left_rectified[:, :, 0]
            #out[:, :, 1] = left_rectified[:, :, 1]
            #out[:, :, 2] = right_rectified[:, :, 2]

            gray_left = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
            #disparity_image2 = cv2.applyColorMap(disparity_image2, cv2.COLORMAP_JET)

            SGBM_disp, BM_disp = self.depth_map_SGBM(gray_left, gray_right, )  # Get the disparity map
            img_top = cv2.hconcat(
                [cv2.resize(left_rectified, None, fx=self.fx, fy=self.fy), cv2.resize(right_rectified, None, fx=self.fx, fy=self.fy)])

            img_bot = np.hstack((cv2.resize(SGBM_disp, None, fx=self.fx, fy=self.fy), cv2.resize(BM_disp, None, fx=self.fx, fy=self.fy)))
            #pointCloudColor = self.depth_and_color(img=BM_disp.copy(),left_rectified=left_rectified)
            pointCloudColor = self.depth_and_color(img=SGBM_disp.copy(),left_rectified=left_rectified)

            cv2.imshow('img_bot ', img_bot)
            cv2.imshow('Result', img_top)
            cv2.imshow('pointCloudColor', cv2.resize(pointCloudColor, None, fx=.4, fy=.4))

            k = cv2.waitKey(wait)
            if k & 0xFF == ord('q'):
                wait = 1
                #break
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def calibrationReport(self):
        Distorsion_models = {'ST': ['Standard', 0, 'Standard'],
                                 'RAT': ['Rational', cv2.CALIB_RATIONAL_MODEL, 'CALIB_RATIONAL_MODEL'],
                                 'THP': ['Thin Prism', cv2.CALIB_THIN_PRISM_MODEL, 'CALIB_THIN_PRISM_MODEL'],
                                 'TIL': ['Tilded', cv2.CALIB_TILTED_MODEL, 'CALIB_TILTED_MODEL'],  # }
                                 'RAT+THP': ['Rational+Thin Prism',
                                             cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_THIN_PRISM_MODEL,
                                             'CALIB_RATIONAL_MODEL + CALIB_THIN_PRISM_MODEL'],
                                 'THP+TIL': ['Thin Prism+Tilded', cv2.CALIB_THIN_PRISM_MODEL + cv2.CALIB_TILTED_MODEL,
                                             'CALIB_THIN_PRISM_MODEL + CALIB_TILTED_MODEL'],
                                 'RAT+TIL': ['Rational+Tilded', cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_TILTED_MODEL,
                                             'CALIB_RATIONAL_MODEL + CALIB_TILTED_MODEL'],
                                 'CMP': ['Complete',
                                         cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_THIN_PRISM_MODEL + cv2.CALIB_TILTED_MODEL,
                                         'Complete']}

        def rot2eul(R):
            beta = -np.arcsin(R[2, 0])
            alpha = np.arctan2(R[2, 1] / np.cos(beta), R[2, 2] / np.cos(beta))
            gamma = np.arctan2(R[1, 0] / np.cos(beta), R[0, 0] / np.cos(beta))
            alpha, beta, gamma = math.degrees(alpha), math.degrees(beta), math.degrees(gamma)
            return np.array((alpha, beta, gamma))

        calibration_results = pd.DataFrame(
            {"params": [ 'Tx','Ty','Tz','Rx','Ry','Rz', 'Error']})
        rms_all = ['Error']

        min_error, flag = 10000000, 0
        for key in Distorsion_models:
            print()
            print(key, '->', Distorsion_models[key][0], ' , ', Distorsion_models[key][1], ' , ',
                  Distorsion_models[key][2])
            flags = Distorsion_models[key][1]

            self.stereoCalibrate(flags=flags, save=False)
            s = np.append(self.T, rot2eul(self.R))
            calibration_results[str(key)] = pd.Series(s)
            calibration_results.fillna('---', inplace=True)
            rms_all.append(self.rms_stereo)
            calibration_results[str(key)] = calibration_results[str(key)].map(
                lambda x: round(x, 4) if isinstance(x, (int, float)) else x).astype(str)

            if self.rms_stereo < min_error:
                min_error=self.rms_stereo
                flag = flags

        calibration_results.iloc[-1, :] = rms_all
        calibration_results.iloc[-1, :] = calibration_results.iloc[-1, :].map(
            lambda x: round(x, 4) if isinstance(x, (int, float)) else x)

        save_csv(obj=calibration_results, name=self.name+'_StereoCalibration')
        print('Best distortion model for stereo calib is {}'.format(flag))
        self.stereoCalibrate(flags=flag,save=True)

        return flag

