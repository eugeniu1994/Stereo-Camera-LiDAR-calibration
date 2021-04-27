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

import glob
import numpy as np
import cv2
import cv2.aruco as aruco
import pandas as pd

from CameraCalibration.scripts.MonoCharuco import MonoCharuco_Calibrator
from CameraCalibration.scripts.MonoChess import MonoChess_Calibrator
from CameraCalibration.scripts.StereoCharuco import StereoCharuco_Calibrator
from CameraCalibration.scripts.StereoChess import StereoChess_Calibrator

from utils import *

class CombinedCalibration(object):
    def __init__(self, name = '', flipVertically = True):
        self.name = name
        self.term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 500, 0.0001)
        self.flipVertically = flipVertically
        self.fx = 0.3
        self.fy = 0.35

        self.image_width = 1936
        self.image_height = 1216

    def calibrateMono_(self, imgChess, imgCharuco,see = False, save = True, cam = 'left'):
        Chess_calibrator = MonoChess_Calibrator(pattern_type="chessboard", pattern_rows=10,
                                          pattern_columns=7, distance_in_world_units=10,  # square is 10 cm
                                          figsize=(14, 10))
        Chess_calibrator.see = see
        Chess_calibrator.read_images(images_path_list=imgChess)
        Chess_objectPoints = np.array(Chess_calibrator.calibration_df.obj_points)
        Chess_imagePoints = np.array(Chess_calibrator.calibration_df.img_points)

        Charuco_calibrator = MonoCharuco_Calibrator()
        Charuco_calibrator.see = see
        Charuco_calibrator.createCalibrationBoard()
        Charuco_calibrator.read_images(images=imgCharuco, threads=1)
        Charuco_objectPoints = np.array(Charuco_calibrator.calibration_df.obj_points)
        Charuco_imagePoints = np.array(Charuco_calibrator.calibration_df.img_points)
        self.image_size = Charuco_calibrator.image_size
        print('Chess_objectPoints:{}, Chess_imagePoints:{}'.format(np.shape(Chess_objectPoints), np.shape(Chess_imagePoints)))
        print('Charuco_objectPoints:{}, Chess_imagePoints:{}'.format(np.shape(Charuco_objectPoints), np.shape(Charuco_imagePoints)))
        print('self.image_size ',self.image_size)
        _objectPoints = np.append(Chess_objectPoints,Charuco_objectPoints)
        _imagePoints = np.append(Chess_imagePoints, Charuco_imagePoints)

        print('_objectPoints:{}, _imagePoints:{}'.format(np.shape(_objectPoints), np.shape(_imagePoints)))
        self.rms, self.K, self.D, self.rvecs, self.tvecs = cv2.calibrateCamera(
            objectPoints=_objectPoints,
            imagePoints=_imagePoints,
            imageSize=self.image_size,
            cameraMatrix=None, distCoeffs=None,
            flags=0, criteria=self.term_criteria)

        print("\nRMS:", self.rms)
        print("camera matrix:\n", self.K)
        print("distortion coefficients: ", self.D.ravel())
        if save:
            result_dictionary = {
                "rms": self.rms,
                "K": self.K,
                "D": self.D,
            }
            save_obj(obj=result_dictionary, name='combined_{}_{}'.format(self.name, cam))

    def calibrateStereo(self,imgChess, imgCharuco,see = False, save = True, flags=None,):
        total = 0
        Chess_calibrator = StereoChess_Calibrator(imgChess)
        Chess_calibrator.see = see
        Chess_calibrator.read_images(test=False)
        total+=Chess_calibrator.total

        Charuco_calibrator = StereoCharuco_Calibrator(imgCharuco)
        Charuco_calibrator.see = see
        Charuco_calibrator.createCalibrationBoard()
        Charuco_calibrator.read_images(test=False)
        total+=Charuco_calibrator.total
        self.image_size = Charuco_calibrator.img_shape
        print('self.image_size ',self.image_size)
        if flags is None:
            flags = 0
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        flags |= cv2.CALIB_FIX_ASPECT_RATIO

        Chess_objectPoints = np.array(Chess_calibrator.objpoints)
        Chess_imagePointsL = np.array(Chess_calibrator.imgpoints_l)
        Chess_imagePointsR = np.array(Chess_calibrator.imgpoints_r)

        Charuco_objectPoints = np.array(Charuco_calibrator.objpoints)
        Charuco_imagePointsL = np.array(Charuco_calibrator.imgpoints_l)
        Charuco_imagePointsR = np.array(Charuco_calibrator.imgpoints_r)

        print('Chess_objectPoints:{}, Chess_imagePointsL:{}, Chess_imagePointsR:{}'.format(np.shape(Chess_objectPoints),
                                                                   np.shape(Chess_imagePointsL), np.shape(Chess_imagePointsR)))

        print('Charuco_objectPoints:{}, Charuco_imagePointsL:{}, Charuco_imagePointsR:{}'.format(np.shape(Charuco_objectPoints),
                                                                                           np.shape(Charuco_imagePointsL),
                                                                                           np.shape(
                                                                                               Charuco_imagePointsR)))
        print('Total of {} points'.format(total))
        _objectPoints = []# np.append(Chess_objectPoints, Charuco_objectPoints)
        _imagePointsL = []# np.append(Chess_imagePointsL, Charuco_imagePointsL)
        _imagePointsR = []# np.append(Chess_imagePointsR, Charuco_imagePointsR)
        for i in range(len(Chess_objectPoints)):
            _objectPoints.append(Chess_objectPoints[i])
            _imagePointsL.append(Chess_imagePointsL[i])
            _imagePointsR.append(Chess_imagePointsR[i])
        for i in range(len(Charuco_objectPoints)):
            _objectPoints.append(Charuco_objectPoints[i])
            _imagePointsL.append(Charuco_imagePointsL[i])
            _imagePointsR.append(Charuco_imagePointsR[i])
        print('_objectPoints:{}, _imagePointsL:{}, _imagePointsR:{}'.format(np.shape(_objectPoints), np.shape(_imagePointsL), np.shape(_imagePointsR)))

        self.rms_stereo, self.K_left, self.D_left, self.K_right, self.D_right, self.R, self.T, self.E, self.F = cv2.stereoCalibrate(
            _objectPoints, _imagePointsL,
            _imagePointsR, self.K_left, self.D_left, self.K_right, self.D_right, self.image_size,
            criteria=self.term_criteria, flags=flags)

        print('Stereo calibraion done')
        print('rms_stereo:{}'.format(self.rms_stereo))
        print('Rotation R')
        print(self.R)
        print('Translation T')
        print(self.T)

        R1, R2, P1, P2, self.Q, self.roi_left, self.roi_right = cv2.stereoRectify(self.K_left, self.D_left,
                                                                                  self.K_right,
                                                                                  self.D_right, self.image_size, self.R,
                                                                                  self.T,flags=cv2.CALIB_ZERO_DISPARITY,
                                                                                  alpha=-1)

        self.leftMapX, self.leftMapY = cv2.initUndistortRectifyMap(
            self.K_left, self.D_left, R1,
            P1, self.image_size, cv2.CV_32FC1)

        self.rightMapX, self.rightMapY = cv2.initUndistortRectifyMap(
            self.K_right, self.D_right, R2,
            P2, self.image_size, cv2.CV_32FC1)

        if save:
            camera_model_rectify = dict([('R1', R1), ('R2', R2), ('P1', P1),
                                         ('P2', P2), ('Q', self.Q),
                                         ('roi_left', self.roi_left), ('roi_right', self.roi_right),
                                         ('leftMapX', self.leftMapX), ('leftMapY', self.leftMapY),
                                         ('rightMapX', self.rightMapX), ('rightMapY', self.rightMapY)])

            camera_model = dict([('K_left', self.K_left), ('K_right', self.K_right), ('D_left', self.D_left),
                                 ('D_right', self.D_right), ('R', self.R), ('T', self.T),
                                 ('E', self.E), ('F', self.F)])

            save_obj(camera_model, self.name + '_combined_camera_model')
            save_obj(camera_model_rectify, self.name + '_combined_camera_model_rectify')

    def calibrationReport(self,imgChess, imgCharuco):
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
            {"params": ['Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz', 'Error']})
        rms_all = ['Error']

        min_error, flag = 10000000, 0
        for key in Distorsion_models:
            print()
            print(key, '->', Distorsion_models[key][0], ' , ', Distorsion_models[key][1], ' , ',
                  Distorsion_models[key][2])
            flags = Distorsion_models[key][1]

            self.calibrateStereo(imgChess, imgCharuco, flags=flags, save=False)
            s = np.append(self.T, rot2eul(self.R))
            calibration_results[str(key)] = pd.Series(s)
            calibration_results.fillna('---', inplace=True)
            rms_all.append(self.rms_stereo)
            calibration_results[str(key)] = calibration_results[str(key)].map(
                lambda x: round(x, 4) if isinstance(x, (int, float)) else x).astype(str)

            if self.rms_stereo < min_error:
                min_error = self.rms_stereo
                flag = flags

        calibration_results.iloc[-1, :] = rms_all
        calibration_results.iloc[-1, :] = calibration_results.iloc[-1, :].map(
            lambda x: round(x, 4) if isinstance(x, (int, float)) else x)

        save_csv(obj=calibration_results, name=self.name + '_combined_StereoCalibration')
        print('Best distortion model for stereo calib is {}'.format(flag))
        #self.stereoCalibrate(flags=flag, save=True)

        return flag

    def readMonoData(self):
        left_ = load_obj(name='combined_{}_left'.format(self.name))
        self.K_left = left_['K']
        self.D_left = left_['D']

        right_ = load_obj(name='combined_{}_right'.format(self.name))
        self.K_right = right_['K']
        self.D_right = right_['D']

        print("Left K:\n", self.K_left)
        print("Left distortion : ", self.D_left.ravel())
        print("Right K:\n", self.K_right)
        print("Right distortion : ", self.D_right.ravel())

        print('Loaded mono internal calibration data')

    def readStereoData(self):
        camera_model = load_obj('{}_combined_camera_model'.format(self.name ))
        camera_model_rectify = load_obj('{}_combined_camera_model_rectify'.format(self.name))
        self.K_left = camera_model['K_left']
        self.K_right = camera_model['K_right']
        self.D_left = camera_model['D_left']
        self.D_right = camera_model['D_right']

        self.R = camera_model['R']
        self.T = camera_model['T']
        self.Q = camera_model_rectify['Q']

        self.P1 = camera_model_rectify['P1']
        self.P2 = camera_model_rectify['P2']

        self.roi_left, self.roi_right = camera_model_rectify['roi_left'],camera_model_rectify['roi_right']
        self.leftMapX, self.leftMapY = camera_model_rectify['leftMapX'], camera_model_rectify['leftMapY']
        self.rightMapX, self.rightMapY = camera_model_rectify['rightMapX'], camera_model_rectify['rightMapY']

        print('Rotation R')
        print(self.R)
        print('Translation T')
        print(self.T)
        print('Stereo data has been loaded...')

        self.image_size = (self.image_width, self.image_height)
        R1, R2, P1, P2, self.Q, self.roi_left, self.roi_right = cv2.stereoRectify(self.K_left, self.D_left,
                                                                                  self.K_right,
                                                                                  self.D_right, self.image_size, self.R,
                                                                                  self.T,
                                                                                  flags=cv2.CALIB_ZERO_DISPARITY,
                                                                                  alpha=0)
                                                                                  #flags=cv2.CALIB_ZERO_DISPARITY,
                                                                                  #alpha=-1)

        '''self.leftMapX, self.leftMapY = cv2.initUndistortRectifyMap(
            self.K_left, self.D_left, R1,
            P1, self.image_size, cv2.CV_32FC1)

        self.rightMapX, self.rightMapY = cv2.initUndistortRectifyMap(
            self.K_right, self.D_right, R2,
            P2, self.image_size, cv2.CV_32FC1)'''

    def read_images(self, path):
        images_right = glob.glob(path + '/Right/*.png')
        images_left = glob.glob(path + '/Left/*.png')

        images_left.sort()
        images_right.sort()
        self.LeftImg, self.RightImg = [], []

        for i, fname in enumerate(images_right):
            img_l = cv2.imread(images_left[i])
            img_r = cv2.imread(images_right[i])

            if self.flipVertically:
                img_l = cv2.flip(img_l, -1)
                img_r = cv2.flip(img_r, -1)

            self.LeftImg.append(img_l)
            self.RightImg.append(img_r)

    def depth_map_SGBM(self, imgL, imgR, numDisparities = 128, window_size = 5):
        if self.left_matcher is None:
            self.left_matcher = cv2.StereoSGBM_create(minDisparity=5,
                                                        numDisparities=numDisparities,
                                                        blockSize=window_size,
                                                    )

            self.stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=window_size)
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
        #SGBM_disp = SGBM_disp.astype(np.float32) / 16.0
        #SGBM_disp = cv2.normalize(SGBM_disp, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        #BM_disp = self.stereo.compute(imgL, imgR)
        #BM_disp = cv2.normalize(src=BM_disp, dst=BM_disp, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
        #BM_disp = np.uint8(BM_disp)
        #BM_disp = BM_disp.astype(np.float32) / 16.0
        window_size = 9

        min_disp = 16
        num_disp = 112 - min_disp

        stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                      numDisparities=num_disp,
                                      blockSize=window_size,
                                      )
        #BM_disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
        BM_disp = stereo.compute(imgR, imgL).astype(np.float32) / 16.0

        return SGBM_disp, BM_disp

    def depth_and_color(self, img, img2=None, left_rectified=None, name = 'out.ply', gray_left=None,gray_right=None):
        disparity_map = img2
        #plt.imshow(disparity_map, cmap='gray', vmin=225, vmax=255)
        plt.imshow(disparity_map, cmap='gray')

        plt.show()

        focalLength, centerX, centerY = self.K_left[0,0], self.K_left[0,2], self.K_left[1,2]
        fx,fy = self.K_left[0,0],self.K_left[1,1]
        Baseline = self.T[0]
        h,w = np.shape(disparity_map)
        print('min {},  max:{}'.format(np.min(disparity_map), np.max(disparity_map)))

        print('np.shape(depth) {}'.format(np.shape(disparity_map)))
        scalingFactor = 150
        points, colors = [],[]

        D = np.zeros_like(disparity_map)
        for v in range(250,w):
            for u in range(h):
                #Z = (Baseline * focalLength) / (depth[u, v] * p)
                if disparity_map[u, v] != 0:
                    color = left_rectified[u, v]

                    Z = disparity_map[u, v] / scalingFactor
                    X = (u - centerX) * Z / fx
                    Y = (v - centerY) * Z / fy
                    Z = (fy*Baseline)/disparity_map[u,v]

                    '''Z = disparity_map[u, v] / scalingFactor
                    if Z == 0: continue
                    X = (u - centerX) * Z / focalLength
                    Y = (v - centerY) * Z / focalLength
                    Z = Baseline * focalLength / disparity_map[u, v]'''

                    points.append([X,Y,Z])
                    colors.append(color)

        print('points:{}, colors:{}'.format(np.shape(points), np.shape(colors)))
        points,colors = np.array(points), np.array(colors)
        write_ply('out1.ply', points, colors)
        print('out1.ply saved')

#--------------------------------------    ----------------------------------------------------------------------------
        points = cv2.reprojectImageTo3D(disparity_map, self.Q)
        print('points:{}---------------------------------------------------------------'.format(np.shape(points)))
        reflect_matrix = np.identity(3)  # reflect on x axis
        reflect_matrix[0] *= -1
        points = np.matmul(points, reflect_matrix)
        colors = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2RGB)

        # filter by min disparity
        mask = img > img.min()
        out_points =  points[mask]
        out_colors =  colors[mask]

        idx = np.fabs(out_points[:, 0]) < 2250  # 10.5 # filter by dimension
        out_points = out_points[idx]
        out_colors = out_colors.reshape(-1, 3)
        out_colors = out_colors[idx]
        write_ply(name, out_points, out_colors)
        print('%s saved' % name)

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

    def d2(self, imgL, imgR):
        import csv
        import random

        camera_focal_length_px = 1364.3372
        stereo_camera_baseline_m = 0.2090607502  # camera baseline in metres
        stereo_camera_baseline_m = 0.9663
        image_centre_h = 968.2298
        image_centre_w = 605.0811

        ## project_disparity_to_3d : project a given disparity image

        def project_disparity_to_3d(disparity, max_disparity, rgb=np.array([])):
            points = []

            f = camera_focal_length_px
            B = stereo_camera_baseline_m

            height, width = disparity.shape[:2]

            # Zmax = ((f * B) / 2)
            for y in range(height):  # 0 - height is the y axis index
                for x in range(width):  # 0 - width is the x axis index
                    if (disparity[y, x] > 0):
                        # calculate corresponding 3D point [X, Y, Z]
                        Z = (f * B) / disparity[y, x]
                        X = ((x - image_centre_w) * Z) / f
                        Y = ((y - image_centre_h) * Z) / f

                        if (rgb.size > 0):
                            points.append([X, Y, Z, rgb[y, x, 2], rgb[y, x, 1], rgb[y, x, 0]])
                        else:
                            points.append([X, Y, Z])

            return points

        # project a set of 3D points back the 2D image domain
        def project_3D_points_to_2D_image_points(points):
            points2 = []
            # Zmax = (camera_focal_length_px * stereo_camera_baseline_m) / 2
            for i1 in range(len(points)):
                # reverse earlier projection for X and Y to get x and y again
                x = ((points[i1][0] * camera_focal_length_px) / points[i1][2]) + image_centre_w
                y = ((points[i1][1] * camera_focal_length_px) / points[i1][2]) + image_centre_h
                points2.append([x, y])

            return points2

        max_disparity = 256
        stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 5)

        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        disparity_scaled = self.SGBM_disp
        points = project_disparity_to_3d(disparity_scaled, max_disparity)
        #points = project_disparity_to_3d(disparity_scaled, max_disparity, imgL)

        point_cloud_file = open('3d_points.txt', 'w')
        csv_writer = csv.writer(point_cloud_file, delimiter=' ')
        csv_writer.writerows(points)
        point_cloud_file.close()

    def depth(self, testImages, second = False):
        self.read_images(path=testImages)
        self.left_matcher = None
        wait = 0
        for i, fname in enumerate(self.LeftImg):
            leftFrame = self.LeftImg[i]
            rightFrame = self.RightImg[i]

            #left_rectified = cv2.remap(leftFrame, self.leftMapX, self.leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
            #right_rectified = cv2.remap(rightFrame, self.rightMapX, self.rightMapY, cv2.INTER_LINEAR,cv2.BORDER_CONSTANT)

            left_rectified = cv2.remap(leftFrame, self.rightMapX, self.rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
            right_rectified = cv2.remap(rightFrame, self.leftMapX, self.leftMapY, cv2.INTER_LINEAR,cv2.BORDER_CONSTANT)

            #out = right_rectified.copy()
            #out[:, :, 0] = left_rectified[:, :, 0]
            #out[:, :, 1] = left_rectified[:, :, 1]
            #out[:, :, 2] = right_rectified[:, :, 2]

            gray_left = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
            #disparity_image2 = cv2.applyColorMap(disparity_image2, cv2.COLORMAP_JET)

            img_top = cv2.hconcat(
                [cv2.resize(left_rectified, None, fx=self.fx, fy=self.fy), cv2.resize(right_rectified, None, fx=self.fx, fy=self.fy)])
            #cv2.imwrite('left_rectified1.png',left_rectified)
            #cv2.imwrite('right_rectified1.png', right_rectified)
            SGBM_disp, BM_disp = self.depth_map_SGBM(gray_left, gray_right, )  # Get the disparity map
            self.SGBM_disp = SGBM_disp
            #if second:
            #    self.d2(left_rectified, right_rectified)  # Get the disparity map

            img_bot = np.hstack((cv2.resize(SGBM_disp, None, fx=self.fx, fy=self.fy), cv2.resize(BM_disp, None, fx=self.fx, fy=self.fy)))
            #pointCloudColor = self.depth_and_color(img=BM_disp.copy(),left_rectified=left_rectified)
            pointCloudColor = self.depth_and_color(img=SGBM_disp.copy(),img2=BM_disp.copy(), left_rectified=left_rectified, gray_left=gray_left, gray_right=gray_right)

            cv2.imshow('img_bot ', img_bot)
            #cv2.imshow('out ', cv2.resize(out, None, fx=self.fx, fy=self.fy))
            cv2.imshow('Result', img_top)
            cv2.imshow('pointCloudColor', cv2.resize(pointCloudColor, None, fx=self.fx, fy=self.fy))

            k = cv2.waitKey(wait)
            if k & 0xFF == ord('q'):
                wait = 1
                #break
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def gridSearchdepth(self, testImages):
        self.read_images(path=testImages)
        self.left_matcher = None
        wait = 0
        for i, fname in enumerate(self.LeftImg):
            leftFrame = self.LeftImg[i]
            rightFrame = self.RightImg[i]

            left_rectified = cv2.remap(leftFrame, self.leftMapX, self.leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
            right_rectified = cv2.remap(rightFrame, self.rightMapX, self.rightMapY, cv2.INTER_LINEAR,
                                        cv2.BORDER_CONSTANT)

            gray_left = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)

            numDisparities = [128, 256,256*2,256*3,256*4,256*5,256*6,256*7,256*8,256*9]
            winsize = [5,15,35,75,125,235]
            for d in numDisparities:
                for w in winsize:
                    print('numDisparities:{}, winsize:{}'.format(d,w))
                    name = 'out_d{}_w{}.ply'.format(d,w)
                    self.left_matcher = None
                    SGBM_disp, BM_disp = self.depth_map_SGBM(gray_left, gray_right, numDisparities = d,window_size = w)  # Get the disparity map

                    pointCloudColor = self.depth_and_color(img=SGBM_disp.copy(), left_rectified=left_rectified, name=name)
                    cv2.imshow('pointCloudColor', cv2.resize(pointCloudColor, None, fx=self.fx, fy=self.fy))

                    cv2.waitKey(100)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def calibrateMono(self, imgChess, imgCharuco,see = False, save = True, cam = 'left'):
        Chess_calibrator = MonoChess_Calibrator(pattern_type="chessboard", pattern_rows=10,
                                          pattern_columns=7, distance_in_world_units=10,  # square is 10 cm
                                          figsize=(14, 10))
        Chess_calibrator.see = see
        Chess_calibrator.read_images(images_path_list=imgChess)
        Chess_objectPoints = np.array(Chess_calibrator.calibration_df.obj_points)
        Chess_imagePoints = np.array(Chess_calibrator.calibration_df.img_points)

        Charuco_calibrator = MonoCharuco_Calibrator()
        Charuco_calibrator.see = see
        Charuco_calibrator.createCalibrationBoard()
        Charuco_calibrator.read_images(images=imgCharuco, threads=1)
        Charuco_objectPoints = np.array(Charuco_calibrator.calibration_df.obj_points)
        Charuco_imagePoints = np.array(Charuco_calibrator.calibration_df.img_points)
        self.image_size = Charuco_calibrator.image_size
        print('Chess_objectPoints:{}, Chess_imagePoints:{}'.format(np.shape(Chess_objectPoints), np.shape(Chess_imagePoints)))
        print('Charuco_objectPoints:{}, Chess_imagePoints:{}'.format(np.shape(Charuco_objectPoints), np.shape(Charuco_imagePoints)))
        print('self.image_size ',self.image_size)
        _objectPoints = np.append(Chess_objectPoints,Charuco_objectPoints)
        _imagePoints = np.append(Chess_imagePoints, Charuco_imagePoints)
        print('Total charuco datapoints:{}'.format(Charuco_calibrator.total))
        total = Charuco_calibrator.total + (len(Chess_objectPoints)*70)
        print('Total datapoints:{}'.format(total))

        #for each distortion model perform the fucking calibration
        Distorsion_models = {'ST': ['Standard', 0, 'Standard'],
                             'RAT': ['Rational', cv2.CALIB_RATIONAL_MODEL, 'CALIB_RATIONAL_MODEL'],
                             'THP': ['Thin Prism', cv2.CALIB_THIN_PRISM_MODEL, 'CALIB_THIN_PRISM_MODEL'],
                             'TIL': ['Tilded', cv2.CALIB_TILTED_MODEL, 'CALIB_TILTED_MODEL'],  # }
                             'RAT+THP': ['Rational+Thin Prism', cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_THIN_PRISM_MODEL,
                                         'CALIB_RATIONAL_MODEL + CALIB_THIN_PRISM_MODEL'],
                             'THP+TIL': ['Thin Prism+Tilded', cv2.CALIB_THIN_PRISM_MODEL + cv2.CALIB_TILTED_MODEL,
                                         'CALIB_THIN_PRISM_MODEL + CALIB_TILTED_MODEL'],
                             'RAT+TIL': ['Rational+Tilded', cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_TILTED_MODEL,
                                         'CALIB_RATIONAL_MODEL + CALIB_TILTED_MODEL'],
                             'CMP': ['Complete',
                                     cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_THIN_PRISM_MODEL + cv2.CALIB_TILTED_MODEL,
                                     'Complete']}

        calibration_results = pd.DataFrame(
            {"params": ['fx', 'fy', 'px', 'py', 'sk', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6',
                        's1', 's2', 's3', 's4', 'tx', 'ty', 'Error']})
        rms_all = ['Error']

        for key in Distorsion_models:
            print()
            print(key, '->', Distorsion_models[key][0], ' , ', Distorsion_models[key][1], ' , ',
                  Distorsion_models[key][2])
            flags = Distorsion_models[key][1]

            #self.calibrate(flags=flags, project=False, K=K)
            print('_objectPoints:{}, _imagePoints:{}'.format(np.shape(_objectPoints), np.shape(_imagePoints)))
            self.rms, self.K, self.D, self.rvecs, self.tvecs = cv2.calibrateCamera(
                objectPoints=_objectPoints,
                imagePoints=_imagePoints,
                imageSize=self.image_size,
                cameraMatrix=None, distCoeffs=None,
                flags=flags, criteria=self.term_criteria)

            print("\nRMS:", self.rms)
            print("camera matrix:\n", self.K)
            print("distortion coefficients: ", self.D.ravel())

            s = np.array([self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2], self.K[0, 1]])
            s = np.append(s, self.D)
            calibration_results[str(key)] = pd.Series(s)
            calibration_results.fillna('---', inplace=True)
            rms_all.append(self.rms)

        calibration_results.iloc[-1, :] = rms_all
        calibration_results.iloc[-1, :] = calibration_results.iloc[-1, :].map(
            lambda x: round(x, 4) if isinstance(x, (int, float)) else x)

        save_csv(obj=calibration_results, name='FinalMonoCalibration_{}_{}_points'.format(self.name, total))







def MonoCombinedCalibration():
    calibrator = CombinedCalibration(name='right_outside')
    # calibrate right camera outside --------------------------------------------------------
    #imgCharuco = glob.glob('/home/eugeniu/Desktop/my_data/CameraCalibration/data/car_cam_data/charuco/outside/Right/*.png')
    #imgChess = glob.glob('/home/eugeniu/Desktop/my_data/CameraCalibration/data/car_cam_data/chess/outside/Right/*.png')
    #calibrator.calibrateMono(imgChess, imgCharuco, save=True, cam='right')

    # calibrate left camera outside-----------------------------------------------------------
    #calibrator.name = 'left_outside'
    #imgCharuco = glob.glob('/home/eugeniu/Desktop/my_data/CameraCalibration/data/car_cam_data/charuco/outside/Left/*.png')
    #imgChess = glob.glob('/home/eugeniu/Desktop/my_data/CameraCalibration/data/car_cam_data/chess/outside/Left/*.png')
    #calibrator.calibrateMono(imgChess, imgCharuco, save=True, cam='left')

    # calibrate right camera inside ----------------------------------------------------------
    calibrator.name = 'right_inside'
    imgCharuco = glob.glob('/home/eugeniu/Desktop/my_data/CameraCalibration/data/car_cam_data/charuco/inside/Right/*.png')
    imgChess = glob.glob('/home/eugeniu/Desktop/my_data/CameraCalibration/data/car_cam_data/chess/inside/Right/*.png')
    calibrator.calibrateMono(imgChess, imgCharuco, save=True, cam='right')

    # calibrate left camera inside-----------------------------------------------------------
    #calibrator.name = 'left_inside'
    #imgCharuco = glob.glob('/home/eugeniu/Desktop/my_data/CameraCalibration/data/car_cam_data/charuco/inside/Left/*.png')
    #imgChess = glob.glob('/home/eugeniu/Desktop/my_data/CameraCalibration/data/car_cam_data/chess/inside/Left/*.png')
    #calibrator.calibrateMono(imgChess, imgCharuco, save=True, cam='left')

    return calibrator

