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

from CameraCalibration.scripts.MonoCharuco import MonoCharuco_Calibrator
from CameraCalibration.scripts.MonoChess import MonoChess_Calibrator
from CameraCalibration.scripts.StereoCharuco import StereoCharuco_Calibrator
from CameraCalibration.scripts.StereoChess import StereoChess_Calibrator

from utils import *

class CombinedCalibration(object):
    def __init__(self, name = ''):
        self.name = name
        self.term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 3000, 0.000001)
        self.flipVertically = True

        self.fx = 0.3
        self.fy = 0.35

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

    def calibrateStereo(self,imgChess, imgCharuco,see = False, save = True):
        Chess_calibrator = StereoChess_Calibrator(imgChess)
        Chess_calibrator.see = see
        Chess_calibrator.read_images(test=False)

        Charuco_calibrator = StereoCharuco_Calibrator(imgCharuco)
        Charuco_calibrator.see = see
        Charuco_calibrator.createCalibrationBoard()
        Charuco_calibrator.read_images(test=False)
        self.image_size = Charuco_calibrator.img_shape
        print('self.image_size ',self.image_size)
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
                                                                                  self.T,
                                                                                  flags=cv2.CALIB_ZERO_DISPARITY,
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

        self.roi_left, self.roi_right = camera_model_rectify['roi_left'],camera_model_rectify['roi_right']
        self.leftMapX, self.leftMapY = camera_model_rectify['leftMapX'], camera_model_rectify['leftMapY']
        self.rightMapX, self.rightMapY = camera_model_rectify['rightMapX'], camera_model_rectify['rightMapY']

        print('Rotation R')
        print(self.R)
        print('Translation T')
        print(self.T)
        print('Stereo data has been loaded...')

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

    def depth_map_SGBM(self, imgL, imgR):
        window_size = 5
        if self.left_matcher is None:
            self.left_matcher = cv2.StereoSGBM_create(minDisparity=5,
                                                        numDisparities=256,
                                                        blockSize=window_size,
                                                    )

            self.left_matcherasdasd = cv2.StereoSGBM_create(minDisparity=5,
                                                      numDisparities=16,
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

    def depth(self, testImages):
        self.read_images(path=testImages)
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
            #cv2.imshow('out ', cv2.resize(out, None, fx=self.fx, fy=self.fy))
            cv2.imshow('Result', img_top)
            cv2.imshow('pointCloudColor', cv2.resize(pointCloudColor, None, fx=self.fx, fy=self.fy))

            k = cv2.waitKey(wait)
            if k & 0xFF == ord('q'):
                wait = 1
                #break
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def MonoCombinedCalibration():
    calibrator = CombinedCalibration(name='outside')
    # calibrate right camera outside --------------------------------------------------------
    imgCharuco = glob.glob(
        '/home/eugeniu/Desktop/my_data/CameraCalibration/data/car_cam_data/charuco/outside/Right/*.png')
    imgChess = glob.glob(
        '/home/eugeniu/Desktop/my_data/CameraCalibration/data/car_cam_data/chess/outside/Right/*.png')
    calibrator.calibrateMono(imgChess, imgCharuco, save=True, cam='right')

    # calibrate right camera inside ----------------------------------------------------------
    calibrator.name = 'inside'
    imgCharuco = glob.glob(
        '/home/eugeniu/Desktop/my_data/CameraCalibration/data/car_cam_data/charuco/inside/Right/*.png')
    imgChess = glob.glob(
        '/home/eugeniu/Desktop/my_data/CameraCalibration/data/car_cam_data/chess/inside/Right/*.png')
    calibrator.calibrateMono(imgChess, imgCharuco, save=True, cam='right')

    # calibrate left camera outside-----------------------------------------------------------
    calibrator.name = 'outside'
    imgCharuco = glob.glob(
        '/home/eugeniu/Desktop/my_data/CameraCalibration/data/car_cam_data/charuco/outside/Left/*.png')
    imgChess = glob.glob(
        '/home/eugeniu/Desktop/my_data/CameraCalibration/data/car_cam_data/chess/outside/Left/*.png')
    calibrator.calibrateMono(imgChess, imgCharuco, save=True, cam='left')

    # calibrate left camera inside-----------------------------------------------------------
    calibrator.name = 'inside'
    imgCharuco = glob.glob(
        '/home/eugeniu/Desktop/my_data/CameraCalibration/data/car_cam_data/charuco/inside/Left/*.png')
    imgChess = glob.glob(
        '/home/eugeniu/Desktop/my_data/CameraCalibration/data/car_cam_data/chess/inside/Left/*.png')
    calibrator.calibrateMono(imgChess, imgCharuco, save=True, cam='left')

    return calibrator

