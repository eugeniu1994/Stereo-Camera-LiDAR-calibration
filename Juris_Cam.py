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

np.set_printoptions(suppress=True)
from sympy import *
class StereoChess_Calibrator(object):
    def __init__(self, path):
        self.term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 1000, 0.0001)
        self.square = 0.1  # m (the size of each chessboard square is 10cm)
        self.objp = np.zeros((10 * 7, 3), np.float32) #chessboard is 7x10
        self.objp[:, :2] = np.mgrid[0:10, 0:7].T.reshape(-1, 2) * self.square

        self.see = True
        self.path = path
        self.thermaImg, self.rgbImg, self.monoImg = [], [], []
        self.axis = np.float32([[0,0,0], [9,0,0], [0,7,0], [0,0,-5]]).reshape(-1,3)*self.square

    def draw(self, img, corners, imgpts):
        corner = tuple(corners[0])
        img = cv2.line(img, corner, tuple(imgpts[0]), (255, 0, 0), 5)
        img = cv2.line(img, corner, tuple(imgpts[1]), (0, 255, 0), 5)
        img = cv2.line(img, corner, tuple(imgpts[2]), (0, 0, 255), 5)
        return img

    def read_images(self):
        '''
            real all camera images (thermal, monochrome and rgb)
        '''
        thermal = glob.glob(self.path + '/themal_image_*.png')
        rgb = glob.glob(self.path + '/rgb_image_*.png')
        mono = glob.glob(self.path + '/monochrome_image_*.png')

        thermal.sort()
        rgb.sort()
        mono.sort()

        for i, fname in enumerate(thermal):
            thermal_img = cv2.imread(thermal[i])
            rgb_img = cv2.imread(rgb[i])
            mono_img = cv2.imread(mono[i])

            self.thermaImg.append(thermal_img)
            self.rgbImg.append(rgb_img)
            self.monoImg.append(mono_img)

        self.thermaImg, self.rgbImg, self.monoImg = np.array(self.thermaImg), np.array(self.rgbImg), np.array(
            self.monoImg)

        print('read_images: thermaImg->{}, rgbImg->{}, monoImg->{} '.format(np.shape(self.thermaImg),
                                                                            np.shape(self.rgbImg),
                                                                            np.shape(self.monoImg)))

    def read_points(self, camera=None):  # camera in [mono,rgb,thermal]
        '''
            extract points from camera (thermal, monochrome and rgb)
        '''
        self.see = True
        wait = 0
        if camera == 'mono':
            print('Mono camera calibration')
            images = self.monoImg.copy()
        elif camera == 'rgb':
            print('RGB camera calibration')
            images = self.rgbImg.copy()
        elif camera == 'thermal':
            print('Thermal camera calibration')
            images = self.thermaImg.copy()
        else:
            print('Add right camera')

        print('images -> {}'.format(np.shape(images)))
        objpoints, imgpoints, img_shape = [], [], 0
        # extract points
        for i, fname in enumerate(images):
            img = images[i]
            if camera == 'thermal':  # invert the thermal camera
                img = np.array(256 - img, dtype='uint8')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (10, 7), flags=cv2.CALIB_CB_ADAPTIVE_THRESH)
            if ret:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.term_criteria)
                cv2.drawChessboardCorners(img, (10, 7), corners2, ret)
                objpoints.append(self.objp)
                imgpoints.append(corners2)
            # else:
            # print('No board at {}'.format(i))
            if self.see:
                if camera == 'thermal':
                    cv2.imshow('Image', img)
                else:
                    cv2.imshow('Image', cv2.resize(img, None, fx=.4, fy=.4))

                k = cv2.waitKey(wait)
                if k % 256 == 32:  # pressed space
                    self.see = False
                    cv2.destroyAllWindows()

            img_shape = gray.shape[::-1]
        print('Camera {} objpoints->{},imgpoints->{}, img_shape->{}'.format(camera, np.shape(objpoints),
                                                                            np.shape(imgpoints), img_shape))
        return objpoints, imgpoints, img_shape

    def calibrate(self, camera=None):
        '''
            perform internal calibration for given camera
        '''
        objpoints, imgpoints, img_shape = self.read_points(camera)
        rms, K, D, _, _ = cv2.calibrateCamera(
            objectPoints=objpoints,
            imagePoints=imgpoints,
            imageSize=img_shape,
            cameraMatrix=None, distCoeffs=None,
            flags=0, criteria=self.term_criteria)

        print('{} camera calibration done with RMS:{}'.format(camera, rms))
        print('K')
        print(K)
        print('D')
        print(D)

        return K, D

    def stereoCalibrate(self, K_thermal, D_thermal,K,D, camera):  # camera in [rgb,thermal]
        '''
            perform stereo calibration between thermal camera and given camera (mono or rgb)
        '''
        objpoints = []  # 3d point in real world space
        imgpoints_l = []  # 2d points in image plane. - thermal camera
        imgpoints_r = []  # 2d points in image plane. - mono or rgb camera

        if camera == 'mono':
            Second_images = self.monoImg.copy()
        elif camera == 'rgb':
            Second_images = self.rgbImg.copy()

        images = self.thermaImg.copy()
        # extract points
        for i, fname in enumerate(images):
            thermal_img = np.array(256 - images[i], dtype='uint8')
            thermal_gray = cv2.cvtColor(thermal_img, cv2.COLOR_BGR2GRAY)
            self.img_shape = thermal_gray.shape[::-1]
            thermal_ret, thermal_corners = cv2.findChessboardCorners(thermal_gray, (10, 7),
                                                                     flags=cv2.CALIB_CB_ADAPTIVE_THRESH)

            img = Second_images[i]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.second_img_shape = gray.shape[::-1]
            ret, corners = cv2.findChessboardCorners(gray, (10, 7), flags=cv2.CALIB_CB_ADAPTIVE_THRESH)

            if thermal_ret and ret:
                objpoints.append(self.objp)
                imgpoints_l.append(thermal_corners)
                imgpoints_r.append(corners)


        print('Thermal -> {} cam, {}-poses'.format(camera, len(objpoints)))

        flags = cv2.CALIB_FIX_INTRINSIC

        rms_stereo, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints_l, imgpoints_r, K_thermal, D_thermal, K, D, imageSize=None, criteria=self.term_criteria, flags=flags)

        print('Stereo calibraion Therma-{} done'.format(camera))
        print('rms_stereo:{}'.format(rms_stereo))
        print('Rotation R')
        print(R)
        print('Translation T')
        print(T)

        return R,T,E,F

    def doStuff(self):
        '''
            -Read all images for all cameras
            -Do internal calibration for each cam
            -Estimate R rotation and T translation between thermal cam and mono cam
            -Estimate R rotation and T translation between thermal cam and rgb cam
            -Save the data
        '''
        #Read all images
        self.read_images()

        #Calibrate mono camera
        K_mono, D_mono = calib.calibrate(camera='mono')

        #Calibrate rgb camera
        K_rgb, D_rgb = calib.calibrate(camera='rgb')

        #Calibrate thermal camera
        K_thermal, D_thermal = calib.calibrate(camera='thermal')

        #Stereo calibrate between Thermal and Mono camera
        R_th_mono, T_th_mono, E_th_mono, F_th_mono = self.stereoCalibrate(K_thermal,D_thermal,K_mono,D_mono,camera='mono')

        # Stereo calibrate between Thermal and Rgb camera
        R_th_rgb, T_th_rgb, E_th_rgb, F_th_rgb = self.stereoCalibrate(K_thermal, D_thermal, K_rgb, D_rgb, camera='rgb')

        calib_data = dict([('K_mono', K_mono), ('D_mono', D_mono),
                            ('K_rgb', K_rgb),('D_rgb', D_rgb),
                            ('K_thermal', K_thermal), ('D_thermal', D_thermal),
                            ('R_th_mono', R_th_mono), ('T_th_mono', T_th_mono),('E_th_mono', E_th_mono), ('F_th_mono', F_th_mono),
                            ('R_th_rgb', R_th_rgb), ('T_th_rgb', T_th_rgb), ('E_th_rgb', E_th_rgb),('F_th_rgb', F_th_rgb),
                            ])

        with open('calib_data.pkl', 'wb') as f:
            pickle.dump(calib_data, f, protocol=2)
        print('calib_data.pkl Object saved')

    def testCalibration(self):
        '''
            -loads images
            -load the calibration data
            -check if patter is visible in all 3 images:
                -Estimate the extrinsic R,T from world to thermal camera
                -Use estimated R,T and reproject pixels from thermal camera to mono and rgb cam
        '''
        self.thermaImg, self.rgbImg, self.monoImg = [], [], []
        # Read all images
        self.read_images()

        with open('calib_data.pkl', 'rb') as f:
            calib_data = pickle.load(f)
        K_mono = calib_data['K_mono']
        D_mono = calib_data['D_mono']
        K_rgb = calib_data['K_rgb']
        D_rgb = calib_data['D_rgb']
        K_thermal = calib_data['K_thermal']
        D_thermal = calib_data['D_thermal']
        R_th_mono = calib_data['R_th_mono']
        T_th_mono = calib_data['T_th_mono']
        R_th_rgb = calib_data['R_th_rgb']
        T_th_rgb = calib_data['T_th_rgb']

        F = calib_data['F_th_rgb']

        # Define test the calibration-----------------------
        for i, fname in enumerate(self.thermaImg):
            thermal_img = np.array(256 - self.thermaImg[i], dtype='uint8')
            thermal_gray = cv2.cvtColor(thermal_img, cv2.COLOR_BGR2GRAY)
            thermal_ret, thermal_corners = cv2.findChessboardCorners(thermal_gray, (10, 7),
                                                                     flags=cv2.CALIB_CB_ADAPTIVE_THRESH)
            mono_img = self.monoImg[i]
            mono_gray = cv2.cvtColor(mono_img, cv2.COLOR_BGR2GRAY)
            mono_ret, mono_corners = cv2.findChessboardCorners(mono_gray, (10, 7), flags=cv2.CALIB_CB_ADAPTIVE_THRESH)

            rgb_img = self.rgbImg[i]
            rgb_gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
            rgb_ret, _ = cv2.findChessboardCorners(rgb_gray, (10, 7), flags=cv2.CALIB_CB_ADAPTIVE_THRESH)

            if thermal_ret and rgb_ret and mono_ret:
                thermal_corners2 = cv2.cornerSubPix(thermal_gray, thermal_corners, (11, 11), (-1, -1),
                                                    self.term_criteria)

                # Find the rotation and translation vectors.
                ret, rvecs, tvecs = cv2.solvePnP(self.objp, thermal_corners2, K_thermal, D_thermal)
                # project 3D points to thermal image plane
                imgpts_thermal, jac = cv2.projectPoints(self.axis[1:], rvecs, tvecs, K_thermal,
                                                        D_thermal)  # thermal camera frame
                thermaImg = self.draw(thermal_img, np.asarray(thermal_corners2).squeeze(),
                                      np.asarray(imgpts_thermal).squeeze())

                T_01 = np.vstack(
                    (np.hstack((cv2.Rodrigues(rvecs)[0], tvecs)), [0, 0, 0, 1]))  # from world to thermal camera

                # project thermal to rgb --------------------------------------------------------------------------------------
                T_12 = np.vstack((np.hstack((R_th_rgb, T_th_rgb)), [0, 0, 0, 1]))  # from thermal cam to rgb cam
                T = np.dot(T_12, T_01)  # world to rgb cam
                rotation, translation = T[:3, :3], T[:3, -1]
                imgpts_rgb, _ = cv2.projectPoints(self.axis, rotation, translation, K_rgb, D_rgb)
                imgpts_rgb = np.array(imgpts_rgb).squeeze()
                rgbImg = self.draw(rgb_img, [imgpts_rgb[0]], imgpts_rgb[1:])

                # project thermal to mono ------------------------------------------------------------------------------------
                '''T_12 = np.vstack((np.hstack((R_th_mono, T_th_mono)), [0, 0, 0, 1]))  # from thermal cam to mono cam
                T = np.dot(T_12, T_01)  # world to mono cam
                rotation, translation = T[:3, :3], T[:3, -1]
                imgpts_mono, _ = cv2.projectPoints(self.axis, rotation, translation, K_mono, D_mono)
                imgpts_mono = np.array(imgpts_mono).squeeze()
                monoImg = self.draw(mono_img, [imgpts_mono[0]], imgpts_mono[1:])'''

                thermal_corners2 = np.array(thermal_corners2).squeeze()
                x_1 = thermal_corners2[0] #pixel in thermal camera
                x_1 = np.array([x_1[0],x_1[1],1])
                print(x_1)
                '''Z = 1
                Z = tvecs[-1]
                print('tvecs -> {},   Z:{}'.format(tvecs,Z))
                x_1 = x_1*Z
                X_cam1 = np.linalg.inv(K_thermal).dot(x_1)
                X_cam1 = np.array([X_cam1[0],X_cam1[1],X_cam1[2],1])
                print('X_cam1 -> {}'.format(X_cam1))
                P = np.hstack((R_th_rgb, T_th_rgb))  # from thermal cam to rgb cam
                print(P)
                x_2 = K_rgb.dot(P) @ X_cam1
                print('x_2 -> {}'.format(x_2))
                x_2 = np.array([x_2[0]/x_2[-1],x_2[1]/x_2[-1]]).astype(int)
                print('x_2 -> {}'.format(x_2))
                print('rgbImg -> {}'.format(np.shape(rgbImg)))
                cv2.circle(rgbImg, (x_2[0], x_2[1]), 12, (0, 255, 0), 12)

                cv2.circle(thermaImg, (thermal_corners2[0][0], thermal_corners2[0][1]), 6, (0, 255, 0), 6)'''

                print('F')
                print(F)

                #x_1 * F * x_2 = 0

                x1 = np.asarray(thermaImg).reshape(-1,3)
                x2 = np.asarray(rgbImg).reshape(-1,3)
                print('x1:{}, F:{}, x2:{}'.format(np.shape(x1), np.shape(F),np.shape(x2)))

                x1F = x1 @ F
                print('x1 * F = {}'.format(np.shape(x1F)))
                x1Fx2 = x1F.dot(x2.T)
                print('x1Fx2= {}'.format(np.shape(x1Fx2)))

                cv2.imshow('thermal_img', thermaImg)
                #cv2.imshow('monoImg', cv2.resize(monoImg, None, fx=.4, fy=.4))
                cv2.imshow('rgbImg', cv2.resize(rgbImg, None, fx=.3, fy=.3))

                cv2.waitKey(0)

        cv2.destroyAllWindows()



if __name__ == '__main__':
    path = '/home/eugeniu/cool'
    calib = StereoChess_Calibrator(path)
    #calib.doStuff()   #this function load the data, does internal and stereo calibration - > save the data
    calib.testCalibration()
