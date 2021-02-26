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
import cv2.aruco as aruco

from CameraCalibration.scripts.MonoCharuco import MonoCharuco_Calibrator
from utils import *

np.set_printoptions(suppress=True)

class InsideOutside_Calibrator(object):
    def __init__(self, filepath, name=''):
        self.term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 5000000, 0.000000001)
        #self.term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 0.001)

        self.objpoints_i = []
        self.imgpoints_i = []
        self.objpoints_o = []
        self.imgpoints_o = []
        self.cal_path = filepath
        self.totalI = 0
        self.totalO = 0

        self.fx = 0.4
        self.fy = 0.45
        self.position = (20, 30)
        self.see = True
        self.flipVertically = True
        self.name = name
        self.K_inside, self.K_outside, self.D_inside, self.D_outside = None, None, None, None
        self.createCalibrationBoard()
        self.image_width = 1936
        self.image_height = 1216
        self.wait = 0

    def createCalibrationBoard(self, squaresY=6, squaresX=15, squareLength=.027, markerLength=0.0205):
        self.pattern_columns = squaresX
        self.pattern_rows = squaresY
        self.distance_in_world_units = squareLength
        self.ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_4X4_1000)
        self.CHARUCO_BOARD = aruco.CharucoBoard_create(
            squaresX=squaresX, squaresY=squaresY,
            squareLength=squareLength,
            markerLength=markerLength,
            dictionary=self.ARUCO_DICT)

    def read_images(self, limit = 5 ,K=None,D=None):
        images_in = glob.glob(self.cal_path + '/inside/*.png')
        images_out = glob.glob(self.cal_path + '/outside/*.png')

        images_in.sort()
        images_out.sort()
        self.inside_images, self.outside_images, self.image_names = [], [], []
        h, w = self.CHARUCO_BOARD.chessboardCorners.shape
        self.wait = 0
        self.objpoints_i = []
        self.imgpoints_i = []
        self.objpoints_o = []
        self.imgpoints_o = []
        if K is not None and D is not None:  # undistort images before calibration
            self.h, self.w = cv2.imread(images_in[0], 0).shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, D, (self.w, self.h), 1, (self.w, self.h))

            def undistort_image(img):
                if False:
                    mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, newcameramtx, (self.w, self.h), 5)
                    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
                    x, y, w1, h1 = roi
                    dst = dst[y:y + h1, x:x + w1]
                    self.h, self.w = dst.shape[:2]
                    #cv2.imshow('undistorted',dst)
                else:
                    distorted_frame = img
                    undistorted_frame = cv2.undistort(
                        distorted_frame, K, D, None, newcameramtx,
                    )
                    roi_x, roi_y, roi_w, roi_h = roi
                    cropped_frame = undistorted_frame[roi_y: roi_y + roi_h, roi_x: roi_x + roi_w]
                    dst = undistorted_frame

                return dst

        for i, fname in enumerate(images_out):

            img_in = cv2.imread(images_in[i])
            img_out = cv2.imread(images_out[i])

            if K is not None and D is not None:  # undistort images before calibration
                #print('Undistorted')
                img_in = undistort_image(img_in)
                img_out = undistort_image(img_out)

            if self.flipVertically:
                img_in = cv2.flip(img_in, -1)
                img_out = cv2.flip(img_out, -1)

            self.inside_images.append(img_in.copy())
            self.outside_images.append(img_out.copy())
            self.image_names.append(os.path.basename(images_in[i]))
            #print('img {}'.format(os.path.basename(images_in[i])))
            gray_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
            gray_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY)

            cornersIn, idsIn, rejectedIn = aruco.detectMarkers(image=gray_in, dictionary=self.ARUCO_DICT)
            cornersIn, idsIn, rejectedImgPointsIn, recoveredIdsIn = aruco.refineDetectedMarkers(image=gray_in,
                                                                                            board=self.CHARUCO_BOARD,
                                                                                            detectedCorners=cornersIn,
                                                                                            detectedIds=idsIn,
                                                                                            rejectedCorners=rejectedIn,
                                                                                            cameraMatrix=self.K_inside,
                                                                                            distCoeffs=self.D_inside)
            ret_in = len(cornersIn) >= limit

            cornersOut, idsOut, rejectedOut = aruco.detectMarkers(image=gray_out, dictionary=self.ARUCO_DICT)
            cornersOut, idsOut, rejectedImgPointsOut, recoveredIdsOut = aruco.refineDetectedMarkers(image=gray_out,
                                                                                                board=self.CHARUCO_BOARD,
                                                                                                detectedCorners=cornersOut,
                                                                                                detectedIds=idsOut,
                                                                                                rejectedCorners=rejectedOut,
                                                                                                cameraMatrix=self.K_outside,
                                                                                                distCoeffs=self.D_outside)
            ret_out = len(cornersOut) >= limit



            if ret_in and ret_out:
                insideHalf = img_in[:int(self.image_height/2),:,:]
                outsideHalf = img_out[int(self.image_height/2):,:,:]
                vertically_splitted = np.concatenate((insideHalf, outsideHalf), axis=0)

                insideHalf = img_in[:, :int(self.image_width / 2), :]
                outsideHalf = img_out[:, int(self.image_width / 2):, :]
                horizontally_splitted = np.concatenate((insideHalf, outsideHalf), axis=1)


                img_in = aruco.drawDetectedMarkers(img_in, cornersIn, idsIn)
                img_out = aruco.drawDetectedMarkers(img_out, cornersOut, idsOut)

                responseIn, charuco_cornersIn, charuco_idsIn = aruco.interpolateCornersCharuco(markerCorners=cornersIn, markerIds=idsIn,
                        image=gray_in, board=self.CHARUCO_BOARD)
                responseOut, charuco_cornersOut, charuco_idsOut = aruco.interpolateCornersCharuco(markerCorners=cornersOut,markerIds=idsOut,
                                                                                                image=gray_out,board=self.CHARUCO_BOARD)

                if responseIn >= limit and responseOut >= limit:
                    imgPtsIn = np.array(charuco_cornersIn)
                    imgPtsOut = np.array(charuco_cornersOut)

                    objPtsIn  = self.CHARUCO_BOARD.chessboardCorners.reshape((h, 1, 3))[np.asarray(charuco_idsIn).squeeze()]
                    objPtsOut = self.CHARUCO_BOARD.chessboardCorners.reshape((h, 1, 3))[np.asarray(charuco_idsOut).squeeze()]

                    if objPtsIn is not None and objPtsOut is not None:
                        if len(objPtsIn) >= limit and len(objPtsOut) >= limit:
                            #print('objPtsIn:{},imgPtsL:{}'.format(np.shape(objPtsIn), np.shape(imgPtsIn)))
                            #print('objPtsOut:{},imgPtsR:{}'.format(np.shape(objPtsOut), np.shape(imgPtsOut)))
                            #print('')

                            self.objpoints_i.append(objPtsIn)
                            self.imgpoints_i.append(imgPtsIn)

                            self.objpoints_o.append(objPtsOut)
                            self.imgpoints_o.append(imgPtsOut)

                            self.totalI+=len(objPtsIn)
                            self.totalO+=len(imgPtsOut)

                if self.see:
                    cam_in_resized = cv2.resize(img_in, None, fx=self.fx, fy=self.fy)
                    cam_out_resized = cv2.resize(img_out, None, fx=self.fx, fy=self.fy)

                    cv2.putText(cam_in_resized, "Inside", self.position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255, 0))
                    cv2.putText(cam_out_resized, "Outside", self.position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255, 0))

                    im_h = cv2.hconcat([cam_in_resized, cam_out_resized])
                    cv2.imshow('Inside-outside correspondences', im_h)
                    #cv2.imshow('vertically_splitted ', cv2.resize(vertically_splitted, None, fx=.6, fy=.6))
                    cv2.imshow('horizontally_splitted ', cv2.resize(horizontally_splitted, None, fx=.6, fy=.6))

                    #cv2.imshow('inside', cam_in_resized)
                    #cv2.imshow('outside', cam_out_resized)
                    k = cv2.waitKey(self.wait)
                    if k % 256 == 32:
                        self.see = False
                        cv2.destroyAllWindows()
                    elif k & 0xFF == ord('q'):
                        self.wait = 50

        self.img_shape = gray_in.shape[::-1]
        cv2.destroyAllWindows()
        print('Ready for calibration')
        print('objPtsIn:{},imgPtsL:{}'.format(np.shape(self.objpoints_i), np.shape(self.imgpoints_i)))
        print('objPtsOut:{},imgPtsR:{}'.format(np.shape(self.objpoints_o), np.shape(self.imgpoints_o)))
        print('Points inside:{},  outside:{}'.format(self.totalI, self.totalO))

    def _calc_reprojection_error(self, errors_ = None, figure_size=(16, 12), name = ''):
        if len(errors_)>1:
            errInside = np.array(errors_[0]).squeeze()
            errOutside = np.array(errors_[1]).squeeze()
            data = []
            for i, p in enumerate(self.inside_images):
                data.append(['img_{}'.format(i), errInside[i], errOutside[i]])

            df = pd.DataFrame(data, columns=["Name", "Inside(px)", "Outside(px)"])

            ax = df.plot(x="Name", y=["Inside(px)", "Outside(px)"], kind="bar", figsize=figure_size, #grid=True,
                    title='Reprojection_error plot - {}'.format(name))

            avg_error_inside = np.sum(errInside) / len(errInside)
            y_mean_inside = [avg_error_inside] * len(self.inside_images)
            ax.plot(y_mean_inside, label='Mean Reprojection error inside:{}'.format(round(avg_error_inside,2)), linestyle='--')

            avg_error_outside = np.sum(errOutside) / len(errOutside)
            y_mean_outside = [avg_error_outside] * len(self.inside_images)
            ax.plot(y_mean_outside, label='Mean Reprojection error outside:{}'.format(round(avg_error_outside,2)), linestyle='--')

            ax.legend(loc='upper right')
            ax.set_xlabel("Image_names")
            ax.set_ylabel("Reprojection error in px")
            plt.show()

        elif errors_ is not None:
            errors = errors_
            errors = np.array(errors).squeeze()
            avg_error = np.sum(errors) / len(errors)
            print("The Mean Reprojection Error in pixels is:  {}".format(avg_error))
            x = ['img_{}'.format(i) for i,p in enumerate(self.inside_images)]
            y_mean = [avg_error] * len(self.inside_images)
            fig, ax = plt.subplots()
            fig.set_figwidth(figure_size[0])
            fig.set_figheight(figure_size[1])
            max_intensity = np.max(errors)
            cmap = cm.get_cmap('jet')
            colors = cmap(errors / max_intensity)  # * 255

            ax.scatter(x, errors, label='Reprojection error', c=colors, marker='o')  # plot before
            ax.bar(x, errors, color=colors, alpha=0.3)
            ax.plot(x, y_mean, label='Mean Reprojection error', linestyle='--')
            ax.legend(loc='upper right')
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)
            ax.set_title("{} - Reprojection_error plot, Mean:{}".format(name, round(avg_error, 2)))
            ax.set_xlabel("Image_names")
            ax.set_ylabel("Reprojection error in pixels")
            plt.show()

    def calibrate(self, flags=0, project=True, save=True, adjust = False):
        self.rmsOut, self.K_outside, self.D_outside, self.rvecsOut, self.tvecsOut,\
        _, _, self.err_Outside = cv2.calibrateCameraExtended(
            objectPoints=self.objpoints_o,
            imagePoints=self.imgpoints_o,
            imageSize=self.img_shape,
            cameraMatrix=self.K_outside, distCoeffs=self.D_outside,
            flags=flags, criteria=self.term_criteria)

        print("\nRMS outside:", self.rmsOut)
        print("K outside :\n", self.K_outside)
        print("D outside : ", self.D_outside.ravel())
        #if project:
        #    self._calc_reprojection_error(errors=self.err_Outside, name='outside')

        self.rmsIn, self.K_inside, self.D_inside, self.rvecsIn, self.tvecsIn, \
                _, _, self.err_Inside = cv2.calibrateCameraExtended(
            objectPoints=self.objpoints_i,
            imagePoints=self.imgpoints_i,
            imageSize=self.img_shape,
            cameraMatrix=self.K_inside, distCoeffs=self.D_inside,
            flags=flags, criteria=self.term_criteria)

        print("\nRMS inside:", self.rmsIn)
        print("K inside :\n", self.K_inside)
        print("D inside : ", self.D_inside.ravel())
        #if project:
        #    self._calc_reprojection_error(errors=self.err_Inside, name='inside')
        #self._calc_reprojection_error(errors_=[self.err_Inside, self.err_Outside], name='inside and outside')
        if save:
            result_dictionary = {
                "K_outside": self.K_outside,
                "D_outside": self.D_outside,
                "K_inside": self.K_inside,
                "D_inside": self.D_inside,
            }
            if save:
                save_obj(obj=result_dictionary, name='correspondences')


        if adjust:
            flags = 0
            flags |= cv2.CALIB_USE_INTRINSIC_GUESS
            flags |= cv2.CALIB_FIX_INTRINSIC
            flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
            flags |= cv2.CALIB_FIX_FOCAL_LENGTH

            rmsIn, K_inside, D_inside_full, _, _, \
            _, _, _ = cv2.calibrateCameraExtended(
                objectPoints=self.objpoints_i,
                imagePoints=self.imgpoints_i,
                imageSize=self.img_shape,
                cameraMatrix=self.K_outside, distCoeffs=None,
                flags=flags, criteria=self.term_criteria)

            print("\nRMS inside(given K):", rmsIn)
            print("K inside(given K) :\n", K_inside)
            print("D inside(given K) : ", D_inside_full.ravel())

            Windshield_distortion = np.abs(D_inside_full - self.D_outside) * np.sign(self.D_outside)
            print('Windshield_distortion')
            print(Windshield_distortion)

            self.see = True
            self.read_images(K=self.K_inside, D=Windshield_distortion)
            self.rmsIn, self.K_inside, self.D_inside, self.rvecsIn, self.tvecsIn, \
            _, _, self.err_Inside = cv2.calibrateCameraExtended(
                objectPoints=self.objpoints_i,
                imagePoints=self.imgpoints_i,
                imageSize=self.img_shape,
                cameraMatrix=self.K_inside, distCoeffs=self.D_inside,
                flags=flags, criteria=self.term_criteria)

            print("\nadjusted RMS inside:", self.rmsIn)
            print("adjusted K inside :\n", self.K_inside)
            print("adjusted D inside : ", self.D_inside.ravel())

            '''flags=0
            flags |= cv2.CALIB_FIX_K1
            flags |= cv2.CALIB_FIX_K2
            flags |= cv2.CALIB_FIX_K3
            self.read_images()
            rmsIn, K_inside, D_inside_full, _, _, \
            _, _, _ = cv2.calibrateCameraExtended(
                objectPoints=self.objpoints_i,
                imagePoints=self.imgpoints_i,
                imageSize=self.img_shape,
                cameraMatrix=self.K_outside, distCoeffs=None,
                flags=flags, criteria=self.term_criteria)
            print("\nRMS inside(given D):", rmsIn)
            print("K inside(given D) :\n", K_inside)
            print("D inside(given D) : ", D_inside_full.ravel())'''

    def readMonoData2(self):
        data = load_obj(name='correspondences')
        self.K_outside = data['K_outside']
        self.D_outside = data['D_outside']
        self.K_inside = data['K_inside']
        self.D_inside = data['D_inside']
        print('Loaded mono calibration data')

        print("K outside :\n", self.K_outside)
        print("D outside : ", self.D_outside.ravel())
        print('-----------------------------------------')
        print("K inside :\n", self.K_inside)
        print("D inside : ", self.D_inside.ravel())

    def readMonoData(self):
        right_ = load_obj(name='combined_{}_right'.format('inside'))
        self.K_inside = right_['K']
        self.D_inside = right_['D']

        right_ = load_obj(name='combined_{}_right'.format('outside'))
        self.K_outside = right_['K']
        self.D_outside = right_['D']
        print('Loaded mono calibration data')

        print("K outside :\n", self.K_outside)
        print("D outside : ", self.D_outside.ravel())
        print('-----------------------------------------')
        print("K inside :\n", self.K_inside)
        print("D inside : ", self.D_inside.ravel())

    def estimatePose(self, limit = 3):
        def rmse(predictions, targets, deg = False):
            if deg:
                diff = [math.atan2(math.sin(x - y), math.cos(x - y)) for x,y in zip(predictions,targets)]
                diff = [math.degrees(i) for i in diff]
                return np.abs(diff).mean()
            else:
                #return np.sqrt(((predictions - targets) ** 2).mean())
                return np.abs(predictions - targets).mean()

        self.wait = 0
        self.see=True
        Inside, Outside, AngleIn, AngleOut = [],[],[],[]
        E = []
        goodImages = []
        for i, img in enumerate(self.inside_images):

            img_in = self.inside_images[i]
            img_out = self.outside_images[i]

            gray_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
            gray_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY)

            cornersIn, idsIn, rejectedIn = aruco.detectMarkers(image=gray_in, dictionary=self.ARUCO_DICT)
            cornersIn, idsIn, rejectedImgPointsIn, recoveredIdsIn = aruco.refineDetectedMarkers(image=gray_in,
                                                                                                board=self.CHARUCO_BOARD,
                                                                                                detectedCorners=cornersIn,
                                                                                                detectedIds=idsIn,
                                                                                                rejectedCorners=rejectedIn,
                                                                                                cameraMatrix=self.K_inside,
                                                                                                distCoeffs=self.D_inside)
            ret_in = len(cornersIn) >= limit

            cornersOut, idsOut, rejectedOut = aruco.detectMarkers(image=gray_out, dictionary=self.ARUCO_DICT)
            cornersOut, idsOut, rejectedImgPointsOut, recoveredIdsOut = aruco.refineDetectedMarkers(image=gray_out,
                                                                                                    board=self.CHARUCO_BOARD,
                                                                                                    detectedCorners=cornersOut,
                                                                                                    detectedIds=idsOut,
                                                                                                    rejectedCorners=rejectedOut,
                                                                                                    cameraMatrix=self.K_outside,
                                                                                                    distCoeffs=self.D_outside)
            ret_out = len(cornersOut) >= limit
            responseIn, charuco_cornersIn, charuco_idsIn = aruco.interpolateCornersCharuco(markerCorners=cornersIn,
                                                                                           markerIds=idsIn,
                                                                                           image=gray_in,
                                                                                           board=self.CHARUCO_BOARD)
            responseOut, charuco_cornersOut, charuco_idsOut = aruco.interpolateCornersCharuco(markerCorners=cornersOut,
                                                                                              markerIds=idsOut,
                                                                                              image=gray_out,
                                                                                              board=self.CHARUCO_BOARD)

            if responseIn >= limit and responseOut >= limit:
                imgPtsIn = np.array(charuco_cornersIn).squeeze()
                imgPtsOut = np.array(charuco_cornersOut).squeeze()
                if len(imgPtsIn) == len(imgPtsOut):
                    #print('imgPtsIn:{},  imgPtsOut:{}'.format(np.shape(imgPtsIn), np.shape(imgPtsOut)))
                    a = rmse(imgPtsIn , imgPtsOut)
                    #print('a:{}')
                    E.append(a)
                    goodImages.append(self.image_names[i])
                else:
                    continue

            if ret_in and ret_out:
                #img_in = aruco.drawDetectedMarkers(img_in, cornersIn, idsIn)
                #img_out = aruco.drawDetectedMarkers(img_out, cornersOut, idsOut)

                self.retvalIn, self.rvecI, self.tvecI = aruco.estimatePoseBoard(cornersIn, idsIn, self.CHARUCO_BOARD, self.K_inside,
                                                                            self.D_inside, None, None)
                img_in = aruco.drawAxis(img_in, self.K_outside, self.D_outside, self.rvecI, self.tvecI, 0.25)

                self.retvalOut, self.rvecO, self.tvecO = aruco.estimatePoseBoard(cornersOut, idsOut, self.CHARUCO_BOARD,
                                                                                self.K_outside,
                                                                                self.D_outside, None, None)
                img_out = aruco.drawAxis(img_out, self.K_outside, self.D_outside, self.rvecO, self.tvecO, 0.25)

                Inside.append(np.asarray(self.tvecI).squeeze())
                Outside.append(np.asarray(self.tvecO).squeeze())

                #self.dstIn, jacobian = cv2.Rodrigues(self.rvecI)
                anglesIn = np.asarray(self.rvecI).squeeze()  #rot2eul(self.dstIn)

                #self.dstOut, jacobian = cv2.Rodrigues(self.rvecO)
                anglesOut = np.asarray(self.rvecO).squeeze()  # rot2eul(self.dstOut)
                AngleIn.append(anglesIn)
                AngleOut.append(anglesOut)

                if self.see:
                    cam_in_resized = cv2.resize(img_in, None, fx=self.fx, fy=self.fy)
                    cam_out_resized = cv2.resize(img_out, None, fx=self.fx, fy=self.fy)

                    cv2.putText(cam_in_resized, "Inside", self.position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255, 0))
                    cv2.putText(cam_out_resized, "Outside", self.position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255, 0))

                    im_h = cv2.hconcat([cam_in_resized, cam_out_resized])
                    cv2.imshow('Inside-outside correspondences', im_h)

                    k = cv2.waitKey(self.wait)
                    if k % 256 == 32:
                        self.see = False
                        cv2.destroyAllWindows()
                    elif k & 0xFF == ord('q'):
                        self.wait = 50


        cv2.destroyAllWindows()
        Inside, Outside = np.array(Inside), np.array(Outside)
        AngleOut, AngleIn = np.array(AngleOut), np.array(AngleIn)

        fig, axs = plt.subplots(3, figsize=(12,10),sharex=True,)
        fig.suptitle('RMS translation mm error X:{}mm,  Y:{}mm,  Z:{}mm'.format(round(rmse(predictions=Inside[:, 0],targets=Outside[:, 0])*1000,2) ,
                                                         round(rmse(predictions=Inside[:, 1],targets=Outside[:, 1])*1000 ,2),
                                                         round(rmse(predictions=Inside[:, 2],targets=Outside[:, 2])*1000 ,2)))
        axs[0].plot(Inside[:, 0], color = 'r', label='Inside')
        axs[0].plot(Outside[:, 0], color = 'g', label='Outside')
        axs[0].grid(True)
        axs[0].legend(loc="upper right")
        axs[0].set_ylabel('translation X',fontweight='bold')

        axs[1].plot(Inside[:, 1], color = 'r', label='Inside')
        axs[1].plot(Outside[:, 1], color='g', label='Outside')
        axs[1].grid(True)
        axs[1].legend(loc="upper right")
        axs[1].set_ylabel('translation Y',fontweight='bold')

        axs[2].plot(Inside[:, 2], color = 'r', label='Inside')
        axs[2].plot(Outside[:, 2], color='g', label='Outside')
        axs[2].grid(True)
        axs[2].legend(loc="upper right")
        axs[2].set_ylabel('translation Z',fontweight='bold')

        plt.xlabel("Images")
        plt.show()
        #---------------------------------------------------------------------
        fig, axs = plt.subplots(3, figsize=(12, 10),sharex=True,)
        fig.suptitle('RMS rotation degrees error X:{},  Y:{},  Z:{}'.format(round(rmse(predictions=AngleIn[:, 0], targets=AngleOut[:, 0],deg=True) ,2),
                                                          round(rmse(predictions=AngleIn[:, 1], targets=AngleOut[:, 1],deg=True),2) ,
                                                          round(rmse(predictions=AngleIn[:, 2], targets=AngleOut[:, 2],deg=True),2) ))
        axs[0].plot(AngleIn[:, 0], color='r', label='Inside')
        axs[0].plot(AngleOut[:, 0], color='g', label='Outside')
        axs[0].grid(True)
        axs[0].legend(loc="upper right")
        axs[0].set_ylabel('rotation X', fontweight='bold')

        axs[1].plot(AngleIn[:, 1], color='r', label='Inside')
        axs[1].plot(AngleOut[:, 1], color='g', label='Outside')
        axs[1].grid(True)
        axs[1].legend(loc="upper right")
        axs[1].set_ylabel('rotation Y', fontweight='bold')

        axs[2].plot(AngleIn[:, 2], color='r', label='Inside')
        axs[2].plot(AngleOut[:, 2], color='g', label='Outside')
        axs[2].grid(True)
        axs[2].legend(loc="upper right")
        axs[2].set_ylabel('rotation Z', fontweight='bold')

        plt.xlabel("Images")
        plt.show()

    def solve(self):
        self.objpoints_i = []
        self.imgpoints_i = []
        print('solve')
        self.wait = 0
        self.see=True
        for i, img in enumerate(self.inside_images):
            inside = cv2.resize(self.inside_images[i], None, fx=self.fx, fy=self.fy)

            h, w = img.shape[:2]
            #print('h:{},  w:{}'.format(h, w))
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.K_outside,
                                                              self.D_outside,
                                                              (w, h), 1, (w, h))

            # undistort
            dst = cv2.undistort(inside, self.K_outside, self.D_outside, None, newcameramtx)
            x, y, w, h = roi
            img_in = dst[y:y + h, x:x + w]
            inside_und = cv2.resize(img_in, None, fx=self.fx, fy=self.fy)
            h, w = self.CHARUCO_BOARD.chessboardCorners.shape
            gray_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)

            cornersIn, idsIn, rejectedIn = aruco.detectMarkers(image=gray_in, dictionary=self.ARUCO_DICT)
            cornersIn, idsIn, rejectedImgPointsIn, recoveredIdsIn = aruco.refineDetectedMarkers(image=gray_in,
                                                                                                board=self.CHARUCO_BOARD,
                                                                                                detectedCorners=cornersIn,
                                                                                                detectedIds=idsIn,
                                                                                                rejectedCorners=rejectedIn,
                                                                                                cameraMatrix=self.K_outside,
                                                                                                distCoeffs=self.D_outside)

            ret_in = len(cornersIn) >= 10

            if ret_in:
                img_in = aruco.drawDetectedMarkers(img_in, cornersIn, idsIn)
                responseIn, charuco_cornersIn, charuco_idsIn = aruco.interpolateCornersCharuco(markerCorners=cornersIn,
                                                                                               markerIds=idsIn,
                                                                                               image=gray_in,
                                                                                               board=self.CHARUCO_BOARD)

                if responseIn >= 10 :
                    imgPtsIn = np.array(charuco_cornersIn)
                    objPtsIn = self.CHARUCO_BOARD.chessboardCorners.reshape((h, 1, 3))[np.asarray(charuco_idsIn).squeeze()]


                    if objPtsIn is not None:
                        if len(objPtsIn) >= 10:
                            self.objpoints_i.append(objPtsIn)
                            self.imgpoints_i.append(imgPtsIn)

            if self.see:
                cam_in_resized = cv2.resize(img_in, None, fx=self.fx, fy=self.fy)

                cv2.putText(cam_in_resized, "Inside", self.position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255, 0))

                cv2.imshow('cam_in_resized ', cam_in_resized)
                cv2.imshow('inside_und ', inside_und)

                k = cv2.waitKey(self.wait)
                if k % 256 == 32:
                    self.see = False
                    cv2.destroyAllWindows()
                elif k & 0xFF == ord('q'):
                    self.wait = 50

        self.img_shape = gray_in.shape[::-1]
        cv2.destroyAllWindows()
        print('Ready for calibration')
        print('objPtsIn:{},imgpoints_i:{}'.format(np.shape(self.objpoints_i), np.shape(self.imgpoints_i)))

        self.rmsIn, self.K_inside, self.D_inside, self.rvecsIn, self.tvecsIn, \
        _, _, self.err_Inside = cv2.calibrateCameraExtended(
            objectPoints=self.objpoints_i,
            imagePoints=self.imgpoints_i,
            imageSize=self.img_shape,
            cameraMatrix=self.K_inside, distCoeffs=self.D_inside,
            flags=flags, criteria=self.term_criteria)

        print('Corrected data')
        print("\nRMS inside:", self.rmsIn)
        print("K inside :\n", self.K_inside)
        print("D inside : ", self.D_inside.ravel())

    def test(self):
        imIn = cv2.imread(
            '/home/eugeniu/Desktop/my_data/CameraCalibration/data/car_cam_data/correspondences/inside/Inside_48.png')[
               700:, :1300, :]
        imOut = cv2.imread(
            '/home/eugeniu/Desktop/my_data/CameraCalibration/data/car_cam_data/correspondences/outside/Outside_48.png')[
                700:, :1300, :]

        cv2.imshow('inside ', cv2.resize(imIn, None, fx=.5, fy=.5))
        cv2.imshow('outside ', cv2.resize(imOut, None, fx=.5, fy=.5))

        image_height, image_width, _ = np.shape(imIn)
        print('shape is ', np.shape(imIn))
        insideHalf = imIn[:int(image_height / 2), :, :]
        outsideHalf = imOut[int(image_height / 2):, :, :]
        vertically_splitted = np.concatenate((insideHalf, outsideHalf), axis=0)
        print('insideHalf:{},outsideHalf:{}'.format(np.shape(insideHalf), np.shape(outsideHalf)))

        insideHalf = imIn[:, :int(image_width / 2), :]
        outsideHalf = imOut[:, int(image_width / 2):, :]
        horizontally_splitted = np.concatenate((insideHalf, outsideHalf), axis=1)
        print('insideHalf:{},outsideHalf:{}'.format(np.shape(insideHalf), np.shape(outsideHalf)))

        cv2.imshow('vertically_splitted ', vertically_splitted)
        cv2.imshow('horizontally_splitted ', horizontally_splitted)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def calibrationReport(self, K=None, old_style = False):
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
        calibration_results_inside = pd.DataFrame(
            {"params": ['fx', 'fy', 'px', 'py', 'sk', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6',
                        's1', 's2', 's3', 's4', 'tx', 'ty', 'Error']})
        calibration_results_outside = pd.DataFrame(
            {"params": ['fx', 'fy', 'px', 'py', 'sk', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6',
                        's1', 's2', 's3', 's4', 'tx', 'ty', 'Error']})
        rms_all_outside = ['Error']
        rms_all_inside = ['Error']

        min_error, flag = 10000000,0
        for key in Distorsion_models:
            print()
            print(key, '->', Distorsion_models[key][0], ' , ', Distorsion_models[key][1], ' , ',
                  Distorsion_models[key][2])
            flags = Distorsion_models[key][1]

            self.calibrate(flags=flags, project=False)

            s = np.array([self.K_outside[0, 0], self.K_outside[1, 1], self.K_outside[0, 2], self.K_outside[1, 2],
                          self.K_outside[0, 1]])
            s = np.append(s, self.D_outside)
            calibration_results_outside[str(key)] = pd.Series(s)
            calibration_results_outside.fillna('---', inplace=True)
            rms_all_outside.append(self.rmsOut)
            calibration_results_outside[str(key)] = calibration_results_outside[str(key)].map(
                lambda x: round(x, 4) if isinstance(x, (int, float)) else x).astype(str)

            s = np.array([self.K_inside[0, 0], self.K_inside[1, 1], self.K_inside[0, 2], self.K_inside[1, 2], self.K_inside[0, 1]])
            s = np.append(s, self.D_inside)
            calibration_results_inside[str(key)] = pd.Series(s)
            calibration_results_inside.fillna('---', inplace=True)
            rms_all_inside.append(self.rmsIn)
            calibration_results_inside[str(key)] = calibration_results_inside[str(key)].map(
                lambda x: round(x, 4) if isinstance(x, (int, float)) else x).astype(str)

            self.K_outside = self.K_inside = self.D_outside = self.D_inside = None

        calibration_results_outside.iloc[-1, :] = rms_all_outside
        calibration_results_outside.iloc[-1, :] = calibration_results_outside.iloc[-1, :].map(
            lambda x: round(x, 4) if isinstance(x, (int, float)) else x)

        calibration_results_inside.iloc[-1, :] = rms_all_inside
        calibration_results_inside.iloc[-1, :] = calibration_results_inside.iloc[-1, :].map(
            lambda x: round(x, 4) if isinstance(x, (int, float)) else x)

        save_csv(obj=calibration_results_inside, name='Inside_correspondences_charuco')
        save_csv(obj=calibration_results_outside, name='Outside_correspondences_charuco')

        return flag

if __name__ == '__main__':
    images = '/home/eugeniu/Desktop/my_data/CameraCalibration/data/car_cam_data/correspondences'
    calibrator = InsideOutside_Calibrator(filepath=images, name='correspondences')

    calibrator.read_images()
    #calibrator.calibrationReport()
    flags = 0
    #calibrator.calibrate(project=True, adjust = False, flags=flags)
    calibrator.readMonoData()
    #calibrator.test()
    calibrator.estimatePose()
    #calibrator.solve()





