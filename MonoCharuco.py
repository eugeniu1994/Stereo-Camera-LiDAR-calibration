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

from __future__ import print_function
import matplotlib.pyplot as plt
import mpl_toolkits
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from numpy import linspace
import numpy as np
import cv2
import math
import os
import glob
import pickle
from multiprocessing.dummy import Pool as ThreadPool
import pandas as pd
import cv2.aruco as aruco
from utils import *
from scipy.spatial.distance import cdist

np.set_printoptions(suppress=True)

class MonoCharuco_Calibrator(object):
    def __init__(self, name='', figsize=(12, 10)):
        self.name=name
        self.image_size = None  # Determined at runtime
        self.term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30000, 0.0000001)
        #self.term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.001)
        self.figure_size=(12,10)
        self.debug_dir = None
        self.figsize = figsize
        self.image_width= 1936
        self.image_height= 1216
        self.image_center = np.array([self.image_width/2, self.image_height/2])
        self.optical_area = (11.345, 7.126)  # mm

        self.K = None
        self.D = None

        self.fx = 0.45
        self.fy = 0.4
        self.see = True
        self.flipVertically = True
        self.stdDeviationsIntrinsics, self.stdDeviationsExtrinsics, self.perViewErrors = None,None,None

    def createCalibrationBoard(self, squaresY = 9, squaresX = 12, squareLength = .06, markerLength = 0.045, display=False):
        '''
        squaresX	number of chessboard squares in X direction
        squaresY	number of chessboard squares in Y direction
        squareLength	chessboard square side length (normally in meters)
        markerLength	marker side length (same unit than squareLength)
        '''
        self.ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_1000)
        #self.ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_4X4_1000)
        self.CHARUCO_BOARD = aruco.CharucoBoard_create(
            squaresX=squaresX, squaresY=squaresY,
            squareLength=squareLength,
            markerLength=markerLength,
            dictionary=self.ARUCO_DICT)

        self.pattern_columns = squaresX
        self.pattern_rows = squaresY
        self.distance_in_world_units = squareLength

        if display:
            imboard = self.CHARUCO_BOARD.draw((900, 700))
            cv2.imshow('CharucoBoard target', imboard)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def reprojection_error_plot(self, errors_, N, figure_size=(16, 12)):
        if len(errors_)>1:
            errInside = np.array(errors_[0]).squeeze()
            errOutside = np.array(errors_[1]).squeeze()
            data = []
            for i in range(N):
                data.append(['img_{}'.format(i), errInside[i], errOutside[i]])

            df = pd.DataFrame(data, columns=["Name", "Inside(px)", "Outside(px)"])

            ax = df.plot(x="Name", y=["Inside(px)", "Outside(px)"], kind="bar", figsize=figure_size, #grid=True,
                    title='Reprojection_error plot - {}'.format(name))

            avg_error_inside = np.sum(errInside) / len(errInside)
            y_mean_inside = [avg_error_inside] * N
            ax.plot(y_mean_inside, label='Mean Reprojection error inside:{}'.format(round(avg_error_inside,2)), linestyle='--')

            avg_error_outside = np.sum(errOutside) / len(errOutside)
            y_mean_outside = [avg_error_outside] * N
            ax.plot(y_mean_outside, label='Mean Reprojection error outside:{}'.format(round(avg_error_outside,2)), linestyle='--')

            ax.legend(loc='upper right')
            ax.set_xlabel("Image_names")
            ax.set_ylabel("Reprojection error in px")
            plt.show()

    def _calc_reprojection_error(self, figure_size=(16, 12), save_dir=None, limit=False):
        if self.perViewErrors is not None:
            print('Reproject train images')

            self.perViewErrors = np.array(self.perViewErrors).squeeze()
            avg_error = np.sum(np.array(self.perViewErrors)) / len(self.perViewErrors)
            print("The Mean Reprojection Error in pixels is:  {}".format(avg_error))
            x = ['img_{}'.format(i) for i,p in enumerate(self.calibration_df.image_names)]
            y_mean = [avg_error] * len(self.calibration_df.image_names)
            fig, ax = plt.subplots()
            fig.set_figwidth(figure_size[0])
            fig.set_figheight(figure_size[1])
            max_intensity = np.max(self.perViewErrors)
            cmap = cm.get_cmap('jet')

            colors = cmap(self.perViewErrors / max_intensity)  # * 255
            print('self.perViewErrors:{},  colors:{}, max_intensity:{}'.format(np.shape(self.perViewErrors), np.shape(colors), max_intensity))

            ax.scatter(x, self.perViewErrors, label='Reprojection error', c=colors, marker='o')  # plot before
            ax.bar(x, self.perViewErrors, color=colors, alpha=0.3)
            ax.plot(x, y_mean, label='Mean Reprojection error', linestyle='--')
            ax.legend(loc='upper right')
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)
            ax.set_title("{} - Reprojection_error plot, Mean:{}".format(self.name, round(avg_error, 2)))
            ax.set_xlabel("Image_names")
            ax.set_ylabel("Reprojection error in pixels")
            if limit:
                ax.set_ylim(0, self.rms)
            if save_dir:
                plt.savefig(os.path.join(save_dir, "reprojection_error.png"))

            plt.show()
        else:
            print('Reproject test images')
            limit = True
            reprojection_error = []
            for i in range(len(self.calibration_df)):
                imgpoints2, _ = cv2.projectPoints(self.calibration_df.obj_points[i], self.calibration_df.rvecs[i],
                                                  self.calibration_df.tvecs[i], self.K, self.D)
                temp_error = cv2.norm(self.calibration_df.img_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                reprojection_error.append(temp_error)

            self.calibration_df['reprojection_error'] = pd.Series(reprojection_error)
            avg_error = np.sum(np.array(reprojection_error)) / len(self.calibration_df.obj_points)
            x = [os.path.basename(p) for p in self.calibration_df.image_names]
            y_mean = [avg_error] * len(self.calibration_df.image_names)
            fig, ax = plt.subplots()
            fig.set_figwidth(figure_size[0])
            fig.set_figheight(figure_size[1])

            max_intensity = np.max(reprojection_error)
            cmap = cm.get_cmap('jet')
            colors = cmap(reprojection_error / max_intensity)  # * 255
            ax.scatter(x, reprojection_error, label='Reprojection error', c=colors, marker='o')  # plot before
            ax.bar(x, reprojection_error, color=colors, alpha=0.3)
            ax.plot(x, y_mean, label='Mean Reprojection error', linestyle='--')

            ax.legend(loc='upper right')
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)
            ax.set_title("{} - Reprojection_error plot, Mean:{}".format(self.name, round(avg_error, 2)))
            ax.set_xlabel("Image_names")
            ax.set_ylabel("Reprojection error in pixels")
            if limit:
                ax.set_ylim(0, self.rms)

            if save_dir:
                plt.savefig(os.path.join(save_dir, "reprojection_error.png"))

            plt.show()
            print("The Mean Reprojection Error in pixels is:  {}".format(avg_error))

    def read_images(self, images, threads=5, K=None,D=None,limit = 9):
        self.total = 0
        self.images=images
        print('There are {} images'.format(np.shape(images)))
        self.h, self.w = cv2.imread(images[0], 0).shape[:2]
        working_images, img_points, obj_points = [],[],[]
        images.sort()
        corners_all = []  # Corners discovered in all images processed - obj_points
        ids_all = []  # Aruco ids corresponding to corners discovered - img_points
        if K is not None and D is not None:  # undistort images before calibration
            self.calibration_df = None
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, D, (self.w, self.h), 1, (self.w, self.h))

        def undistort_image(img):
            img2 = img
            mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, newcameramtx, (self.w, self.h), 5)
            dst = cv2.remap(img2, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = roi
            dst = dst[y:y + h, x:x + w]
            #print('dst:{}'.format(np.shape(dst)))
            self.h, self.w = dst.shape[:2]
            return dst

        h, w = self.CHARUCO_BOARD.chessboardCorners.shape
        def process_single_image(img_path):
            img = cv2.imread(img_path)  # gray scale
            if img is None:
                print("Failed to load {}".format(img_path))
                return None
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized1 = img
            if K is not None and D is not None:  # undistort images before calibration
                img = undistort_image(img)

            if threads<=0:
                resized2 = img
                if self.flipVertically:
                    resized1 = cv2.flip(resized1, -1)
                    resized2 = cv2.flip(resized2, -1)

                cv2.imshow('original ',  cv2.resize(resized1, (0, 0), fx=self.fx, fy=self.fy))
                cv2.imshow('undistorted',  cv2.resize(resized2, (0, 0), fx=self.fx, fy=self.fy))
                cv2.waitKey(0)

            assert self.w == img.shape[1] and self.h == img.shape[0], "All the images must have same shape"
            corners, ids, rejected = aruco.detectMarkers(image=gray, dictionary=self.ARUCO_DICT)
            corners, ids, rejectedImgPoints, recoveredIds = aruco.refineDetectedMarkers(image=gray,board=self.CHARUCO_BOARD,
                                                                                          detectedCorners=corners,detectedIds=ids,
                                                                                          rejectedCorners=rejected,cameraMatrix=K,
                                                                                          distCoeffs=D)

            if len(corners) >= limit:
                response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                    markerCorners=corners, markerIds=ids,
                    image=gray, board=self.CHARUCO_BOARD)
                if response >= limit:
                    #objPts, imgPts = aruco.getBoardObjectAndImagePoints(self.CHARUCO_BOARD, charuco_corners, charuco_ids)
                    imgPts = np.array(charuco_corners)
                    objPts = self.CHARUCO_BOARD.chessboardCorners.reshape((h, 1, 3))[np.asarray(charuco_ids).squeeze()] * self.distance_in_world_units
                    self.total += len(imgPts)
                    if self.see:
                        #img1 = aruco.drawDetectedMarkers(image=img, corners=corners)
                        #img1 = cv2.drawChessboardCorners(img, (11, 8), charuco_corners, True)
                        img1 = img.copy()
                        for i in imgPts:
                            (x,y) = i.ravel()
                            cv2.circle(img1, (x,y),3,(0,0,255),3)
                        img2 = aruco.drawDetectedCornersCharuco(image=img.copy(), charucoCorners=charuco_corners,charucoIds=charuco_ids)

                        if self.flipVertically:
                            img1 = cv2.flip(img1, -1)
                            img2 = cv2.flip(img2, -1)

                        img1 = cv2.resize(img1, (0, 0), fx=self.fx, fy=self.fy)
                        img2 = cv2.resize(img2, (0, 0), fx=self.fx, fy=self.fy)

                        _horizontal = np.vstack((img1, img2))
                        cv2.imshow('Images', _horizontal)
                        k = cv2.waitKey(0)
                        if k % 256 == 32:  # pressed space
                            self.see = False
                            cv2.destroyAllWindows()

                    if not self.image_size:
                        self.image_size = gray.shape[::-1]

                    return (img_path, charuco_ids, charuco_corners, objPts, imgPts)
                else:
                    # print("Calibration board NOT FOUND")
                    return (None)
            # print("Calibration board NOT FOUND")
            return (None)

        threads_num = int(threads)
        if threads_num <= 1:
            calibrationBoards = [process_single_image(img_path) for img_path in images]
        else:
            print("Running with %d threads..." % threads_num)
            self.see = False
            pool = ThreadPool(threads_num)
            calibrationBoards = pool.map(process_single_image, images)

        calibrationBoards = [x for x in calibrationBoards if x is not None]
        for (img_path, corners, pattern_points, objPts, imgPts) in calibrationBoards:
            working_images.append(img_path)
            ids_all.append(corners)
            corners_all.append(pattern_points)
            img_points.append(imgPts)
            obj_points.append(objPts)

        self.calibration_df = pd.DataFrame({"image_names": working_images,
                                            "ids_all": ids_all,
                                            "corners_all": corners_all,
                                            "img_points": img_points,
                                            "obj_points": obj_points,
                                            })
        self.calibration_df.sort_values("image_names")
        self.calibration_df = self.calibration_df.reset_index(drop=True)
        cv2.destroyAllWindows()
        print('Total datapoints:{}'.format(self.total))
        print('start calibration corners_all:{},  ids_all:{}'.format(np.shape(np.array(self.calibration_df.corners_all).squeeze()),
                                                                       np.shape(np.array(self.calibration_df.ids_all).ravel())))

    def calibrate(self, flags=0, project=True, K=None, D=None, save=False, extended = False, old_style = False):
        self.K = K
        self.D = D
        if self.K is not None:
            print('Use fixed K - estimate only distortion')
            flags |= cv2.CALIB_FIX_INTRINSIC
            flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
            flags |= cv2.CALIB_USE_INTRINSIC_GUESS
            flags |= cv2.CALIB_FIX_FOCAL_LENGTH


        charucoCorners = np.array(self.calibration_df.corners_all)
        charucoIds = np.array(self.calibration_df.ids_all)
        if extended:
            print('Extended version - calibration')
            self.rms, self.K, self.D, self.rvecs, self.tvecs, \
            self.stdDeviationsIntrinsics, self.stdDeviationsExtrinsics, self.perViewErrors = aruco.calibrateCameraCharucoExtended(
                charucoCorners=charucoCorners, charucoIds=charucoIds,
                board=self.CHARUCO_BOARD, imageSize=self.image_size,
                cameraMatrix=K, distCoeffs=None,
                flags=flags, criteria=self.term_criteria)
        else:
            if old_style:
                objectPoints = np.array(self.calibration_df.obj_points)
                imagePoints = np.array(self.calibration_df.img_points)
                self.rms, self.K, self.D, self.rvecs, self.tvecs = cv2.calibrateCamera(
                    objectPoints=objectPoints,
                    imagePoints=imagePoints,
                    imageSize=self.image_size,
                    cameraMatrix=K, distCoeffs=None,
                    flags=flags, criteria=self.term_criteria)
            else:
                self.rms, self.K, self.D, self.rvecs, self.tvecs = aruco.calibrateCameraCharuco(
                    charucoCorners=charucoCorners, charucoIds=charucoIds,
                    board=self.CHARUCO_BOARD, imageSize=self.image_size,
                    cameraMatrix=K, distCoeffs=None,
                    flags=flags, criteria = self.term_criteria)

        self.calibration_df['rvecs'] = pd.Series(self.rvecs)
        self.calibration_df['tvecs'] = pd.Series(self.tvecs)

        print("\nRMS:", self.rms)
        print("camera matrix:\n", self.K)
        print("distortion coefficients: ", self.D.ravel())
        if project:
            self._calc_reprojection_error(figure_size=self.figsize)

        result_dictionary = {
            "rms": self.rms,
            "K": self.K,
            "D": self.D,
        }
        if save:
            save_obj(obj=result_dictionary, name=self.name)

        return result_dictionary

    def calibrationReport(self, K=None, old_style = False):
        if K is None:
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
        else:
            Distorsion_models = {'ST': ['Standard', 0, 'Standard'],
                                 'RAT': ['Rational', cv2.CALIB_RATIONAL_MODEL, 'CALIB_RATIONAL_MODEL'],
                                 'THP': ['Thin Prism', cv2.CALIB_THIN_PRISM_MODEL, 'CALIB_THIN_PRISM_MODEL'],
                                 'TIL': ['Tilded', cv2.CALIB_TILTED_MODEL, 'CALIB_TILTED_MODEL'],  # }
                                 'RAT+THP': ['Rational+Thin Prism',
                                             cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_THIN_PRISM_MODEL,
                                             'CALIB_RATIONAL_MODEL + CALIB_THIN_PRISM_MODEL']
                                 }
        calibration_results = pd.DataFrame(
            {"params": ['fx', 'fy', 'px', 'py', 'sk', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6',
                        's1', 's2', 's3', 's4', 'tx', 'ty', 'Error']})
        rms_all = ['Error']

        min_error, flag = 10000000,0
        for key in Distorsion_models:
            print()
            print(key, '->', Distorsion_models[key][0], ' , ', Distorsion_models[key][1], ' , ',
                  Distorsion_models[key][2])
            flags = Distorsion_models[key][1]

            self.calibrate(flags=flags, project=False, K=K, old_style = old_style)
            s = np.array([self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2], self.K[0, 1]])
            s = np.append(s, self.D)
            calibration_results[str(key)] = pd.Series(s)
            calibration_results.fillna('---', inplace=True)
            rms_all.append(self.rms)
            calibration_results[str(key)] = calibration_results[str(key)].map(
                lambda x: round(x, 4) if isinstance(x, (int, float)) else x).astype(str)

            #if self.rms < min_error:
            #    min_error=self.rms
            #    flag = flags

        calibration_results.iloc[-1, :] = rms_all
        calibration_results.iloc[-1, :] = calibration_results.iloc[-1, :].map(
            lambda x: round(x, 4) if isinstance(x, (int, float)) else x)

        save_csv(obj=calibration_results, name=self.name+"_givenK" if K is not None else self.name)
        return flag

    def visualize_calibration_boards(self,cam_width=2,cam_height=1,scale_focal=4,scale = .05):
        # Plot the camera centric view
        self.visualize_views(
                        board_width=self.pattern_columns,
                        board_height=self.pattern_rows,
                        square_size=self.distance_in_world_units,
                        cam_width=cam_width*scale,
                        cam_height=cam_height*scale,
                        scale_focal=scale_focal*scale,
                        patternCentric=False,
                        )

        # Plot the pattern centric view
        self.visualize_views(
                        board_width=self.pattern_columns,
                        board_height=self.pattern_rows,
                        square_size=self.distance_in_world_units,
                        cam_width=cam_width*scale,
                        cam_height=cam_height*scale,
                        scale_focal=scale_focal*scale,
                        patternCentric=True,
                        )

    def visualize_views(self,board_width,board_height,square_size,cam_width=32,cam_height=16,scale_focal=25,patternCentric=False,Animate=False):

        def _inverse_homogeneoux_matrix(M):
            # util_function
            R = M[0:3, 0:3]
            T = M[0:3, 3]
            M_inv = np.identity(4)
            M_inv[0:3, 0:3] = R.T
            M_inv[0:3, 3] = -(R.T).dot(T)

            return M_inv

        def _transform_to_matplotlib_frame(cMo, X, inverse=False):
            # util function
            M = np.identity(4)
            M[1, 1] = 0
            M[1, 2] = 1
            M[2, 1] = -1
            M[2, 2] = 0

            if inverse:
                return M.dot(_inverse_homogeneoux_matrix(cMo).dot(X))
            else:
                return M.dot(cMo.dot(X))

        def _create_camera_model(camera_matrix, width, height, scale_focal, draw_frame_axis=False):
            # util function
            fx = camera_matrix[0, 0]
            fy = camera_matrix[1, 1]
            focal = 2 / (fx + fy)
            f_scale = scale_focal * focal

            # draw image plane
            X_img_plane = np.ones((4, 5))
            X_img_plane[0:3, 0] = [-width, height, f_scale]
            X_img_plane[0:3, 1] = [width, height, f_scale]
            X_img_plane[0:3, 2] = [width, -height, f_scale]
            X_img_plane[0:3, 3] = [-width, -height, f_scale]
            X_img_plane[0:3, 4] = [-width, height, f_scale]

            # draw triangle above the image plane
            X_triangle = np.ones((4, 3))
            X_triangle[0:3, 0] = [-width, -height, f_scale]
            X_triangle[0:3, 1] = [0, -2 * height, f_scale]
            X_triangle[0:3, 2] = [width, -height, f_scale]

            # draw camera
            X_center1 = np.ones((4, 2))
            X_center1[0:3, 0] = [0, 0, 0]
            X_center1[0:3, 1] = [-width, height, f_scale]

            X_center2 = np.ones((4, 2))
            X_center2[0:3, 0] = [0, 0, 0]
            X_center2[0:3, 1] = [width, height, f_scale]

            X_center3 = np.ones((4, 2))
            X_center3[0:3, 0] = [0, 0, 0]
            X_center3[0:3, 1] = [width, -height, f_scale]

            X_center4 = np.ones((4, 2))
            X_center4[0:3, 0] = [0, 0, 0]
            X_center4[0:3, 1] = [-width, -height, f_scale]

            # draw camera frame axis
            X_frame1 = np.ones((4, 2))
            X_frame1[0:3, 0] = [0, 0, 0]
            X_frame1[0:3, 1] = [f_scale / 2, 0, 0]

            X_frame2 = np.ones((4, 2))
            X_frame2[0:3, 0] = [0, 0, 0]
            X_frame2[0:3, 1] = [0, f_scale / 2, 0]

            X_frame3 = np.ones((4, 2))
            X_frame3[0:3, 0] = [0, 0, 0]
            X_frame3[0:3, 1] = [0, 0, f_scale / 2]

            if draw_frame_axis:
                return [X_img_plane, X_triangle, X_center1, X_center2, X_center3, X_center4, X_frame1, X_frame2,
                        X_frame3]
            else:
                return [X_img_plane, X_triangle, X_center1, X_center2, X_center3, X_center4]

        def _create_board_model(extrinsics, board_width, board_height, square_size, draw_frame_axis=False):
            # util function
            width = board_width * square_size
            height = board_height * square_size

            # draw calibration board
            X_board = np.ones((4, 5))
            # X_board_cam = np.ones((extrinsics.shape[0],4,5))
            X_board[0:3, 0] = [0, 0, 0]
            X_board[0:3, 1] = [width, 0, 0]
            X_board[0:3, 2] = [width, height, 0]
            X_board[0:3, 3] = [0, height, 0]
            X_board[0:3, 4] = [0, 0, 0]

            # draw board frame axis
            X_frame1 = np.ones((4, 2))
            X_frame1[0:3, 0] = [0, 0, 0]
            X_frame1[0:3, 1] = [height / 2, 0, 0]

            X_frame2 = np.ones((4, 2))
            X_frame2[0:3, 0] = [0, 0, 0]
            X_frame2[0:3, 1] = [0, height / 2, 0]

            X_frame3 = np.ones((4, 2))
            X_frame3[0:3, 0] = [0, 0, 0]
            X_frame3[0:3, 1] = [0, 0, height / 2]

            if draw_frame_axis:
                return [X_board, X_frame1, X_frame2, X_frame3]
            else:
                return [X_board]


        def _draw_camera_boards(ax, camera_matrix, cam_width, cam_height, scale_focal,
                                extrinsics, board_width, board_height, square_size,
                                patternCentric):
            min_values = np.zeros((3, 1))
            min_values = np.inf
            max_values = np.zeros((3, 1))
            max_values = -np.inf

            if patternCentric:
                X_moving = _create_camera_model(camera_matrix, cam_width, cam_height, scale_focal)
                X_static = _create_board_model(extrinsics, board_width, board_height, square_size)
            else:
                X_static = _create_camera_model(camera_matrix, cam_width, cam_height, scale_focal, True)
                X_moving = _create_board_model(extrinsics, board_width, board_height, square_size)

            cm_subsection = np.linspace(0.0, 1.0, extrinsics.shape[0])
            colors = [cm.jet(x) for x in cm_subsection]

            for i in range(len(X_static)):
                X = np.zeros(X_static[i].shape)
                for j in range(X_static[i].shape[1]):
                    X[:, j] = _transform_to_matplotlib_frame(np.eye(4), X_static[i][:, j])
                ax.plot3D(X[0, :], X[1, :], X[2, :], color='r')
                min_values = np.minimum(min_values, X[0:3, :].min(1))
                max_values = np.maximum(max_values, X[0:3, :].max(1))

            for idx in range(extrinsics.shape[0]):
                R, _ = cv2.Rodrigues(extrinsics[idx, 0:3])
                cMo = np.eye(4, 4)
                cMo[0:3, 0:3] = R
                cMo[0:3, 3] = extrinsics[idx, 3:6]
                for i in range(len(X_moving)):
                    X = np.zeros(X_moving[i].shape)
                    for j in range(X_moving[i].shape[1]):
                        X[0:4, j] = _transform_to_matplotlib_frame(cMo, X_moving[i][0:4, j], patternCentric)
                    ax.plot3D(X[0, :], X[1, :], X[2, :], color=colors[idx])
                    min_values = np.minimum(min_values, X[0:3, :].min(1))
                    max_values = np.maximum(max_values, X[0:3, :].max(1))

            return min_values, max_values


        i = 0
        extrinsics = np.zeros((len(self.rvecs), 6))
        for rot, trans in zip(self.rvecs, self.tvecs):
            extrinsics[i] = np.append(rot.flatten(), trans.flatten())
            i += 1
        # The extrinsics  matrix is of shape (N,6) (No default)
        # Where N is the number of board patterns
        # the first 3  columns are rotational vectors
        # the last 3 columns are translational vectors

        fig = plt.figure(figsize=self.figure_size)
        ax = fig.gca(projection='3d')

        ax.set_aspect("auto")
        #ax.set_aspect("equal")

        min_values, max_values = _draw_camera_boards(ax, self.K, cam_width, cam_height,
                                                     scale_focal, extrinsics, board_width,
                                                     board_height, square_size, patternCentric)

        X_min = min_values[0]
        X_max = max_values[0]
        Y_min = min_values[1]
        Y_max = max_values[1]
        Z_min = min_values[2]
        Z_max = max_values[2]
        max_range = np.array([X_max - X_min, Y_max - Y_min, Z_max - Z_min]).max() / 2.0

        mid_x = (X_max + X_min) * 0.5
        mid_y = (Y_max + Y_min) * 0.5
        mid_z = (Z_max + Z_min) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('-y')
        if patternCentric:
            ax.set_title('Pattern Centric View')
            if self.debug_dir:
                plt.savefig(os.path.join(self.debug_dir, "pattern_centric_view.png"))
        else:
            ax.set_title('Camera Centric View')
            if self.debug_dir:
                plt.savefig(os.path.join(self.debug_dir, "camera_centric_view.png"))
        plt.show()

    def doStuff(self, images, project=False, single_flag=True, K=None, extended = False):
        self.createCalibrationBoard()
        self.read_images(images = images, threads=1)
        if single_flag:
            flags = 0
        else:
            flags = self.calibrationReport()  # report with original data
        self.calibrate(flags=flags, project=project, K=K, save=False, extended = extended)
        #self.visualize_calibration_boards()

    def adjustCalibration(self, K, D):
        print('adjustCalibration -> estimate Windshield distortion -> undistort images -> calibrate again')
        K_outside = K                         #assume the outside K matrix is ideal
        D_outside = D                         #take the outside camera distortion
        self.doStuff(images=self.images, project=False, single_flag=True, K=K_outside) #calibration with given K -> estimate only distortion params
        D_inside_full = self.D                #Estimated distortion = lens D + Windshield D

        Windshield_distortion = np.abs(D_inside_full - D_outside) * np.sign(D_outside)
        print('Windshield_distortion')
        print(Windshield_distortion)
        #read images and undistort them
        self.read_images(images=self.images, K=K_outside, D=Windshield_distortion)
        self.name = 'adjusted_'+self.name
        flags = self.calibrationReport()  # report with original data

        #for the final calibration fix aspect ration
        flags |= cv2.CALIB_FIX_ASPECT_RATIO
        self.calibrate(flags=flags, project=True, save=True)

if __name__ == '__main__':
    images = glob.glob(
        '/home/eugeniu/Desktop/my_data/CameraCalibration/data/car_cam_data/charuco/outside/Left/*.png')
    name = 'charuco_outside_left'
    calibrator = MonoCharuco_Calibrator(name=name,figsize=(16, 14))
    calibrator.createCalibrationBoard()
    calibrator.read_images(images=images, threads=1)
    calibrator.calibrate(extended=False, project=True, old_style=True)





