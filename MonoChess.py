
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
from multiprocessing.dummy import Pool as ThreadPool
import glob
import pickle
import pandas as pd
from utils import *

np.set_printoptions(suppress=True)

class MonoChess_Calibrator:
    def __init__(self, pattern_type, pattern_rows, pattern_columns, distance_in_world_units=1.0,figsize=(12, 10), debug_dir=None,
                 term_criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 1000, 0.0001)):

        pattern_types = ["chessboard", "symmetric_circles", "asymmetric_circles", "custom"]
        assert pattern_type in pattern_types, "pattern type must be one of {}".format(pattern_types)

        self.pattern_type = pattern_type
        self.pattern_rows = pattern_rows
        self.pattern_columns = pattern_columns
        self.distance_in_world_units = distance_in_world_units

        self.figsize = figsize
        self.debug_dir = debug_dir
        self.term_criteria = term_criteria
        self.subpixel_refinement = True  # turn on or off subpixel refinement
        if self.pattern_type in ["asymmetric_circles", "symmetric_circles"]:
            self.subpixel_refinement = False
            self.use_clustering = True
            self.blobParams = cv2.SimpleBlobDetector_Params()
            self.blobParams.minThreshold = 8
            self.blobParams.maxThreshold = 255
            self.blobParams.filterByArea = True
            self.blobParams.minArea = 50  # minArea may be adjusted to suit for your experiment
            self.blobParams.maxArea = 10e5  # maxArea may be adjusted to suit for your experiment
            self.blobParams.filterByCircularity = True
            self.blobParams.minCircularity = 0.8
            self.blobParams.filterByConvexity = True
            self.blobParams.minConvexity = 0.87
            self.blobParams.filterByInertia = True
            self.blobParams.minInertiaRatio = 0.01
        if self.pattern_type == "asymmetric_circles":
            self.double_count_in_column = True
        if self.debug_dir and not os.path.isdir(self.debug_dir):
            os.mkdir(self.debug_dir)

        self.name = None
        self.image_width = 1936
        self.image_height = 1216
        self.image_center = np.array([self.image_width / 2, self.image_height / 2])
        self.optical_area = (11.345, 7.126)  # mm
        self.see = True

    @staticmethod
    def _splitfn(fn):
        path, fn = os.path.split(fn)
        name, ext = os.path.splitext(fn)
        return path, name, ext

    def _symmetric_world_points(self):
        x, y = np.meshgrid(range(self.pattern_columns), range(self.pattern_rows))
        prod = self.pattern_rows * self.pattern_columns
        pattern_points = np.hstack((x.reshape(prod, 1), y.reshape(prod, 1), np.zeros((prod, 1)))).astype(np.float32)
        return (pattern_points)

    def _asymmetric_world_points(self):
        pattern_points = []
        if self.double_count_in_column:
            for i in range(self.pattern_rows):
                for j in range(self.pattern_columns):
                    x = j / 2
                    if j % 2 == 0:
                        y = i
                    else:
                        y = i + 0.5
                    pattern_points.append((x, y))
        else:
            for i in range(self.pattern_rows):
                for j in range(self.pattern_columns):
                    y = i / 2
                    if i % 2 == 0:
                        x = j
                    else:
                        x = j + 0.5

                    pattern_points.append((x, y))

        pattern_points = np.hstack((pattern_points, np.zeros((self.pattern_rows * self.pattern_columns, 1)))).astype(
            np.float32)
        return (pattern_points)

    def _chessboard_image_points(self, img):
        found, corners = cv2.findChessboardCorners(img, (self.pattern_columns, self.pattern_rows))
        return (found, corners)

    def _circulargrid_image_points(self, img, flags, blobDetector):
        found, corners = cv2.findCirclesGrid(img, (self.pattern_columns, self.pattern_rows),
                                             flags=flags, blobDetector=blobDetector)
        return (found, corners)

    def _calc_reprojection_error(self, figure_size=(12, 8), save_dir=None, limit=True):
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

    def read_images(self, images_path_list, threads=4, custom_world_points_function=None,
                         custom_image_points_function=None, K=None,D=None):
        self.images = images_path_list
        if self.pattern_type == "custom":
            assert custom_world_points_function is not None, "Must implement a custom_world_points_function for 'custom' pattern "
            assert custom_image_points_function is not None, "Must implement a custom_image_points_function for 'custom' pattern"

        img_points = []
        obj_points = []
        working_images = []
        images_path_list.sort()
        print("There are {} {} images given for calibration".format(len(images_path_list), self.pattern_type))

        if self.pattern_type == "chessboard":
            pattern_points = self._symmetric_world_points() * self.distance_in_world_units

        elif self.pattern_type == "symmetric_circles":
            pattern_points = self._symmetric_world_points() * self.distance_in_world_units
            blobDetector = cv2.SimpleBlobDetector_create(self.blobParams)
            flags = cv2.CALIB_CB_SYMMETRIC_GRID
            if self.use_clustering:
                flags = cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING

        elif self.pattern_type == "asymmetric_circles":
            pattern_points = self._asymmetric_world_points() * self.distance_in_world_units
            blobDetector = cv2.SimpleBlobDetector_create(self.blobParams)
            flags = cv2.CALIB_CB_ASYMMETRIC_GRID
            if self.use_clustering:
                flags = cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING

        elif self.pattern_type == "custom":
            pattern_points = custom_world_points_function(self.pattern_rows, self.pattern_columns)

        self.h, self.w = cv2.imread(images_path_list[0], 0).shape[:2]
        if K is not None and D is not None: #undistort images before calibration
            self.calibration_df = None
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, D, (self.w, self.h), 1, (self.w, self.h))

        def undistort_image(img):
            img2 = img
            # undistort
            mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, newcameramtx, (self.w, self.h), 5)
            dst = cv2.remap(img2, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = roi
            dst = dst[y:y + h, x:x + w]
            #print('dst:{}'.format(np.shape(dst)))
            self.h, self.w = dst.shape[:2]
            return dst

        def process_single_image(img_path):
            img = cv2.imread(img_path, 0)  # gray scale
            if img is None:
                print("Failed to load {}".format(img_path))
                return None

            resized1 = img
            if K is not None and D is not None:  # undistort images before calibration
                img = undistort_image(img)

            if threads<=0:
                resized2 = img
                if True:
                    resized1 = cv2.flip(resized1, -1)
                    resized2 = cv2.flip(resized2, -1)

                cv2.imshow('original ', resized1)
                cv2.imshow('undistorted', resized2)
                cv2.waitKey(0)

            assert self.w == img.shape[1] and self.h == img.shape[0], "All the images must have same shape"

            if self.pattern_type == "chessboard":
                found, corners = self._chessboard_image_points(img)
            elif self.pattern_type == "asymmetric_circles" or self.pattern_type == "symmetric_circles":
                found, corners = self._circulargrid_image_points(img, flags, blobDetector)

            elif self.pattern_type == "custom":
                found, corners = custom_image_points_function(img, self.pattern_rows, self.pattern_columns)
                assert corners[0] == pattern_points[
                    0], "custom_image_points_function should return a numpy array of length matching the number of control points in the image"

            if found:
                if self.subpixel_refinement:
                    corners2 = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), self.term_criteria)
                else:
                    corners2 = corners.copy()

                if self.debug_dir or self.see:
                    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    cv2.drawChessboardCorners(vis, (self.pattern_columns, self.pattern_rows), corners2, found)
                    #path, name, ext = self._splitfn(img_path)
                    #outfile = os.path.join(self.debug_dir, name + '_pts_vis.png')
                    #cv2.imwrite(outfile, vis)
            else:
                # print("Calibration board NOT FOUND")
                return (None)

            return (img_path, corners2, pattern_points)

        threads_num = int(threads)
        if threads_num <= 1:
            calibrationBoards = [process_single_image(img_path) for img_path in images_path_list]
        else:
            print("Running with %d threads..." % threads_num)
            pool = ThreadPool(threads_num)
            calibrationBoards = pool.map(process_single_image, images_path_list)

        calibrationBoards = [x for x in calibrationBoards if x is not None]
        for (img_path, corners, pattern_points) in calibrationBoards:
            working_images.append(img_path)
            img_points.append(corners)
            obj_points.append(pattern_points)

        self.calibration_df = pd.DataFrame({"image_names": working_images,
                                            "img_points": img_points,
                                            "obj_points": obj_points, })
        self.calibration_df.sort_values("image_names")
        self.calibration_df = self.calibration_df.reset_index(drop=True)
        print('start calibration obj_points:{},  img_points:{}'.format(np.shape(self.calibration_df.obj_points),
                                                                       np.shape(self.calibration_df.img_points)))

    def calibrate(self, flags=0, project=True, K=None,D=None, save=False):
        self.K = K
        self.D = D
        if self.K is not None:
            print('Use fixed K - estimate only distortion')
            flags |= cv2.CALIB_FIX_INTRINSIC
            flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
            flags |= cv2.CALIB_USE_INTRINSIC_GUESS
            flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        elif self.D is not None:
            print('Use fixed D - estimate only K matrix')
            flags |= cv2.CALIB_FIX_K1
            flags |= cv2.CALIB_FIX_K2
            flags |= cv2.CALIB_FIX_K3
            flags |= cv2.CALIB_FIX_TANGENT_DIST

        self.rms, self.K, self.D, \
        rvecs, tvecs = cv2.calibrateCamera(objectPoints=self.calibration_df.obj_points,
                                           imagePoints=self.calibration_df.img_points,
                                           imageSize=(self.w, self.h), cameraMatrix=K, distCoeffs=D,
                                           criteria=self.term_criteria,
                                           flags=flags)

        self.calibration_df['rvecs'] = pd.Series(rvecs)
        self.calibration_df['tvecs'] = pd.Series(tvecs)

        print("\nRMS:", self.rms)
        print("camera matrix:\n", self.K)
        print("distortion coefficients: ", self.D.ravel())
        if project:
            self._calc_reprojection_error(figure_size=self.figsize, save_dir=self.debug_dir)

        result_dictionary = {
            "rms": self.rms,
            "K": self.K,
            "D": self.D,
        }
        if save:
            save_obj(obj=result_dictionary, name=self.name)

        return result_dictionary

    def calibrationReport(self, K=None):
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

            self.calibrate(flags=flags, project=False, K=K)
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

    def visualize_calibration_boards(self, cam_width=20.0, cam_height=10.0, scale_focal=40):
        visualize_views(camera_matrix=self.K,
                        rvecs=self.calibration_df.rvecs,
                        tvecs=self.calibration_df.tvecs,
                        board_width=self.pattern_columns,
                        board_height=self.pattern_rows,
                        square_size=self.distance_in_world_units,
                        cam_width=cam_width,
                        cam_height=cam_height,
                        scale_focal=scale_focal,
                        patternCentric=False,
                        figsize=self.figsize,
                        save_dir=self.debug_dir
                        )
        visualize_views(camera_matrix=self.K,
                        rvecs=self.calibration_df.rvecs,
                        tvecs=self.calibration_df.tvecs,
                        board_width=self.pattern_columns,
                        board_height=self.pattern_rows,
                        square_size=self.distance_in_world_units,
                        cam_width=cam_width,
                        cam_height=cam_height,
                        scale_focal=scale_focal,
                        patternCentric=True,
                        figsize=self.figsize,
                        save_dir=self.debug_dir
                        )
        plt.show()

    def doStuff(self, images, project = True, single_flag = True, K = None,D=None):
        self.read_images(images_path_list=images)
        if single_flag:
            flags=0
        else:
            flags = self.calibrationReport() #report with original data
        self.calibrate(flags=flags, project=project, K=K, D=D, save=True)

    def adjustCalibration(self,images,  K, D):
        print('adjustCalibration -> estimate Windshield distortion -> undistort images -> calibrate again')
        K_outside = K                         #assume the outside K matrix is ideal
        D_outside = D                         #take the outside camera distortion
        self.doStuff(images=images, single_flag=True, K=K_outside) #calibration with given K -> estimate only distortion params
        D_inside_full = self.D                #Estimated distortion = lens D + Windshield D

        Windshield_distortion = np.abs(D_inside_full - D_outside) * np.sign(D_outside)
        print('Windshield_distortion')
        print(Windshield_distortion)

        #read images and undistort them
        self.read_images(images_path_list=self.images, K=K_outside, D=Windshield_distortion)
        self.name = 'adjusted_'+self.name
        flags = self.calibrationReport()  # report with original data

        # for the final calibration fix aspect ration
        flags |= cv2.CALIB_FIX_ASPECT_RATIO
        self.calibrate(flags=flags, project=True)




