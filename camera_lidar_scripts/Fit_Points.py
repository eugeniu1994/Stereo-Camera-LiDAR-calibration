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
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
try:
    import pcl
    from pyquaternion import Quaternion
except:
    print('cannot import pcl -> change python version')
import matplotlib.cm as cmx
from scipy.spatial import distance_matrix
from scipy.optimize import leastsq
import matplotlib
import matplotlib.animation as animation
import open3d as o3d
import glob
import cv2
import cv2.aruco as aruco

import os
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation
import pickle
from matplotlib.lines import Line2D
import pandas as pd
import random
from scipy.spatial import ConvexHull
from math import sqrt
from math import atan2, cos, sin, pi
from collections import namedtuple
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d

np.set_printoptions(suppress=True)

global globalTrigger
globalTrigger = True

stereoRectify = False# True

class Annotation3D(Annotation):
    def __init__(self, s, xyz, *args, **kwargs):
        Annotation.__init__(self, s, xy=(0, 0), *args, **kwargs)
        self._verts3d = xyz

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.xy = (xs, ys)
        Annotation.draw(self, renderer)

def save_obj(obj, name):
    with open('/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/data/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, protocol=2)
    print('{}.pkl Object saved'.format(name))

def load_obj(name):
    with open('/home/eugeniu/Desktop/my_data/CameraCalibration/data/saved_files/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def showErros(_3DErros, IMageNames):
    print('len(_3DErros)->{}'.format(np.shape(_3DErros)))
    if len(_3DErros)>1:
        _3DErros = np.array(_3DErros).squeeze()
        # norm_total = np.array(_3DErros[:,0]).squeeze()
        norm_axis = np.array(_3DErros).squeeze() * 1000

        index, bar_width = np.arange(len(IMageNames)), 0.24
        fig, ax = plt.subplots()
        X = ax.bar(index, norm_axis[:, 0], bar_width, label="X")
        Y = ax.bar(index + bar_width, norm_axis[:, 1], bar_width, label="Y")
        Z = ax.bar(index + bar_width + bar_width, norm_axis[:, 2], bar_width, label="Z")

        ax.set_xlabel('images')
        ax.set_ylabel('errors in mm')
        ax.set_title('3D error')
        ax.set_xticks(index + bar_width / 3)
        ax.set_xticklabels(IMageNames)
        ax.legend()

        plt.show()

def triangulation(kp1, kp2, T_1w, T_2w):
    """Triangulation to get 3D points
    Args:
        kp1 (Nx2): keypoint in view 1 (normalized)
        kp2 (Nx2): keypoints in view 2 (normalized)
        T_1w (4x4): pose of view 1 w.r.t  i.e. T_1w (from w to 1)
        T_2w (4x4): pose of view 2 w.r.t world, i.e. T_2w (from w to 2)
    Returns:
        X (3xN): 3D coordinates of the keypoints w.r.t world coordinate
        X1 (3xN): 3D coordinates of the keypoints w.r.t view1 coordinate
        X2 (3xN): 3D coordinates of the keypoints w.r.t view2 coordinate
    """
    kp1_3D = np.ones((3, kp1.shape[0]))
    kp2_3D = np.ones((3, kp2.shape[0]))
    kp1_3D[0], kp1_3D[1] = kp1[:, 0].copy(), kp1[:, 1].copy()
    kp2_3D[0], kp2_3D[1] = kp2[:, 0].copy(), kp2[:, 1].copy()
    X = cv2.triangulatePoints(T_1w[:3], T_2w[:3], kp1_3D[:2], kp2_3D[:2])
    X /= X[3]
    X1 = T_1w[:3].dot(X)
    X2 = T_2w[:3].dot(X)
    return X[:3].T, X1.T, X2.T

def triangulate(R1,R2,t1,t2,K1,K2,D1,D2, pts1, pts2):
    P1 = np.hstack([R1.T, -R1.T.dot(t1)])
    P2 = np.hstack([R2.T, -R2.T.dot(t2)])

    P1 = K1.dot(P1)
    P2 = K2.dot(P2)

    # Triangulate
    _3d_points = []
    for i,point in enumerate(pts1):
        point3D = cv2.triangulatePoints(P1, P2, pts1[i], pts2[i]).T
        point3D = point3D[:, :3] / point3D[:, 3:4]
        _3d_points.append(point3D)
    print('Triangulate _3d_points -> {}'.format(np.shape(_3d_points)))

    return np.array(_3d_points).squeeze()

def mai(R1,R2,t1,t2,imagePoint1,imagePoint2, K2=None,K1=None, D2=None,D1=None):
    # Set up two cameras near each other
    if K1 is None:
        K = np.array([
            [718.856, 0., 607.1928],
            [0., 718.856, 185.2157],
            [0., 0., 1.],
        ])

        R1 = np.array([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]
        ])

        R2 = np.array([
            [0.99999183, -0.00280829, -0.00290702],
            [0.0028008, 0.99999276, -0.00257697],
            [0.00291424, 0.00256881, 0.99999245]
        ])

        t1 = np.array([[0.], [0.], [0.]])

        t2 = np.array([[-0.02182627], [0.00733316], [0.99973488]])

        # Corresponding image points
        imagePoint1 = np.array([371.91915894, 221.53485107])
        imagePoint2 = np.array([368.26071167, 224.86262512])

    P1 = np.hstack([R1.T, -R1.T.dot(t1)])
    P2 = np.hstack([R2.T, -R2.T.dot(t2)])

    P1 = K1.dot(P1)
    P2 = K2.dot(P2)

    # Triangulate
    point3D = cv2.triangulatePoints(P1, P2, imagePoint1, imagePoint2).T
    point3D = point3D[:, :3] / point3D[:, 3:4]
    print('Triangulate point3D -> {}'.format(point3D))

    # Reproject back into the two cameras
    rvec1, _ = cv2.Rodrigues(R1.T)  # Change
    rvec2, _ = cv2.Rodrigues(R2.T)  # Change

    p1, _ = cv2.projectPoints(point3D, rvec1, -t1, K1, distCoeffs=D1)  # Change
    p2, _ = cv2.projectPoints(point3D, rvec2, -t2, K2, distCoeffs=D2)  # Change

    # measure difference between original image point and reporjected image point

    reprojection_error1 = np.linalg.norm(imagePoint1 - p1[0, :])
    reprojection_error2 = np.linalg.norm(imagePoint2 - p2[0, :])

    print('difference between original image point and reporjected image point')
    print(reprojection_error1, reprojection_error2)
    return p1,p2

class PointCloud_filter(object):
    def __init__(self, file, img_file=None, img_file2=None, debug=True):
        self.debug = debug
        self.img_file = img_file
        self.img_file2 = img_file2
        self.name = os.path.basename(file).split('.')[0]
        self.file = file
        self.useVoxel, self.voxel_size = False, 0.15
        self.lowerTemplate, self.showImage = False, True
        self.showError = False
        self.points_correspondences = None
        self.OK = False
        self.useInitialPointCloud = False #user all point to fit or only margins
        self.chessBoard = False
        self.applyICP_directly = False
        self.s = .1  # scale
        self.plotInit, self.axis_on, self.colour, self.Annotate = False, True, False, False
        self.chess, self.corn, self.p1, self.p2, self.p3, self.ICP_finetune_plot = None, None, None, None, None, None

        if self.showImage:
            b = 1
            self.pts = np.float32([[0, b, 0], [b, b, 0], [b, 0, 0], [-0.03, -0.03, 0]])

            self.ImageNames = []
            self._3DErros = []
            self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
            self.axis = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, -1]]).reshape(-1, 3)
            self.objp = np.zeros((7 * 10, 3), np.float32)
            self.objp[:, :2] = np.mgrid[0:10, 0:7].T.reshape(-1, 2) * self.s

            self.fig = plt.figure(figsize=plt.figaspect(0.5))
            self.fig.suptitle('Data collection', fontsize=16)

            self.ax = self.fig.add_subplot(1, 2, 1, projection='3d')
            #self.ax = self.fig.add_subplot(1, 2, 2, projection='3d')

            self.readCameraIntrin()
            self.QueryImg = cv2.imread(img_file)
            self.ImageNames.append(os.path.basename(img_file))
            if self.img_file2:  # use stereo case
                self.QueryImg2 = cv2.imread(img_file2)
                if stereoRectify:
                    self.QueryImg = cv2.remap(src=self.QueryImg, map1=self.leftMapX, map2=self.leftMapY,
                                              interpolation=cv2.INTER_LINEAR, dst=None, borderMode=cv2.BORDER_CONSTANT)
                    self.QueryImg2 = cv2.remap(src=self.QueryImg2, map1=self.rightMapX, map2=self.rightMapY,
                                               interpolation=cv2.INTER_LINEAR, dst=None, borderMode=cv2.BORDER_CONSTANT)

                gray_left = cv2.cvtColor(self.QueryImg, cv2.COLOR_BGR2GRAY)
                ret_left, corners_left = cv2.findChessboardCorners(gray_left, (10, 7), None)
                gray_right = cv2.cvtColor(self.QueryImg2, cv2.COLOR_BGR2GRAY)
                ret_right, corners_right = cv2.findChessboardCorners(gray_right, (10, 7), None)
                if ret_right and ret_left:
                    print('Found chessboard in both images')
                    self.chessBoard = True
                    corners2_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), self.criteria)
                    self.corners2 = corners2_left
                    cv2.drawChessboardCorners(self.QueryImg, (10, 7), self.corners2, ret_left)
                    ret, self.rvecs, self.tvecs = cv2.solvePnP(self.objp, self.corners2, self.K_left, self.D_left)
                    imgpts, jac = cv2.projectPoints(self.axis, self.rvecs, self.tvecs, self.K_left, self.D_left)
                    self.QueryImg = self.draw(self.QueryImg, corners=corners2_left, imgpts=imgpts)
                    self.pixelsPoints = np.asarray(corners2_left).squeeze()
                    self.pixels_left = np.asarray(corners2_left).squeeze()

                    corners2_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), self.criteria)
                    cv2.drawChessboardCorners(self.QueryImg2, (10, 7), corners2_right, ret_right)
                    self.pixels_right = np.asarray(corners2_right).squeeze()

                    self.baseline = abs(self.T[0])
                    #self.baseline = 1.07

                    print('baseline:{} m'.format(self.baseline))
                    self.focal_length, self.cx, self.cy = self.K[0, 0], self.K[0, 2], self.K[1, 2]
                    self.x_left, self.x_right = self.pixels_left, self.pixels_right
                    disparity = np.sum(np.sqrt((self.x_left - self.x_right) ** 2), axis=1)
                    # depth = baseline (meter) * focal length (pixel) / disparity-value (pixel) -> meter
                    self.depth = (self.baseline * self.focal_length / disparity)
                    print('depth:{}'.format(np.shape(self.depth)))
                    self.fxypxy = [self.K[0, 0], self.K[1, 1], self.cx, self.cy]

                    '''print('TRIANGULATE HERE==========================================')
                    P_1 = np.vstack((np.hstack((np.eye(3), np.zeros(3)[:, np.newaxis])), [0, 0, 0, 1]))  # left  camera
                    P_2 = np.vstack((np.hstack((self.R, self.T)), [0, 0, 0, 1]))  # right camera
                    print('P1_{}, P_2{}, x_left:{}, x_right:{}'.format(np.shape(P_1), np.shape(P_2),
                                                                       np.shape(self.x_left), np.shape(self.x_right)))

                    X_w, X1, X2 = triangulation(self.x_left,self.x_right,P_1,P_2)
                    print('X_w:{}, X1:{}, X2:{}, '.format(np.shape(X_w), np.shape(X1), np.shape(X2)))
                    print(X_w[0])
                    print(X1[0])
                    print(X2[0])'''


                    '''R1 = np.eye(3)
                    R2 = self.R
                    t1 = np.array([[0.], [0.], [0.]])
                    t2 = self.T

                    # Corresponding image points
                    imagePoint1 = np.array([371.91915894, 221.53485107])
                    imagePoint2 = np.array([368.26071167, 224.86262512])
                    imagePoint1 = self.x_left[0]
                    imagePoint2 = self.x_right[0]
                    print('imagePoint1:{}, imagePoint2:{}'.format(np.shape(imagePoint1), np.shape(imagePoint2)))

                    print('self.K_left ')
                    print(self.K_left)
                    print('self.K_right ')
                    print(self.K_right)

                    p1,p2 = test(R1,R2,t1,t2,imagePoint1,imagePoint2,K1=self.K_left,K2=self.K_right, D1=self.D_left,D2=self.D_right)
                    p1 = np.array(p1).squeeze().astype(int)
                    p2 = np.array(p2).squeeze().astype(int)
                    print('p1:{}, p2:{}'.format(np.shape(p1), np.shape(p2)))
                    #d2 = distance_matrix(X_w, X_w)
                    #print('d2:{}'.format(d2))
                    cv2.circle(self.QueryImg, (p1[0],p1[1]), 7, (255, 0, 0), 7)
                    cv2.circle(self.QueryImg2, (p2[0], p2[1]), 7, (255, 0, 0), 7)

                    cv2.imshow('QueryImg', cv2.resize(self.QueryImg,None,fx=.5,fy=.5))
                    cv2.imshow('QueryImg2', cv2.resize(self.QueryImg2, None, fx=.5, fy=.5))
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()'''

                else:
                    self.chessBoard = False
                    self.useVoxel = False
                    print('No chessboard ')
                    corners2_left, ids_left, rejectedImgPoints = aruco.detectMarkers(gray_left, self.ARUCO_DICT)
                    corners2_left, ids_left, _, _ = aruco.refineDetectedMarkers(image=gray_left,
                                                                                board=self.calibation_board,
                                                                                detectedCorners=corners2_left,
                                                                                detectedIds=ids_left,
                                                                                rejectedCorners=rejectedImgPoints,
                                                                                cameraMatrix=self.K_left,
                                                                                distCoeffs=self.D_left)

                    corners2_right, ids_right, rejectedImgPoints = aruco.detectMarkers(gray_right, self.ARUCO_DICT)
                    corners2_right, ids_right, _, _ = aruco.refineDetectedMarkers(image=gray_right,
                                                                                  board=self.calibation_board,
                                                                                  detectedCorners=corners2_right,
                                                                                  detectedIds=ids_right,
                                                                                  rejectedCorners=rejectedImgPoints,
                                                                                  cameraMatrix=self.K_right,
                                                                                  distCoeffs=self.D_right)

                    if np.all(ids_left != None) and np.all(ids_right != None):
                        print('found charuco board, in both images')
                        retval_left, self.rvecs, self.tvecs = aruco.estimatePoseBoard(corners2_left, ids_left,
                                                                                      self.calibation_board,
                                                                                      self.K_left, self.D_left, None,
                                                                                      None)
                        retval_right, self.rvecs_right, self.tvecs_right = aruco.estimatePoseBoard(corners2_right,
                                                                                                   ids_right,
                                                                                                   self.calibation_board,
                                                                                                   self.K_right,
                                                                                                   self.D_right, None,
                                                                                                   None)

                        if retval_left and retval_right:
                            self.QueryImg = aruco.drawAxis(self.QueryImg, self.K_left, self.D_left, self.rvecs,
                                                           self.tvecs, 0.3)
                            self.QueryImg = aruco.drawDetectedMarkers(self.QueryImg, corners2_left, ids_left,
                                                                      borderColor=(0, 0, 255))
                            b = 1
                            imgpts, _ = cv2.projectPoints(self.pts, self.rvecs_right, self.tvecs_right, self.K_right,
                                                          self.D_right)
                            self.corners2_right = np.append(imgpts, np.mean(imgpts, axis=0)).reshape(-1, 2)

                            self.dst, jacobian = cv2.Rodrigues(self.rvecs)
                            a, circle_tvec, b = .49, [], 1
                            circle_tvec.append(
                                np.asarray(self.tvecs).squeeze() + np.dot(self.dst, np.asarray([a, a, 0])))
                            circle_tvec = np.mean(circle_tvec, axis=0)
                            self.QueryImg = aruco.drawAxis(self.QueryImg, self.K_left, self.D_left, self.rvecs,
                                                           circle_tvec, 0.2)

                            imgpts, _ = cv2.projectPoints(self.pts, self.rvecs, self.tvecs, self.K_left, self.D_left)
                            self.corners2 = np.append(imgpts, np.mean(imgpts, axis=0)).reshape(-1, 2)
                            self.pt_dict = {}
                            for i in range(len(self.pts)):
                                self.pt_dict[tuple(self.pts[i])] = tuple(imgpts[i].ravel())
                            top_right = self.pt_dict[tuple(self.pts[0])]
                            bot_right = self.pt_dict[tuple(self.pts[1])]
                            bot_left = self.pt_dict[tuple(self.pts[2])]
                            top_left = self.pt_dict[tuple(self.pts[3])]
                            cv2.circle(self.QueryImg, top_right, 4, (0, 0, 255), 5)
                            cv2.circle(self.QueryImg, bot_right, 4, (0, 0, 255), 5)
                            cv2.circle(self.QueryImg, bot_left, 4, (0, 0, 255), 5)
                            cv2.circle(self.QueryImg, top_left, 4, (0, 0, 255), 5)

                            self.QueryImg = cv2.line(self.QueryImg, top_right, bot_right, (0, 255, 0), 4)
                            self.QueryImg = cv2.line(self.QueryImg, bot_right, bot_left, (0, 255, 0), 4)
                            self.QueryImg = cv2.line(self.QueryImg, bot_left, top_left, (0, 255, 0), 4)
                            self.QueryImg = cv2.line(self.QueryImg, top_left, top_right, (0, 255, 0), 4)
                        else:
                            print('Cannot estimate board position for both charuco')

                        self.pixelsPoints = self.corners2.squeeze()
                        self.pixels_left = self.pixelsPoints
                        self.pixels_right = self.corners2_right.squeeze()
                        self.baseline = abs(self.T[0])
                        print('baseline:{} m'.format(self.baseline))
                        self.focal_length, self.cx, self.cy = self.K[0, 0], self.K[0, 2], self.K[1, 2]
                        self.x_left, self.x_right = self.pixels_left, self.pixels_right
                        disparity = np.sum(np.sqrt((self.x_left - self.x_right) ** 2), axis=1)
                        print('disparity:{}'.format(np.shape(disparity)))
                        # depth = baseline (meter) * focal length (pixel) / disparity-value (pixel) -> meter
                        self.depth = (self.baseline * self.focal_length / disparity)
                        print('depth:{}'.format(np.shape(self.depth)))
                        self.fxypxy = [self.K[0, 0], self.K[1, 1], self.cx, self.cy]
                    else:
                        print('No any board found!!!')
            else:
                # Undistortion
                h, w = self.QueryImg.shape[:2]
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.K, self.D, (w, h), 1, (w, h))
                dst = cv2.undistort(self.QueryImg, self.K, self.D, None, newcameramtx)
                x, y, w, h = roi
                self.QueryImg = dst[y:y + h, x:x + w]

                gray = cv2.cvtColor(self.QueryImg, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (10, 7), None)
                if ret:  # found chessboard
                    print('Found chessboard')
                    self.chessBoard = True
                    self.corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                    cv2.drawChessboardCorners(self.QueryImg, (10, 7), corners, ret)
                    ret, self.rvecs, self.tvecs = cv2.solvePnP(self.objp, self.corners2, self.K, self.D)
                    # ret, self.rvecs, self.tvecs, inliers = cv2.solvePnPRansac(self.objp, self.corners2, self.K, self.D)
                    self.imgpts, jac = cv2.projectPoints(self.axis, self.rvecs, self.tvecs, self.K, self.D)
                    self.QueryImg = self.draw(self.QueryImg, self.corners2, self.imgpts)
                    self.pixelsPoints = np.asarray(self.corners2).squeeze()
                else:  # check for charuco
                    self.chessBoard = False
                    self.useVoxel = False
                    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.ARUCO_DICT)
                    corners, ids, rejectedImgPoints, recoveredIds = aruco.refineDetectedMarkers(
                        image=gray, board=self.calibation_board, detectedCorners=corners, detectedIds=ids,
                        rejectedCorners=rejectedImgPoints, cameraMatrix=self.K, distCoeffs=self.D)

                    if np.all(ids != None):
                        print('found charuco board, ids:{}'.format(np.shape(ids)))
                        self.chessBoard = False
                        if len(ids) > 0:
                            retval, self.rvecs, self.tvecs = aruco.estimatePoseBoard(corners, ids,
                                                                                     self.calibation_board, self.K,
                                                                                     self.D, None, None)
                            if retval:
                                self.QueryImg = aruco.drawAxis(self.QueryImg, self.K, self.D, self.rvecs, self.tvecs,
                                                               0.3)
                                self.QueryImg = aruco.drawDetectedMarkers(self.QueryImg, corners, ids,
                                                                          borderColor=(0, 0, 255))
                                self.dst, jacobian = cv2.Rodrigues(self.rvecs)
                                a, circle_tvec, b = .49, [], 1
                                circle_tvec.append(
                                    np.asarray(self.tvecs).squeeze() + np.dot(self.dst, np.asarray([a, a, 0])))
                                circle_tvec = np.mean(circle_tvec, axis=0)
                                self.QueryImg = aruco.drawAxis(self.QueryImg, self.K, self.D, self.rvecs, circle_tvec,
                                                               0.2)
                                imgpts, _ = cv2.projectPoints(self.pts, self.rvecs, self.tvecs, self.K, self.D)
                                self.corners2 = np.append(imgpts, np.mean(imgpts, axis=0)).reshape(-1, 2)
                                self.pt_dict = {}
                                for i in range(len(self.pts)):
                                    self.pt_dict[tuple(self.pts[i])] = tuple(imgpts[i].ravel())
                                top_right = self.pt_dict[tuple(self.pts[0])]
                                bot_right = self.pt_dict[tuple(self.pts[1])]
                                bot_left = self.pt_dict[tuple(self.pts[2])]
                                top_left = self.pt_dict[tuple(self.pts[3])]
                                cv2.circle(self.QueryImg, top_right, 4, (0, 0, 255), 5)
                                cv2.circle(self.QueryImg, bot_right, 4, (0, 0, 255), 5)
                                cv2.circle(self.QueryImg, bot_left, 4, (0, 0, 255), 5)
                                cv2.circle(self.QueryImg, top_left, 4, (0, 0, 255), 5)

                                self.QueryImg = cv2.line(self.QueryImg, top_right, bot_right, (0, 255, 0), 4)
                                self.QueryImg = cv2.line(self.QueryImg, bot_right, bot_left, (0, 255, 0), 4)
                                self.QueryImg = cv2.line(self.QueryImg, bot_left, top_left, (0, 255, 0), 4)
                                self.QueryImg = cv2.line(self.QueryImg, top_left, top_right, (0, 255, 0), 4)
                    else:
                        print('No board Found')
            self.image_ax = self.fig.add_subplot(1, 2, 2)
            #self.image_ax = self.fig.add_subplot(1, 2, 1)
            self.image_ax.imshow(self.QueryImg)
            self.image_ax.set_axis_off()
            self.image_ax.set_xlabel('Y')
            self.image_ax.set_ylabel('Z')
        else:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_xlabel('X', fontsize=10)
        self.ax.set_ylabel('Y', fontsize=10)
        self.ax.set_zlabel('Z', fontsize=10)
        self.fig.tight_layout()
        plt.subplots_adjust(left=.15, bottom=0.2)
        #plt.subplots_adjust( bottom=0.2)

        self.Rx, self.Ry, self.Rz = [np.deg2rad(-90), 0, np.deg2rad(-40)] if self.chessBoard else [0, 0, 0]
        self.Tx, self.Ty, self.Tz = 0, 0, 0
        self.board_origin = [self.Tx, self.Ty, self.Tz]

        self.savePoints = Button(plt.axes([0.03, 0.45, 0.15, 0.04], ), 'filter points', color='white')
        self.savePoints.on_clicked(self.getClosestPoints)
        self.resetBtn = Button(plt.axes([0.03, 0.25, 0.15, 0.04], ), 'reset', color='white')
        self.resetBtn.on_clicked(self.reset)

        self.X_btn = Button(plt.axes([0.03, 0.9, 0.024, 0.04], ), 'X', color='red')
        self.X_btn.on_clicked(self.Close)

        self.OK_btn = Button(plt.axes([0.03, 0.83, 0.074, 0.04], ), 'OK', color='green')
        self.OK_btn.on_clicked(self.OK_btnClick)
        self.not_OK_btn = Button(plt.axes([0.105, 0.83, 0.074, 0.04], ), 'not OK', color='red')
        self.not_OK_btn.on_clicked(self.not_OK_btnClick)

        self.saveCorrespondences = Button(plt.axes([0.03, 0.76, 0.15, 0.04], ), 'Save points', color='white')
        self.saveCorrespondences.on_clicked(self.savePointsCorrespondences)

        self.fitChessboard = Button(plt.axes([0.03, 0.66, 0.15, 0.04], ), 'auto fit', color='white')
        self.fitChessboard.on_clicked(self.auto_fitBoard)

        # set up sliders
        self.Rx_Slider = Slider(plt.axes([0.25, 0.15, 0.65, 0.03]), 'Rx', -180, 180.0, valinit=np.degrees(self.Rx))
        self.Ry_Slider = Slider(plt.axes([0.25, 0.1, 0.65, 0.03]), 'Ry', -180, 180.0, valinit=np.degrees(self.Ry))
        self.Rz_Slider = Slider(plt.axes([0.25, 0.05, 0.65, 0.03]), 'Rz', -180, 180.0, valinit=np.degrees(self.Rz))
        self.Rx_Slider.on_changed(self.update_R)
        self.Ry_Slider.on_changed(self.update_R)
        self.Rz_Slider.on_changed(self.update_R)

        self.check = CheckButtons(plt.axes([0.03, 0.3, 0.15, 0.12]), ('Axes', 'Black', 'Annotate'),
                                  (self.axis_on, self.colour, self.Annotate))
        self.check.on_clicked(self.func_CheckButtons)

        # set up translation buttons
        self.step = .1  # m
        self.trigger = True

        self.Tx_btn_plus = Button(plt.axes([0.05, 0.15, 0.04, 0.045]), '+Tx', color='white')
        self.Tx_btn_plus.on_clicked(self.Tx_plus)
        self.Tx_btn_minus = Button(plt.axes([0.12, 0.15, 0.04, 0.045]), '-Tx', color='white')
        self.Tx_btn_minus.on_clicked(self.Tx_minus)

        self.Ty_btn_plus = Button(plt.axes([0.05, 0.1, 0.04, 0.045]), '+Ty', color='white')
        self.Ty_btn_plus.on_clicked(self.Ty_plus)
        self.Ty_btn_minus = Button(plt.axes([0.12, 0.1, 0.04, 0.045]), '-Ty', color='white')
        self.Ty_btn_minus.on_clicked(self.Ty_minus)

        self.Tz_btn_plus = Button(plt.axes([0.05, 0.05, 0.04, 0.045]), '+Tz', color='white')
        self.Tz_btn_plus.on_clicked(self.Tz_plus)
        self.Tz_btn_minus = Button(plt.axes([0.12, 0.05, 0.04, 0.045]), '-Tz', color='white')
        self.Tz_btn_minus.on_clicked(self.Tz_minus)

        self.Tx_flip = Button(plt.axes([0.17, 0.15, 0.04, 0.045]), 'FlipX', color='white')
        self.Tx_flip.on_clicked(self.flipX)
        self.Ty_flip = Button(plt.axes([0.17, 0.1, 0.04, 0.045]), 'FlipY', color='white')
        self.Ty_flip.on_clicked(self.flipY)
        self.Tz_flip = Button(plt.axes([0.17, 0.05, 0.04, 0.045]), 'FlipZ', color='white')
        self.Tz_flip.on_clicked(self.flipZ)

        self.radio = RadioButtons(plt.axes([0.03, 0.5, 0.15, 0.15], ), ('Final', 'Init'), active=0)
        self.radio.on_clicked(self.colorfunc)
        self.tag = None
        self.circle_center = None
        self.errors = {0: "Improper input parameters were entered.",
                       1: "The solution converged.",
                       2: "The number of calls to function has "
                          "reached maxfev = %d.",
                       3: "xtol=%f is too small, no further improvement "
                          "in the approximate\n  solution "
                          "is possible.",
                       4: "The iteration is not making good progress, as measured "
                          "by the \n  improvement from the last five "
                          "Jacobian evaluations.",
                       5: "The iteration is not making good progress, "
                          "as measured by the \n  improvement from the last "
                          "ten iterations.",
                       'unknown': "An error occurred."}

        self.legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Original pointcloud', markerfacecolor='g', markersize=4),
            Line2D([0], [0], marker='o', color='w', label='template', markerfacecolor='b', markersize=4),
            Line2D([0], [0], marker='o', color='w', label='ICP finetuned', markerfacecolor='r', markersize=4),
        ]

    def setUp(self):
        self.getPointCoud()
        self.axisEqual3D(centers=np.mean(self.point_cloud, axis=0))
        self.board()
        self.ax.legend(handles=self.legend_elements, loc='best')
        if self.showImage:
            self.getDepth_Inside_Outside()

        self.fitNewPlan()

    def auto_fitBoard(self, args):
        # estimate 3D-R and 3D-t between chess and PointCloud
        # Inital guess of the transformation
        x0 = np.array([np.degrees(self.Rx), np.degrees(self.Ry), np.degrees(self.Rz), self.Tx, self.Ty, self.Tz])
        report = {"error": [], "template": []}
        def f_min(x):
            self.Rx, self.Ry, self.Rz = np.deg2rad(x[0]), np.deg2rad(x[1]), np.deg2rad(x[2])
            self.Tx, self.Ty, self.Tz = x[3], x[4], x[5]
            template = self.board(plot=False)

            if self.useInitialPointCloud:
                dist_mat = distance_matrix(template, self.point_cloud)
            else:
                dist_mat = distance_matrix(template, self.corners_)
            err_func = dist_mat.sum(axis=1)  # N x 1
            # err_func = dist_mat.sum(axis=0)  # N x 1

            if self.debug:
                print('errors = {}, dist_mat:{}, err_func:{}'.format(round(np.sum(err_func), 2), np.shape(dist_mat),
                                                                     np.shape(err_func)))
            report["error"].append(np.sum(err_func))
            report["template"].append(template)
            return err_func

        maxIters = 700
        sol, status = leastsq(f_min, x0, ftol=1.49012e-07, xtol=1.49012e-07, maxfev=maxIters)
        print('sol:{}, status:{}'.format(sol, status))
        print(self.errors[status])

        if self.chess:
            self.chess.remove()
        if self.corn:
            self.corn.remove()
        if self.ICP_finetune_plot:
            self.ICP_finetune_plot.remove()
        self.lowerTemplate = False
        self.board()
        point_cloud = np.asarray(self.point_cloud, dtype=np.float32)
        template = np.asarray(report["template"][0], dtype=np.float32) if self.applyICP_directly else np.asarray(
            self.template_cloud, dtype=np.float32)
        converged, self.transf, estimate, fitness = self.ICP_finetune(template, point_cloud)
        # converged, self.transf, estimate, fitness = self.ICP_finetune(point_cloud,template)

        self.estimate = np.array(estimate)
        if self.chessBoard:
            self.ICP_finetune_plot = self.ax.scatter(self.estimate[:, 0], self.estimate[:, 1], self.estimate[:, 2],
                                                     c='k', marker='o', alpha=0.8, s=4)
        else:
            idx = np.arange(start=0, stop=100, step=1)
            idx = np.delete(idx, [44, 45, 54, 55])
            cornersToPLot = self.estimate[idx, :]
            self.ICP_finetune_plot = self.ax.scatter(cornersToPLot[:, 0], cornersToPLot[:, 1], cornersToPLot[:, 2],
                                                     c='k', marker='o', alpha=0.8, s=4)

        self.trigger = False
        # set values of sol to Sliders
        self.Rx_Slider.set_val(np.rad2deg(self.Rx))
        self.Ry_Slider.set_val(np.rad2deg(self.Ry))
        self.Rz_Slider.set_val(np.rad2deg(self.Rz))
        if self.chess:
            self.chess.remove()
        if self.corn:
            self.corn.remove()
        self.trigger = True
        self.board()
        self.AnnotateEdges()
        self.fig.canvas.draw_idle()

        if self.showError:
            print('min error:{} , at index:{}'.format(np.min(report["error"]), np.argmin(report["error"])))
            rep = plt.figure(figsize=(15, 8))
            plt.xlim(0, len(report["error"]) + 1)
            plt.xlabel('Iteration')
            plt.ylabel('RMSE')
            plt.yticks(color='w')
            plt.plot(np.arange(len(report["error"])) + 1, report["error"])

            print('Start animation gif')

            def update_graph(num):
                data = np.asarray(report["template"][num])
                graph._offsets3d = (data[:, 0], data[:, 1], data[:, 2])
                title.set_text('Iteration {}'.format(num))

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            title = ax.set_title('3D Test')

            data = report["template"][0]
            graph = ax.scatter(data[:, 0], data[:, 1], data[:, 2])
            ax.scatter(self.point_cloud[:, 0], self.point_cloud[:, 1], self.point_cloud[:, 2])
            ani = animation.FuncAnimation(fig, update_graph, 101, interval=2, blit=False, repeat=False)
            ani.save('myAnimation.gif', writer='imagemagick', fps=30)
            print('Animation done')
            plt.show()

    def flipX(self, event):
        self.Rx_Slider.set_val(np.rad2deg(self.Rx + np.pi))
        self.update_R(0)

    def flipY(self, event):
        self.Ry_Slider.set_val(np.rad2deg(self.Ry + np.pi))
        self.update_R(0)

    def flipZ(self, event):
        self.Rz_Slider.set_val(np.rad2deg(self.Rz + np.pi))
        self.update_R(0)

    def update_R(self, val):
        if self.trigger:
            if self.chess:
                self.chess.remove()
            if self.corn:
                self.corn.remove()
            self.Rx = np.deg2rad(self.Rx_Slider.val)
            self.Ry = np.deg2rad(self.Ry_Slider.val)
            self.Rz = np.deg2rad(self.Rz_Slider.val)
            self.board()
            self.fig.canvas.draw_idle()

    def board(self, plot=True, given_origin=None, angle=None):
        self.board_origin = [self.Tx, self.Ty, self.Tz] if given_origin is None else given_origin
        if self.chessBoard:
            self.nCols, self.nRows, org = 7 + 2, 10 + 2, np.asarray(self.board_origin)
            #org[0] -= self.nCols / 2
            #org[1] -= self.nRows / 2

            org[0] -= 4
            org[1] -= 6
            #org = np.zeros(3)
            if self.lowerTemplate:
                nrCols, nrRows = 2, 3
            else:
                nrCols, nrRows = self.nCols, self.nRows
                #nrCols, nrRows = self.nCols+1, self.nRows+1 #remove later
            print('org:{}, self.nCols - >{}, nrCols:{}'.format(org,self.nCols,nrCols))
            X, Y = np.linspace(org[0], org[0] + self.nCols, num=nrCols), np.linspace(org[1], org[1] + self.nRows,num=nrRows)
            X, Y = np.linspace(org[0], org[0] + self.nCols-1, num=nrCols), np.linspace(org[1], org[1] + self.nRows-1,
                                                                                     num=nrRows)
            print('X:{}'.format(X))
            X, Y = np.meshgrid(X, Y)
            Z = np.full(np.shape(X), org[2])
            colors, colortuple = np.empty(X.shape, dtype=str), ('k', 'w')
            for y in range(nrCols):
                for x in range(nrRows):
                    colors[x, y] = colortuple[(x + y) % len(colortuple)]
            colors[0, 0] = 'r'
            alpha = 0.65
        else:
            self.nCols, self.nRows, org = 10, 10, np.asarray(self.board_origin)
            org[0] -= self.nCols / 2
            org[1] -= self.nRows / 2
            # nrCols, nrRows = 4,4z
            nrCols, nrRows = self.nCols, self.nRows
            # nrCols, nrRows = 20, 20
            X, Y = np.linspace(org[0], org[0] + self.nCols, num=nrCols), np.linspace(org[1], org[1] + self.nRows,
                                                                                     num=nrRows)
            X, Y = np.meshgrid(X, Y)
            Z = np.full(np.shape(X), org[2])
            alpha = 0.25

        angles = np.array([self.Rx, self.Ry, self.Rz]) if angle is None else np.array(angle)
        Rot_matrix = self.eulerAnglesToRotationMatrix(angles)
        X, Y, Z = X * self.s, Y * self.s, Z * self.s
        corners = np.transpose(np.array([X, Y, Z]), (1, 2, 0))

        init = corners.reshape(-1, 3)

        print('corners-----------------------------------------------------')
        #print(init)
        print('corners -> {}'.format(np.shape(init)))
        dist_Lidar = distance_matrix(init, init)
        print('dist_Lidar corners---------------------------------------------------------')
        print(dist_Lidar[0, :11])

        translation = np.mean(init, axis=0)  # get the mean point
        corners = np.subtract(corners, translation)  # substract it from all the other points
        X, Y, Z = np.transpose(np.add(np.dot(corners, Rot_matrix), translation), (2, 0, 1))
        # corners = np.transpose(np.array([X, Y, Z]), (1, 2, 0)).reshape(-1, 3)
        corners = np.transpose(np.array([X, Y, Z]), (2, 1, 0)).reshape(-1, 3)

        if plot:
            if self.chessBoard:
                self.chess = self.ax.plot_surface(X, Y, Z, facecolors=colors, linewidth=0.2, cmap='gray', alpha=alpha)
            else:
                self.chess = self.ax.plot_surface(X, Y, Z, linewidth=0.2, cmap='gray', alpha=alpha)
                idx = np.arange(start=0, stop=100, step=1)
                idx = np.delete(idx, [44, 45, 54, 55])
                cornersToPLot = corners[idx, :]
                self.corn = self.ax.scatter(cornersToPLot[:, 0], cornersToPLot[:, 1], cornersToPLot[:, 2], c='tab:blue',
                                            marker='o', s=5)
        self.template_cloud = corners



        return np.array(corners)

    def getPointCoud(self, colorsMap='jet', skip=1, useRing = True):
        # X, Y, Z, intensity, ring
        if useRing:
            originalCloud = np.array(np.load(self.file, mmap_mode='r'))[:,:5]
            #mean_x = np.mean(originalCloud[:, 0])
            #originalCloud[:, 0] = mean_x
            df = pd.DataFrame(data=originalCloud, columns=["X", "Y", "Z","intens","ring"])
            gp = df.groupby('ring')
            keys = gp.groups.keys()
            #groups = gp.groups
            coolPoints, circlePoints = [],[]
            for i in keys:
                line = np.array(gp.get_group(i), dtype=np.float)
                first,last = np.array(line[0], dtype=np.float)[:3],np.array(line[-1], dtype=np.float)[:3]
                coolPoints.append(first)
                coolPoints.append(last)
                if self.chessBoard == False:
                    if len(line) > 50:
                        l = line[:,:3]
                        for i in range(2,len(l)-2,1):
                            d = np.linalg.norm(l[i]-l[i+1])
                            if d > 0.08: #half of the circle
                                circlePoints.append(l[i])
                                circlePoints.append(l[i+1])

            self.coolPoints = np.array(coolPoints).squeeze()
            self.ax.scatter(*self.coolPoints.T, color='r', marker='o', alpha=1, s=2)

            print('coolPoints:{},  circlePoints:{}'.format(np.shape(self.coolPoints), np.shape(circlePoints)))
            circlePoints = np.array(circlePoints)
            if len(circlePoints)>0:
                self.ax.scatter(*circlePoints.T, color='r', marker='o', alpha=1, s=5)

            self.fitCircle(circlePoints)

        #self.point_cloud = np.array(self.coolPoints, dtype=np.float32)
        self.point_cloud = np.array(np.load(self.file, mmap_mode='r')[::skip, :3], dtype=np.float32)

        # center the point_cloud
        #mean_x = np.mean(self.point_cloud[:, 0])
        #self.point_cloud[:, 0] = mean_x

        self.point_cloud_mean = np.mean(self.point_cloud, axis=0)
        self.Tx, self.Ty, self.Tz = self.point_cloud_mean
        # self.point_cloud = self.point_cloud - self.point_cloud_mean
        self.point_cloud_colors = np.array(np.load(self.file, mmap_mode='r'))[::skip, 3]
        if self.plotInit:
            cm = plt.get_cmap(colorsMap)
            cNorm = matplotlib.colors.Normalize(vmin=min(self.point_cloud_colors), vmax=max(self.point_cloud_colors))
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

            self.p1 = self.ax.scatter(self.point_cloud[:, 0], self.point_cloud[:, 1], self.point_cloud[:, 2],
                                      color=scalarMap.to_rgba(self.point_cloud_colors), s=0.2)
        else:
            self.p = pcl.PointCloud(self.point_cloud)
            inlier, outliner, coefficients = self.do_ransac_plane_segmentation(self.p, pcl.SACMODEL_PLANE,
                                                                               pcl.SAC_RANSAC, 0.01)
            self.planeEquation(coef=np.array(coefficients).squeeze())
            self.point_cloud_init = self.point_cloud.copy()
            if self.useVoxel:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(self.point_cloud)
                self.point_cloud = np.array(pcd.voxel_down_sample(voxel_size=self.voxel_size).points)

            # self.p1 = self.ax.scatter(outliner[:, 0], outliner[:, 1], outliner[:, 2], c='y', s=0.2)
            self.p2 = self.ax.scatter(inlier[:, 0], inlier[:, 1], inlier[:, 2], c='g', s=0.2)

            w, v = self.PCA(inlier)
            point = np.mean(inlier, axis=0)
            if self.chessBoard == False and self.circle_center:
                #point[1:] = self.circle_center
                point[[0,2]]= self.circle_center
            w *= 2
            if self.chessBoard==False and self.circle_center:
                p = Circle(self.circle_center, self.circle_radius, alpha = .3, color='tab:blue')
                self.ax.add_patch(p)
                art3d.pathpatch_2d_to_3d(p, z=point[1], zdir="y")

            self.p3 = self.ax.quiver([point[0]], [point[1]], [point[2]], [v[0, :] * np.sqrt(w[0])],
                                     [v[1, :] * np.sqrt(w[0])],
                                     [v[2, :] * np.sqrt(w[0])], linewidths=(1.8,))

    def axisEqual3D(self, centers=None):
        extents = np.array([getattr(self.ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
        sz = extents[:, 1] - extents[:, 0]
        # centers = np.mean(extents, axis=1) if centers is None
        maxsize = max(abs(sz))
        r = maxsize / 2
        for ctr, dim in zip(centers, 'xyz'):
            getattr(self.ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

    def planeEquation(self, coef):
        a, b, c, d = coef
        mean = np.mean(self.point_cloud, axis=0)
        normal = [a, b, c]
        d2 = -mean.dot(normal)
        # print('d2:{}'.format(d2))
        # print('mean:{}'.format(mean))
        # print('The equation is {0}x + {1}y + {2}z = {3}'.format(a, b, c, d))

        # plot the normal vector
        startX, startY, startZ = mean[0], mean[1], mean[2]
        startZ = (-normal[0] * startX - normal[1] * startY - d) * 1. / normal[2]
        self.ax.quiver([startX], [startY], [startZ], [normal[0]], [normal[1]], [normal[2]], linewidths=(3,),edgecolor="red")

    def PCA(self, data, correlation=False, sort=True):
        # data = nx3
        mean = np.mean(data, axis=0)
        data_adjust = data - mean
        #: the data is transposed due to np.cov/corrcoef syntax
        if correlation:
            matrix = np.corrcoef(data_adjust.T)
        else:
            matrix = np.cov(data_adjust.T)
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        if sort:
            #: sort eigenvalues and eigenvectors
            sort = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[sort]
            eigenvectors = eigenvectors[:, sort]

        return eigenvalues, eigenvectors

    def eulerAnglesToRotationMatrix(self, theta):
        R_x = np.array([[1, 0, 0],
                        [0, math.cos(theta[0]), -math.sin(theta[0])],
                        [0, math.sin(theta[0]), math.cos(theta[0])]
                        ])

        R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                        [0, 1, 0],
                        [-math.sin(theta[1]), 0, math.cos(theta[1])]
                        ])

        R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                        [math.sin(theta[2]), math.cos(theta[2]), 0],
                        [0, 0, 1]
                        ])

        R = np.dot(R_z, np.dot(R_y, R_x))
        return R

    def do_ransac_plane_segmentation(self, pcl_data, pcl_sac_model_plane, pcl_sac_ransac, max_distance):
        """
        Create the segmentation object
        :param pcl_data: point could data subscriber
        :param pcl_sac_model_plane: use to determine plane models
        :param pcl_sac_ransac: RANdom SAmple Consensus
        :param max_distance: Max distance for apoint to be considered fitting the model
        :return: segmentation object
        """
        seg = pcl_data.make_segmenter()
        seg.set_model_type(pcl_sac_model_plane)
        seg.set_method_type(pcl_sac_ransac)
        seg.set_distance_threshold(max_distance)

        inliers, coefficients = seg.segment()
        inlier_object = pcl_data.extract(inliers, negative=False)
        outlier_object = pcl_data.extract(inliers, negative=True)
        if len(inliers) <= 1:
            outlier_object = [0, 0, 0]
        inlier_object, outlier_object = np.array(inlier_object), np.array(outlier_object)
        return inlier_object, outlier_object, coefficients

    def func_CheckButtons(self, label):
        if label == 'Axes':
            if self.axis_on:
                self.ax.set_axis_off()
                self.axis_on = False
            else:
                self.ax.set_axis_on()
                self.axis_on = True
        elif label == 'Black':
            if self.colour:
                self.colour = False
                self.ax.set_facecolor((1, 1, 1))
            else:
                self.colour = True
                self.ax.set_facecolor((0, 0, 0))
        elif label == 'Annotate':
            self.Annotate = not self.Annotate
            self.AnnotateEdges()

        self.fig.canvas.draw_idle()

    def ICP_finetune(self, points_in, points_out):
        cloud_in = pcl.PointCloud()
        cloud_out = pcl.PointCloud()
        cloud_in.from_array(points_in)
        cloud_out.from_array(points_out)

        # icp = cloud_in.make_IterativeClosestPoint()
        icp = cloud_out.make_IterativeClosestPoint()
        converged, transf, estimate, fitness = icp.icp(cloud_in, cloud_out)

        print('fitness:{}, converged:{}, transf:{}, estimate:{}'.format(fitness, converged, np.shape(transf),
                                                                        np.shape(estimate)))
        return converged, transf, estimate, fitness

    def colorfunc(self, label):
        if label == 'Init':
            self.plotInit = True
        else:
            self.plotInit = False

        self.reset(0)

    def OK_btnClick(self, args):
        self.OK = True
        plt.close()

    def not_OK_btnClick(self, args):
        self.OK = False
        plt.close()

    def Close(self, args):
        global globalTrigger
        globalTrigger = False
        plt.close()

    def reset(self, args):
        self.ax.cla()
        self.getPointCoud()
        self.axisEqual3D(centers=np.mean(self.point_cloud, axis=0))
        self.Rx, self.Ry, self.Rz = 0, 0, 0
        self.Tx, self.Ty, self.Tz = 0, 0, 0
        self.board_origin = [self.Tx, self.Ty, self.Tz]
        self.board()
        self.fig.canvas.draw_idle()

    def getClosestPoints(self, arg):
        dist_mat = distance_matrix(self.template_cloud, self.point_cloud_init)
        self.neighbours = np.argsort(dist_mat, axis=1)[:, 0]
        self.finaPoints = np.asarray(self.point_cloud_init[self.neighbours, :]).squeeze()

        if self.chess:
            self.chess.remove()
        if self.corn:
            self.corn.remove()
        if self.p3:
            self.p3.remove()
        if self.p2:
            self.p2.remove()
        if self.p1:
            self.p1.remove()

        self.scatter_finalPoints = self.ax.scatter(self.finaPoints[:, 0], self.finaPoints[:, 1], self.finaPoints[:, 2],
                                                   c='k', marker='x', s=1)
        self.corn = self.ax.scatter(self.template_cloud[:, 0], self.template_cloud[:, 1], self.template_cloud[:, 2],
                                    c='blue', marker='o', s=5)

        self.fig.canvas.draw_idle()

    def Tz_plus(self, event):
        self.Tz += self.step
        self.update_R(0)

    def Tz_minus(self, event):
        self.Tz -= self.step
        self.update_R(0)

    def Ty_plus(self, event):
        self.Ty += self.step
        self.update_R(0)

    def Ty_minus(self, event):
        self.Ty -= self.step
        self.update_R(0)

    def Tx_plus(self, event):
        self.Tx += self.step
        self.update_R(0)

    def Tx_minus(self, event):
        self.Tx -= self.step
        self.update_R(0)

    def readCameraIntrin(self):
        name = 'inside'
        name = 'outside'
        self.camera_model = load_obj('{}_combined_camera_model'.format(name))
        self.camera_model_rectify = load_obj('{}_combined_camera_model_rectify'.format(name))

        self.K_left = self.camera_model['K_left']
        self.K_right = self.camera_model['K_right']
        self.D_left = self.camera_model['D_left']
        self.D_right = self.camera_model['D_right']

        # self.K_left = self.camera_model['K_right']
        # self.K_right = self.camera_model['K_left']
        # self.D_left = self.camera_model['D_right']
        # self.D_right = self.camera_model['D_left']

        # print('K_left')
        # print(self.K_left)
        # print('K_right')
        # print(self.K_right)

        self.R = self.camera_model['R']
        self.T = self.camera_model['T']

        #self.T = np.array([-0.98, 0., 0.12])[:, np.newaxis]
        #self.T = np.array([-.75, 0., 0.])[:, np.newaxis]

        #print('self T after {}'.format(np.shape(self.T)))
        #angles = np.array([np.deg2rad(0.68), np.deg2rad(22.66), np.deg2rad(-1.05)])
        #self.R = euler_matrix(angles)

        #Q = self.camera_model_rectify['Q']
        #roi_left, roi_right = self.camera_model_rectify['roi_left'], self.camera_model_rectify['roi_right']
        self.leftMapX, self.leftMapY = self.camera_model_rectify['leftMapX'], self.camera_model_rectify['leftMapY']
        self.rightMapX, self.rightMapY = self.camera_model_rectify['rightMapX'], self.camera_model_rectify['rightMapY']

        img_shape = (1936, 1216)
        print('img_shape:{}'.format(img_shape))
        R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(self.K_left, self.D_left, self.K_right, self.D_right,
                                                                   imageSize=img_shape,
                                                                   R=self.camera_model['R'], T=self.camera_model['T'],
                                                                   flags=cv2.CALIB_ZERO_DISPARITY,
                                                                   # alpha=-1
                                                                   alpha=0
                                                                   )
        self.leftMapX, self.leftMapY = cv2.initUndistortRectifyMap(
            self.K_left, self.D_left, R1,
            P1, img_shape, cv2.CV_32FC1)

        self.rightMapX, self.rightMapY = cv2.initUndistortRectifyMap(
            self.K_right, self.D_right, R2,
            P2, img_shape, cv2.CV_32FC1)

        self.K = self.K_right
        self.D = self.D_right
        try:
            N = 5
            aruco_dict = aruco.custom_dictionary(0, N, 1)
            aruco_dict.bytesList = np.empty(shape=(4, N - 1, N - 1), dtype=np.uint8)
            A = np.array([[0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 1, 0, 1, 0], [0, 1, 1, 1, 0], [0, 1, 0, 1, 0]],
                         dtype=np.uint8)
            aruco_dict.bytesList[0] = aruco.Dictionary_getByteListFromBits(A)
            R = np.array([[1, 1, 1, 1, 0], [1, 0, 0, 1, 0], [1, 1, 1, 0, 0], [1, 0, 0, 1, 0], [1, 0, 0, 0, 1]],
                         dtype=np.uint8)
            aruco_dict.bytesList[1] = aruco.Dictionary_getByteListFromBits(R)
            V = np.array([[1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0]],
                         dtype=np.uint8)
            O = np.array([[0, 1, 1, 1, 0], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [0, 1, 1, 1, 0]],
                         dtype=np.uint8)
            aruco_dict.bytesList[2] = aruco.Dictionary_getByteListFromBits(O)
            aruco_dict.bytesList[3] = aruco.Dictionary_getByteListFromBits(V)

            self.ARUCO_DICT = aruco_dict
            self.calibation_board = aruco.GridBoard_create(
                markersX=2, markersY=2,
                markerLength=0.126, markerSeparation=0.74,
                dictionary=self.ARUCO_DICT)
        except:
            print('Install Aruco')

    def draw(self, img, corners, imgpts):
        corner = tuple(corners[0].ravel())
        cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
        cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
        cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
        return img

    def annotate3D(self, ax, s, *args, **kwargs):
        self.tag = Annotation3D(s, *args, **kwargs)
        ax.add_artist(self.tag)

    def AnnotateEdges(self, giveAX=None, givenPoints=None):
        if self.Annotate:
            # add vertices annotation.
            if giveAX is None:
                if self.lowerTemplate or self.chessBoard == False:
                    if self.chessBoard == False:
                        pts = np.asarray(self.template_cloud.copy()).reshape(self.nCols, self.nRows, 3)
                        idx = np.array([44, 45, 54, 55])
                        center = np.mean(self.template_cloud[idx], axis=0)
                        self.templatePoints = [pts[0, -1, :], pts[-1, -1, :], pts[-1, 0, :], pts[0, 0, :], center]

                        self.templatePoints = np.array(self.templatePoints).reshape(-1, 3)

                        cornersToPLot = self.estimate[idx, :]
                        for j, xyz_ in enumerate(self.templatePoints):
                            self.annotate3D(self.ax, s=str(j), xyz=xyz_, fontsize=12, xytext=(-1, 1),
                                            textcoords='offset points', ha='right', va='bottom')
                    else:
                        for j, xyz_ in enumerate(self.template_cloud):
                            self.annotate3D(self.ax, s=str(j), xyz=xyz_, fontsize=8, xytext=(-1, 1),
                                            textcoords='offset points', ha='right', va='bottom')

                else:
                    try:
                        templatePoints = np.asarray(self.template_cloud.copy()).reshape(self.nCols, self.nRows, 3)[
                                         1:self.nCols - 1, 1:self.nRows - 1, :]
                    except:
                        templatePoints = np.asarray(self.template_cloud.copy()).reshape(self.nCols+1, self.nRows+1, 3)[
                                         1:self.nCols - 1, 1:self.nRows - 1, :]
                    # templatePoints = np.asarray(self.template_cloud.copy()).reshape(self.nRows,self.nCols, 3)[1:self.nRows-1,1:self.nCols-1,:]

                    self.templatePoints = np.array(templatePoints).reshape(-1, 3)
                    for j, xyz_ in enumerate(self.templatePoints):
                        self.annotate3D(self.ax, s=str(j), xyz=xyz_, fontsize=8, xytext=(-3, 3),
                                        textcoords='offset points', ha='right', va='bottom')
            else:
                for j, xyz_ in enumerate(givenPoints):
                    self.annotate3D(giveAX, s=str(j), xyz=xyz_, fontsize=10, xytext=(-3, 3),
                                    textcoords='offset points', ha='right', va='bottom')

            if self.showImage:
                # annotate image
                points = np.asarray(self.corners2).squeeze()
                font, lineType = cv2.FONT_HERSHEY_SIMPLEX, 2 if self.chessBoard else 10
                for i, point in enumerate(points):
                    point = tuple(point.ravel())
                    cv2.putText(self.QueryImg, '{}'.format(i), point, font, 1 if self.chessBoard else 3, (0, 0, 0)
                    if self.chessBoard else (255, 0, 0), lineType)
                self.image_ax.imshow(self.QueryImg)

    def getCamera_XYZ_Stereo(self):
        #cam_rot, jac = cv2.Rodrigues(self.rvecs)
        #mR = np.matrix(cam_rot)
        #mT = np.matrix(self.tvecs)
        #cam_trans = -mR * mT

        _3DPoints = []
        for i, pixel in enumerate(self.x_left):
            u, v = pixel.ravel()
            u, v = int(u), int(v)
            distance = self.depth[i]
            pt = np.array([u, v, distance])

            pt[0] = pt[2] * (pt[0] - self.fxypxy[2]) / self.fxypxy[0]
            pt[1] = pt[2] * (pt[1] - self.fxypxy[3]) / self.fxypxy[1]

            # pt = pt.dot(cam_rot.T) + self.tvecs
            _3DPoints.append(pt)
        print('_3DPoints {}'.format(np.shape(_3DPoints)))
        print('tvec : {}'.format(np.asarray(self.tvecs).squeeze()))
        print('Camera_XYZ_Stereo mean {}'.format(np.mean(_3DPoints, axis=0)))

        return np.array(_3DPoints).squeeze()

    def getCamera_XYZ(self):
        R_mtx, jac = cv2.Rodrigues(self.rvecs)
        inv_R_mtx = np.linalg.inv(R_mtx)
        inv_K = np.linalg.inv(self.K)

        def compute_XYZ(u, v):  # from 2D pixels to 3D world
            uv_ = np.array([[u, v, 1]], dtype=np.float32).T
            suv_ = uv_
            xyz_ = inv_K.dot(suv_) - self.tvecs
            XYZ = inv_R_mtx.dot(xyz_)
            pred = XYZ.T[0]
            return pred

        Camera_XYZ = []
        for i, point in enumerate(self.pixelsPoints):
            xyz = compute_XYZ(u=point[0], v=point[1])
            # print 'xyz:{}'.format(xyz)
            Camera_XYZ.append(xyz)
        Camera_XYZ = np.array(Camera_XYZ)
        print('init tvec : {}'.format(np.asarray(self.tvecs).squeeze()))
        print('Camera_XYZ mean {}'.format(np.mean(Camera_XYZ, axis=0)))
        if self.img_file2 is None:
            for i, point in enumerate(Camera_XYZ):
                imgpts, jac = cv2.projectPoints(point, self.rvecs, self.tvecs, self.K, self.D)
                imgpts = np.asarray(imgpts).squeeze()
                cv2.circle(self.QueryImg, (int(imgpts[0]), int(imgpts[1])), 7, (255, 0, 0), 7)
                self.image_ax.imshow(self.QueryImg)

        return Camera_XYZ

    def getImagePixels(self):
        img = cv2.imread(self.img_file) #left image
        img2 = cv2.imread(self.img_file2)  # left image

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        pixelsPoints,pixelsPoints2, _3DreconstructedBoard = [],[],[]
        if self.chessBoard:
            ret, corners = cv2.findChessboardCorners(gray, (10, 7), None)
            ret2, corners2 = cv2.findChessboardCorners(gray2, (10, 7), None)
            if ret and ret2:  # found chessboard
                print('Found chessboard')
                corners_2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                corners2_2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), self.criteria)
                pixelsPoints = np.asarray(corners_2).squeeze()
                pixelsPoints2 = np.asarray(corners2_2).squeeze()
                cv2.drawChessboardCorners(img, (10, 7), corners_2, ret)
                cv2.drawChessboardCorners(img2, (10, 7), corners2_2, ret)

                # Find the rotation and translation vectors.
                success, rvecs, tvecs, inliers = cv2.solvePnPRansac(self.objp, corners_2, self.K, self.D)
                rvecs, _ = cv2.Rodrigues(rvecs)
                _3Dpoints = self.objp
                # project 3D points to image plane
                _2Dpoints, jac = cv2.projectPoints(_3Dpoints, rvecs, tvecs, self.K, self.D)
                _2Dpoints = np.array(_2Dpoints, dtype=np.float32).squeeze()
                print('_2Dpoints -> {}'.format(np.shape(_2Dpoints)))
                for i in range(len(_2Dpoints)):
                    cv2.circle(img, tuple(_2Dpoints[i]), 5, (0, 255, 0), 3)
                _3Dpoints = rvecs.dot(_3Dpoints.T) + tvecs
                _3Dpoints = _3Dpoints.T
                print('_3Dpoints->{}'.format(np.shape(_3Dpoints)))
                dist_mat = distance_matrix(_3Dpoints, _3Dpoints)
                print('dist_mat for OpencvReconstructed')
                print(dist_mat[0, :11])
                _3DreconstructedBoard = _3Dpoints
            else:
                return None,None
        else:
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.ARUCO_DICT)
            corners, ids, rejectedImgPoints, recoveredIds = aruco.refineDetectedMarkers(
                image=gray, board=self.calibation_board, detectedCorners=corners, detectedIds=ids,
                rejectedCorners=rejectedImgPoints, cameraMatrix=self.K, distCoeffs=self.D)

            corners2, ids2, rejectedImgPoints2 = aruco.detectMarkers(gray2, self.ARUCO_DICT)
            corners2, ids2, rejectedImgPoints2, recoveredIds2 = aruco.refineDetectedMarkers(
                image=gray2, board=self.calibation_board, detectedCorners=corners2, detectedIds=ids2,
                rejectedCorners=rejectedImgPoints2, cameraMatrix=self.K, distCoeffs=self.D)

            if np.all(ids != None) and np.all(ids2 != None):
                print('found charuco board, ids:{}'.format(np.shape(ids)))
                if len(ids) and len(ids2) > 0:
                    retval, self.rvecs, self.tvecs = aruco.estimatePoseBoard(corners, ids,
                                                                             self.calibation_board, self.K,
                                                                             self.D, None, None)

                    retval2, self.rvecs2, self.tvecs2 = aruco.estimatePoseBoard(corners2, ids2,
                                                                             self.calibation_board, self.K,
                                                                             self.D, None, None)
                    img = aruco.drawDetectedMarkers(img, corners, ids,borderColor=(0, 0, 255))
                    img2 = aruco.drawDetectedMarkers(img2, corners2, ids2, borderColor=(0, 0, 255))
                    if retval and retval2:
                        self.dst, jacobian = cv2.Rodrigues(self.rvecs)
                        self.dst2, jacobian = cv2.Rodrigues(self.rvecs2)
                        #self.pts = np.float32([[0, b, 0], [b, b, 0], [b, 0, 0], [-0.03, -0.03, 0]])
                        b = 1
                        self.pts = np.float32([[0, b, 0], [b, b, 0], [b, 0, 0], [-0.03, -0.03, 0],[.5,.5,0]])
                        _3Dpoints = self.dst.T.dot(np.array(self.pts).squeeze().T) + self.tvecs

                        _3Dpoints = _3Dpoints.T
                        print('_3Dpoints->{}'.format(np.shape(_3Dpoints)))
                        dist_mat = distance_matrix(_3Dpoints, _3Dpoints)
                        print('dist_mat for OpencvReconstructed')
                        print(dist_mat)
                        _3DreconstructedBoard = _3Dpoints


                        imgpts, _ = cv2.projectPoints(self.pts, self.rvecs, self.tvecs, self.K, self.D)
                        #corners2 = np.append(imgpts, np.mean(imgpts, axis=0)).reshape(-1, 2)
                        corners2 = np.array(imgpts).squeeze()
                        self.pt_dict = {}
                        for i in range(len(self.pts)):
                            self.pt_dict[tuple(self.pts[i])] = tuple(imgpts[i].ravel())
                        top_right = self.pt_dict[tuple(self.pts[0])]
                        bot_right = self.pt_dict[tuple(self.pts[1])]
                        bot_left = self.pt_dict[tuple(self.pts[2])]
                        top_left = self.pt_dict[tuple(self.pts[3])]
                        img = cv2.line(img, top_right, bot_right, (0, 255, 0), 4)
                        img = cv2.line(img, bot_right, bot_left, (0, 255, 0), 4)
                        img = cv2.line(img, bot_left, top_left, (0, 255, 0), 4)
                        img = cv2.line(img, top_left, top_right, (0, 255, 0), 4)
                        cv2.circle(img, tuple(corners2[-1]), 5, (0, 255, 0), 3)
                        cv2.circle(img, tuple(corners2[-2]), 5, (0, 0, 255), 3)
                        pixelsPoints = np.asarray(corners2).squeeze()


                        imgpts, _ = cv2.projectPoints(self.pts, self.rvecs2, self.tvecs2, self.K, self.D)
                        #corners2 = np.append(imgpts, np.mean(imgpts, axis=0)).reshape(-1, 2)
                        corners2 = np.array(imgpts).squeeze()
                        self.pt_dict = {}
                        for i in range(len(self.pts)):
                            self.pt_dict[tuple(self.pts[i])] = tuple(imgpts[i].ravel())
                        top_right = self.pt_dict[tuple(self.pts[0])]
                        bot_right = self.pt_dict[tuple(self.pts[1])]
                        bot_left = self.pt_dict[tuple(self.pts[2])]
                        top_left = self.pt_dict[tuple(self.pts[3])]

                        img2 = cv2.line(img2, top_right, bot_right, (0, 255, 0), 4)
                        img2 = cv2.line(img2, bot_right, bot_left, (0, 255, 0), 4)
                        img2 = cv2.line(img2, bot_left, top_left, (0, 255, 0), 4)
                        img2 = cv2.line(img2, top_left, top_right, (0, 255, 0), 4)
                        cv2.circle(img2, tuple(corners2[-1]), 5, (0, 255, 0), 3)
                        #cv2.circle(img2, tuple(corners2[-2]), 5, (0, 0, 255), 3)
                        pixelsPoints2 = np.asarray(corners2).squeeze()


                    else:
                        return None,None
                else:
                    return None,None
            else:
                return None,None

        scale = .4
        _horizontal = np.hstack(
            (cv2.resize(img, None, fx=scale, fy=scale), cv2.resize(img2, None, fx=scale, fy=scale)))

        cv2.imshow('_horizontal', _horizontal)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return pixelsPoints,pixelsPoints2, _3DreconstructedBoard

    def savePointsCorrespondences(self, args):
        display = True
        fig = plt.figure(figsize=plt.figaspect(1))
        ax = plt.axes(projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        if self.chessBoard:
            legend_elements = [
                Line2D([0], [0], marker='o', label='board template', markerfacecolor='tab:blue', markersize=6),
                Line2D([0], [0], marker='o', label='ICP finetuned', markerfacecolor='green', markersize=6),
                Line2D([0], [0], marker='o', label='closest lidar points', markerfacecolor='k', markersize=6),
                Line2D([0], [0], marker='o', label='Camera_XYZ', markerfacecolor='red', markersize=6),
            ]

            board_template = self.template_cloud
            board_template_ICP_finetuned = self.estimate
            closest_lidar_points = self.finaPoints
            try:
                icp_finetuned_inside = np.asarray(self.estimate).reshape(self.nCols, self.nRows, 3)[1:self.nCols - 1,
                                       1:self.nRows - 1, :]

                board_template_inside = board_template.reshape(self.nCols, self.nRows, 3)[1:self.nCols - 1,
                                        1:self.nRows - 1, :]

                closest_lidar_points_inside = closest_lidar_points.reshape(self.nCols, self.nRows, 3)[1:self.nCols - 1,
                                              1:self.nRows - 1, :]

            except:
                print('Second-----------------------------')
                icp_finetuned_inside = np.asarray(self.estimate).reshape(self.nCols+1, self.nRows+1, 3)[1:self.nCols - 1,
                                       1:self.nRows - 1, :]

                board_template_inside = board_template.reshape(self.nCols+1, self.nRows+1, 3)[1:self.nCols - 1,
                                        1:self.nRows - 1, :]
                closest_lidar_points_inside = closest_lidar_points.reshape(self.nCols+1, self.nRows+1, 3)[1:self.nCols - 1,
                                              1:self.nRows - 1, :]

            icp_finetuned_inside = np.array(icp_finetuned_inside).reshape(-1, 3)
            board_template_inside = np.array(board_template_inside).reshape(-1, 3)
            print('board_template_inside-----------------------------------------------------')
            print(board_template_inside)
            print('board_template_inside -> {}'.format(np.shape(board_template_inside)))
            dist_Lidar = distance_matrix(board_template_inside, board_template_inside)
            print('dist_Lidar---------------------------------------------------------')
            print(dist_Lidar[0, :11])



            closest_lidar_points_inside = np.array(closest_lidar_points_inside).reshape(-1, 3)

            Camera_XYZ = self.getCamera_XYZ()
            if self.img_file2:
                Camera_XYZ_Stereo = self.getCamera_XYZ_Stereo()
            else:
                Camera_XYZ_Stereo = np.array([[0, 0, 0]])

            display = True
            if display:
                print('board_template:{}'.format(np.shape(board_template)))
                print('board_template_ICP_finetuned:{}'.format(np.shape(board_template_ICP_finetuned)))
                print('icp_finetuned_inside:{}'.format(np.shape(icp_finetuned_inside)))
                print('board_template_inside:{}'.format(np.shape(board_template_inside)))
                print('closest_lidar_points:{}'.format(np.shape(closest_lidar_points)))
                print('closest_lidar_points_inside:{}'.format(np.shape(closest_lidar_points_inside)))
                print('Camera_XYZ:{}'.format(np.shape(Camera_XYZ)))
                print('Camera_XYZ_Stereo:{}'.format(np.shape(Camera_XYZ_Stereo)))

            #dist = distance_matrix(Camera_XYZ_Stereo, Camera_XYZ_Stereo)
            #print('distance matrix Camera_XYZ_Stereo:{}'.format(dist))

            ax.scatter(*board_template.T, color='b', marker='o', alpha=.5, s=8)
            ax.scatter(*board_template_ICP_finetuned.T, color='r', marker='o', alpha=.5, s=8)
            ax.scatter(*board_template_inside.T, color='tab:blue', marker='x', alpha=1, s=10)
            ax.scatter(*icp_finetuned_inside.T, color='g', marker='x', alpha=1, s=10)
            ax.scatter(*closest_lidar_points.T, color='r', marker='x', alpha=.8, s=10)
            ax.scatter(*closest_lidar_points_inside.T, color='k', marker='x', alpha=1, s=20)
            ax.scatter(*Camera_XYZ.T, color='k', marker='x', alpha=1, s=30)
            ax.scatter(*Camera_XYZ_Stereo.T, color='r', marker='o', alpha=1, s=3)

            self.AnnotateEdges(giveAX=ax, givenPoints=board_template_inside)

            extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
            sz = extents[:, 1] - extents[:, 0]
            centers = np.mean(board_template, axis=0)
            # centers = np.mean(Camera_XYZ_Stereo, axis=0) if self.img_file2 is not None else np.mean(board_template,axis=0)
            maxsize = max(abs(sz))
            r = maxsize / 2
            for ctr, dim in zip(centers, 'xyz'):
                getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

            self.pixelsPointsLeft, self.pixelsPointsRight, _3DreconstructedBoard = self.getImagePixels()
            print('_3DreconstructedBoard -> {}'.format(np.shape(_3DreconstructedBoard)))
            if len(self.pixelsPointsLeft)<=0:
                print('Cannot get pixels points !!! ')

            self.points_correspondences = dict([
                ('board_template', board_template),
                ('board_template_ICP_finetuned', board_template_ICP_finetuned),
                ('board_template_inside', board_template_inside),
                ('icp_finetuned_inside', icp_finetuned_inside),
                ('closest_lidar_points', closest_lidar_points),
                ('closest_lidar_points_inside', closest_lidar_points_inside),
                ('pixelsPointsLeft', self.pixelsPointsLeft),
                ('pixelsPointsRight', self.pixelsPointsRight),
                ('Camera_XYZ_Stereo', Camera_XYZ_Stereo),
                ('_3DreconstructedBoard',_3DreconstructedBoard),
                ('Camera_XYZ', Camera_XYZ)])

            # save_obj(self.points_correspondences, self.name)

        else:
            legend_elements = [
                Line2D([0], [0], marker='o', label='board template all', markerfacecolor='b', markersize=6),
                Line2D([0], [0], marker='o', label='ICP finetuned', markerfacecolor='red', markersize=6),
                Line2D([0], [0], marker='o', label='board template inside', markerfacecolor='tab:blue', markersize=6),
                Line2D([0], [0], marker='o', label='closest lidar points', markerfacecolor='red', markersize=6),
            ]

            pts = np.asarray(self.template_cloud.copy()).reshape(self.nCols, self.nRows, 3)
            idx = np.array([44, 45, 54, 55])
            center = np.mean(self.template_cloud[idx], axis=0)
            board_template = np.array([pts[0, -1, :], pts[-1, -1, :], pts[-1, 0, :], pts[0, 0, :], center]).reshape(-1,
                                                                                                                    3)
            board_template = board_template

            pts = np.asarray(self.estimate.copy()).reshape(self.nCols, self.nRows, 3)
            center = np.mean(self.estimate[idx], axis=0)
            board_template_ICP_finetuned = np.array(
                [pts[0, -1, :], pts[-1, -1, :], pts[-1, 0, :], pts[0, 0, :], center]).reshape(-1, 3)

            board_template_inside = self.templatePoints

            pts = np.asarray(self.finaPoints.copy()).reshape(self.nCols, self.nRows, 3)
            center = np.mean(self.finaPoints[idx], axis=0)
            closest_lidar_points = np.array(
                [pts[0, -1, :], pts[-1, -1, :], pts[-1, 0, :], pts[0, 0, :], center]).reshape(-1, 3)

            if self.img_file2:
                Camera_XYZ_Stereo = self.getCamera_XYZ_Stereo()
            else:
                Camera_XYZ_Stereo = np.array([[0, 0, 0]])

            if display:
                print('board_template:{}'.format(np.shape(board_template)))
                print('board_template_ICP_finetuned:{}'.format(np.shape(board_template_ICP_finetuned)))
                print('board_template_inside:{}'.format(np.shape(board_template_inside)))
                print('closest_lidar_points:{}'.format(np.shape(closest_lidar_points)))
                print('Camera_XYZ_Stereo:{}'.format(np.shape(Camera_XYZ_Stereo)))


            ax.scatter(*board_template.T, color='b', marker='o', alpha=.5, s=8)
            ax.scatter(*board_template_ICP_finetuned.T, color='r', marker='o', alpha=.5, s=8)
            ax.scatter(*board_template_inside.T, color='tab:blue', marker='x', alpha=1, s=10)
            ax.scatter(*closest_lidar_points.T, color='r', marker='x', alpha=.8, s=10)
            ax.scatter(*Camera_XYZ_Stereo.T, color='r', marker='o', alpha=.8, s=20)


            self.AnnotateEdges(giveAX=ax, givenPoints=board_template_inside)

            extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
            sz = extents[:, 1] - extents[:, 0]
            centers = np.mean(board_template, axis=0)
            # centers = np.mean(Camera_XYZ, axis=0) if self.img_file2 is not None else np.mean(board_template, axis=0)

            maxsize = max(abs(sz))
            r = maxsize / 2
            for ctr, dim in zip(centers, 'xyz'):
                getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

            self.pixelsPointsLeft, self.pixelsPointsRight, _3DreconstructedBoard = self.getImagePixels()
            _3DreconstructedBoard = np.array(_3DreconstructedBoard).squeeze()
            print('_3DreconstructedBoard -> {}'.format(np.shape(_3DreconstructedBoard)))
            if len(self.pixelsPointsLeft) <= 0:
                print('Cannot get pixels points !!! ')
            ax.scatter(*_3DreconstructedBoard.T, color='b', marker='x', alpha=1, s=20)
            print('pixelsPointsLeft:{}'.format(np.shape(self.pixelsPointsLeft)))
            print('pixelsPointsRight:{}'.format(np.shape(self.pixelsPointsRight)))
            print('_3DreconstructedBoard:{}'.format(np.shape(_3DreconstructedBoard)))
            self.points_correspondences = dict([
                ('board_template', board_template),
                ('board_template_ICP_finetuned', board_template_ICP_finetuned),
                ('board_template_inside', board_template_inside),
                ('pixelsPointsLeft', self.pixelsPointsLeft),
                ('pixelsPointsRight', self.pixelsPointsRight),
                ('_3DreconstructedBoard',_3DreconstructedBoard),
                ('Camera_XYZ_Stereo', Camera_XYZ_Stereo),
                ('closest_lidar_points', closest_lidar_points)])

            # save_obj(self.points_correspondences, self.name)

        ax.legend(handles=legend_elements, loc='best')
        plt.show()

    def getDepth_Inside_Outside(self):
        calibrations = ['inside', 'outside']
        output = []
        for calib in calibrations:
            camera_model = load_obj('{}_combined_camera_model'.format(calib))
            camera_model_rectify = load_obj('{}_combined_camera_model_rectify'.format(calib))

            K_left = camera_model['K_right']
            D_left = camera_model['D_right']
            T = camera_model['T']
            leftMapX, leftMapY = camera_model_rectify['leftMapX'], camera_model_rectify['leftMapY']
            rightMapX, rightMapY = camera_model_rectify['rightMapX'], camera_model_rectify['rightMapY']

            imgleft = cv2.imread(self.img_file)
            imgright = cv2.imread(self.img_file2)

            if stereoRectify:
                imgleft = cv2.remap(src=imgleft, map1=leftMapX, map2=leftMapY, interpolation=cv2.INTER_LINEAR, dst=None,borderMode=cv2.BORDER_CONSTANT)
                imgright = cv2.remap(src=imgright, map1=rightMapX, map2=rightMapY, interpolation=cv2.INTER_LINEAR, dst=None,borderMode=cv2.BORDER_CONSTANT)

            gray_left = cv2.cvtColor(imgleft, cv2.COLOR_BGR2GRAY)
            ret_left, corners_left = cv2.findChessboardCorners(gray_left, (10, 7), None)
            gray_right = cv2.cvtColor(imgright, cv2.COLOR_BGR2GRAY)
            ret_right, corners_right = cv2.findChessboardCorners(gray_right, (10, 7), None)

            if ret_left and ret_right:  # found chessboard
                corners2_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), self.criteria)
                x_left = np.asarray(corners2_left).squeeze()

                corners2_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), self.criteria)
                x_right = np.asarray(corners2_right).squeeze()

                baseline = abs(T[0])
                focal_length, cx, cy = K_left[0, 0], K_left[0, 2], K_left[1, 2]
                disparity = np.sum(np.sqrt((x_left - x_right) ** 2), axis=1)
                # depth = baseline (meter) * focal length (pixel) / disparity-value (pixel) -> meter
                depth = (baseline * focal_length / disparity)  # .reshape(10,7)
                fxypxy = [K_left[0, 0], K_left[1, 1], cx, cy]
                print('{}  fx:{}, fy:{}'.format(calib, round(K_left[0, 0],2), round(K_left[1, 1],2)))
                _3DPoints = []
                for i, pixel in enumerate(x_left):
                    u, v = pixel.ravel()
                    u, v = int(u), int(v)
                    distance = depth[i]
                    # print('u:{},v:{},distance:{}'.format(u,v, distance))
                    pt = np.array([u, v, distance])

                    pt[0] = pt[2] * (pt[0] - fxypxy[2]) / fxypxy[0]
                    pt[1] = pt[2] * (pt[1] - fxypxy[3]) / fxypxy[1]

                    _3DPoints.append(pt)

                _3DPoints = np.array(_3DPoints)
                output.append(_3DPoints)
            else:
                print('cannot detect board in both images')

        if len(output)>1:
            inside_3D = np.array(output[0]).squeeze()
            outisde_3D = np.array(output[1]).squeeze()

            #get the error for each point
            a_min_b = inside_3D - outisde_3D
            norm_total = np.linalg.norm(a_min_b)/70
            norm_axis = np.linalg.norm(a_min_b, axis=0)/70
            print('norm_total:{}, norm_axis:{}'.format(norm_total,norm_axis))
            self._3DErros.append(norm_axis)

    def fitNewPlan(self):
        coolPoints = self.coolPoints
        def minimum_bounding_rectangle(points):
            pi2 = np.pi / 2.
            # get the convex hull for the points
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            y_saved = []
            for simplex in hull.simplices:
                y = coolPoints[simplex,1]
                x = points[simplex, 0]
                z = points[simplex, 1]
                self.ax.plot(x, y, z, 'k-', alpha = .5)
                y_saved.append(y)
            y_saved = np.array(y_saved)
            # calculate edge angles
            edges = hull_points[1:] - hull_points[:-1]
            angles = np.arctan2(edges[:, 1], edges[:, 0])

            angles = np.abs(np.mod(angles, pi2))
            angles = np.unique(angles)

            rotations = np.vstack([
                np.cos(angles),np.cos(angles - pi2),
                np.cos(angles + pi2),np.cos(angles)]).T
            rotations = rotations.reshape((-1, 2, 2))
            # apply rotations to the hull
            rot_points = np.dot(rotations, hull_points.T)

            # find the bounding points
            min_x = np.nanmin(rot_points[:, 0], axis=1)
            max_x = np.nanmax(rot_points[:, 0], axis=1)
            min_y = np.nanmin(rot_points[:, 1], axis=1)
            max_y = np.nanmax(rot_points[:, 1], axis=1)

            # find the box with the best area
            areas = (max_x - min_x) * (max_y - min_y)
            best_idx = np.argmin(areas)
            # return the best box
            x1 = max_x[best_idx]
            x2 = min_x[best_idx]
            y1 = max_y[best_idx]
            y2 = min_y[best_idx]
            r = rotations[best_idx]

            rval = np.zeros((4, 2))
            rval[0] = np.dot([x1, y2], r)
            rval[1] = np.dot([x2, y2], r)
            rval[2] = np.dot([x2, y1], r)
            rval[3] = np.dot([x1, y1], r)
            rval = np.array(rval)
            d_matrix = distance_matrix(rval, points)
            neighbours = np.argsort(d_matrix, axis=1)[:, 0]
            rval2 = np.asarray(coolPoints[neighbours, 1]).squeeze()

            return rval, rval2

        points = list(self.coolPoints[:, [0, -1]])
        y = np.mean(self.coolPoints[:, 1])
        c, c2 = minimum_bounding_rectangle(np.array(points))
        self.corners_ = []
        for i,point in enumerate(c):
            #self.corners_.append([point[0],y, point[1]])
            self.corners_.append([point[0],c2[i], point[1]])
        if self.chessBoard==False and self.circle_center:
            self.corners_.append([self.circle_center[0],y,self.circle_center[1]])
        self.corners_ = np.array(self.corners_)
        self.ax.scatter(*self.corners_.T, color='k', marker='x', alpha=1, s=50)

    def fitCircle(self, points):
        if len(points)>0:
            def calc_R(x, y, xc, yc):
                """calculate the distance of each 2D points from the center (xc, yc)"""
                return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

            def f(c, x, y):
                """calculate the algebraic distance between the data points
                and the mean circle centered at c=(xc, yc)"""
                Ri = calc_R(x, y, *c)
                return Ri - Ri.mean()

            def sigma(coords, x, y, r):
                """Computes Sigma for circle fit."""
                dx, dy, sum_ = 0., 0., 0.
                for i in range(len(coords)):
                    dx = coords[i][1] - x
                    dy = coords[i][0] - y
                    sum_ += (sqrt(dx * dx + dy * dy) - r) ** 2
                return sqrt(sum_ / len(coords))

            def hyper_fit(coords, IterMax=99, verbose=False):
                """
                Fits coords to circle using hyperfit algorithm.
                Inputs:
                    - coords, list or numpy array with len>2 of the form:
                    [
                [x_coord, y_coord],
                ...,
                [x_coord, y_coord]
                ]
                    or numpy array of shape (n, 2)
                Outputs:
                    - xc : x-coordinate of solution center (float)
                    - yc : y-coordinate of solution center (float)
                    - R : Radius of solution (float)
                    - residu : s, sigma - variance of data wrt solution (float)
                """
                X, Y = None, None
                if isinstance(coords, np.ndarray):
                    X = coords[:, 0]
                    Y = coords[:, 1]
                elif isinstance(coords, list):
                    X = np.array([x[0] for x in coords])
                    Y = np.array([x[1] for x in coords])
                else:
                    raise Exception("Parameter 'coords' is an unsupported type: " + str(type(coords)))

                n = X.shape[0]

                Xi = X - X.mean()
                Yi = Y - Y.mean()
                Zi = Xi * Xi + Yi * Yi

                # compute moments
                Mxy = (Xi * Yi).sum() / n
                Mxx = (Xi * Xi).sum() / n
                Myy = (Yi * Yi).sum() / n
                Mxz = (Xi * Zi).sum() / n
                Myz = (Yi * Zi).sum() / n
                Mzz = (Zi * Zi).sum() / n

                # computing the coefficients of characteristic polynomial
                Mz = Mxx + Myy
                Cov_xy = Mxx * Myy - Mxy * Mxy
                Var_z = Mzz - Mz * Mz

                A2 = 4 * Cov_xy - 3 * Mz * Mz - Mzz
                A1 = Var_z * Mz + 4. * Cov_xy * Mz - Mxz * Mxz - Myz * Myz
                A0 = Mxz * (Mxz * Myy - Myz * Mxy) + Myz * (Myz * Mxx - Mxz * Mxy) - Var_z * Cov_xy
                A22 = A2 + A2

                # finding the root of the characteristic polynomial
                y = A0
                x = 0.
                for i in range(IterMax):
                    Dy = A1 + x * (A22 + 16. * x * x)
                    xnew = x - y / Dy
                    if xnew == x or not np.isfinite(xnew):
                        break
                    ynew = A0 + xnew * (A1 + xnew * (A2 + 4. * xnew * xnew))
                    if abs(ynew) >= abs(y):
                        break
                    x, y = xnew, ynew

                det = x * x - x * Mz + Cov_xy
                Xcenter = (Mxz * (Myy - x) - Myz * Mxy) / det / 2.
                Ycenter = (Myz * (Mxx - x) - Mxz * Mxy) / det / 2.

                x = Xcenter + X.mean()
                y = Ycenter + Y.mean()
                r = sqrt(abs(Xcenter ** 2 + Ycenter ** 2 + Mz))
                s = sigma(coords, x, y, r)
                iter_ = i
                if verbose:
                    print('Regression complete in {} iterations.'.format(iter_))
                    print('Sigma computed: ', s)
                return x, y, r, s

            def least_squares_circle(coords):
                """Circle fit using least-squares solver.
                Inputs:
                    - coords, list or numpy array with len>2 of the form:
                    [
                [x_coord, y_coord],
                ...,
                [x_coord, y_coord]
                ]
                    or numpy array of shape (n, 2)
                Outputs:
                    - xc : x-coordinate of solution center (float)
                    - yc : y-coordinate of solution center (float)
                    - R : Radius of solution (float)
                    - residu : MSE of solution against training data (float)
                """

                x, y = None, None
                if isinstance(coords, np.ndarray):
                    x = coords[:, 0]
                    y = coords[:, 1]
                elif isinstance(coords, list):
                    x = np.array([point[0] for point in coords])
                    y = np.array([point[1] for point in coords])
                else:
                    raise Exception("Parameter 'coords' is an unsupported type: " + str(type(coords)))

                # coordinates of the barycenter
                x_m = np.mean(x)
                y_m = np.mean(y)
                center_estimate = x_m, y_m
                center, _ = leastsq(f, center_estimate, args=(x, y))
                xc, yc = center
                Ri = calc_R(x, y, *center)
                R = Ri.mean()
                residu = np.sum((Ri - R) ** 2)
                return xc, yc, R, residu

            def plot_data_circle(x, y, xc, yc, R):
                """
                Plot data and a fitted circle.
                Inputs:
                    x : data, x values (array)
                    y : data, y values (array)
                    xc : fit circle center (x-value) (float)
                    yc : fit circle center (y-value) (float)
                    R : fir circle radius (float)
                Output:
                    None (generates matplotlib plot).
                """
                f = plt.figure(facecolor='white')
                plt.axis('equal')

                theta_fit = np.linspace(-pi, pi, 180)

                x_fit = xc + R * np.cos(theta_fit)
                y_fit = yc + R * np.sin(theta_fit)
                plt.plot(x_fit, y_fit, 'b-', label="fitted circle", lw=2)
                plt.plot([xc], [yc], 'bD', mec='y', mew=1)
                plt.xlabel('x')
                plt.ylabel('y')
                # plot data
                plt.scatter(x, y, c='red', label='data')

                plt.legend(loc='best', labelspacing=0.1)
                plt.grid()
                plt.title('Fit Circle')

            x1, y1, r1, resid1 = hyper_fit(points[:,[0,2]])
            x2, y2, r2, resid2 = least_squares_circle(points[:,[0,2]])
            #plot_data_circle(points[:,1], points[:,2],x,y,r)
            if resid1>resid2:
                x, y, r = x2, y2, r2
            else:
                x, y, r = x1, y1, r1

            self.circle_center = (x, y)
            self.circle_radius = r

def getData(chess=True):
    pcl_files = glob.glob('/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/data/{}/*.npy'.format('chess' if chess else 'charuco'))
    imgleft_files = glob.glob('/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/data/{}/left/*.png'.format('chess' if chess else 'charuco'))
    imgright_files = glob.glob('/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/data/{}/right/*.png'.format('chess' if chess else 'charuco'))
    pcl_files.sort()
    imgleft_files.sort()
    imgright_files.sort()

    GoodPoints,_3DErros, IMageNames = [],[],[]
    for i, file in enumerate(pcl_files):
        if globalTrigger:
            print('work with {}'.format(file))
            image_left = imgleft_files[i]
            image_right = imgright_files[i]
            filt = PointCloud_filter(file=file, img_file=image_left, img_file2=image_right, debug=False)
            filt.setUp()

            plt.show()
            plt.close()
            print('\n OK:{},  Save points_correspondences : {}'.format(filt.OK, np.shape(filt.points_correspondences)))
            if filt.OK:
                GoodPoints.append(filt.points_correspondences)
                print('save data {} '.format(np.shape(GoodPoints)))
                _3DErros.append(filt._3DErros)
                IMageNames.append(os.path.basename(image_left))
        else:
            print('Close')
            break

    save_obj(GoodPoints, 'GoodPoints2_{}'.format('chess' if chess else 'charuco'))
    print('Data saved in GoodPoints')
    showErros(_3DErros, IMageNames)

def euler_from_matrix(R):
    beta = -np.arcsin(R[2, 0])
    alpha = np.arctan2(R[2, 1] / np.cos(beta), R[2, 2] / np.cos(beta))
    gamma = np.arctan2(R[1, 0] / np.cos(beta), R[0, 0] / np.cos(beta))
    return np.array((alpha, beta, gamma))

def euler_matrix(theta):
    R = np.array([[np.cos(theta[1]) * np.cos(theta[2]),
                   np.sin(theta[0]) * np.sin(theta[1]) * np.cos(theta[2]) - np.sin(theta[2]) * np.cos(theta[0]),
                   np.sin(theta[1]) * np.cos(theta[0]) * np.cos(theta[2]) + np.sin(theta[0]) * np.sin(
                       theta[2])],
                  [np.sin(theta[2]) * np.cos(theta[1]),
                   np.sin(theta[0]) * np.sin(theta[1]) * np.sin(theta[2]) + np.cos(theta[0]) * np.cos(theta[2]),
                   np.sin(theta[1]) * np.sin(theta[2]) * np.cos(theta[0]) - np.sin(theta[0]) * np.cos(
                       theta[2])],
                  [-np.sin(theta[1]), np.sin(theta[0]) * np.cos(theta[1]),
                   np.cos(theta[0]) * np.cos(theta[1])]])

    return R

class LiDAR_Camera_Calibration(object):
    def __init__(self, file, chess = True, debug=True):
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
        self.objp = np.zeros((7 * 10, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:10, 0:7].T.reshape(-1, 2) * .1
        self.debug = debug
        self.file = file
        self.chess = chess

        if chess:
            self.data_key = ['board_template','board_template_ICP_finetuned','board_template_inside',
                             'icp_finetuned_inside','closest_lidar_points','closest_lidar_points_inside',
                             'pixelsPoints','Camera_XYZ_Stereo','Camera_XYZ']

        else:
            self.data_key = ['board_template','board_template_ICP_finetuned','board_template_inside','pixelsPoints',
                             'Camera_XYZ_Stereo','closest_lidar_points']

        self.readIntrinsics()
        self.load_points()

        '''self.Rotation = np.array([[ 0.94901505,  0.01681284,  0.3147821 ],
                                 [-0.01003801,  0.99968204, -0.02313113],
                                 [-0.31507091,  0.018792,    0.94888207]]).squeeze()
        self.Translation = np.array([[-0.98078971],
                                     [ 0.00600202],
                                     [ 0.19497569]]).squeeze()
        #self.Translation[0] = -.64

        euler = euler_from_matrix(self.Rotation)
        # print('euler1->{}'.format(euler))
        angles = euler_from_matrix(self.Rotation)
        print('rotation1: ', [(180.0 / math.pi) * i for i in angles])
        euler[1] = np.deg2rad(22.598)
        self.Rotation = euler_matrix(euler)'''

    def rmse(self, objp, imgp, K, D, rvec, tvec):
        print('objp:{}, imgp:{}'.format(np.shape(objp), np.shape(imgp)))
        predicted, _ = cv2.projectPoints(objp, rvec, tvec, K, D)
        predicted = cv2.undistortPoints(predicted, K, D, P=K)
        predicted = predicted.squeeze()

        pix_serr = []
        for i in range(len(predicted)):
            xp = predicted[i, 0]
            yp = predicted[i, 1]
            xo = imgp[i, 0]
            yo = imgp[i, 1]
            pix_serr.append((xp - xo) ** 2 + (yp - yo) ** 2)
        ssum = sum(pix_serr)
        return math.sqrt(ssum / len(pix_serr))

    def readIntrinsics(self):
        name = 'inside'
        name = 'outside'
        self.camera_model = load_obj('{}_combined_camera_model'.format(name))
        self.camera_model_rectify = load_obj('{}_combined_camera_model_rectify'.format(name))

        self.K_right = self.camera_model['K_left']
        self.K_left = self.camera_model['K_right']
        self.D_right = self.camera_model['D_left']
        self.D_left = self.camera_model['D_right']
        print(' self.K_right')
        print( self.K_right)
        print(' self.K_left')
        print(self.K_left)
        self.R = self.camera_model['R']
        self.T = self.camera_model['T']

        self.K = self.K_right
        self.D = self.D_right

        print('self T before {}'.format(np.shape(self.T)))
        #self.T = np.array([-0.98, 0., 0.12])[:,np.newaxis]
        self.T = np.array([-0.96, 0., 0.12])[:, np.newaxis]
        print('self T after {}'.format(np.shape(self.T)))
        angles = np.array([np.deg2rad(0.68), np.deg2rad(22.66), np.deg2rad(-1.05)])
        self.R = euler_matrix(angles)
        #print(self.R)
        print('translation is {}-----------------------------'.format(self.T))

        img_shape = (1936, 1216)
        print('img_shape:{}'.format(img_shape))
        R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(self.K_left, self.D_left, self.K_right, self.D_right,
                                                                   imageSize=img_shape,
                                                                   R=self.camera_model['R'], T=self.camera_model['T'],
                                                                   flags=cv2.CALIB_ZERO_DISPARITY,
                                                                   alpha=-1
                                                                   #alpha=0
                                                                   )
        #print('R1:{}'.format(R1))
        #print('R2:{}'.format(R2))

        # print('euler1->{}'.format(euler))
        angles = euler_from_matrix(self.R)
        print('self.R: ', [(180.0 / math.pi) * i for i in angles])

        euler = euler_from_matrix(R1)
        #print('euler1->{}'.format(euler))
        angles = euler_from_matrix(R1)
        #print('rotation1: ', [(180.0 / math.pi) * i for i in angles])

        euler = euler_from_matrix(R2)
        #print('euler2->{}'.format(euler))
        angles = euler_from_matrix(R2)
        #print('rotation2: ', [(180.0 / math.pi) * i for i in angles])
        self.R1 = R1
        self.R2 = R2
        self.P1 = P1
        self.leftMapX, self.leftMapY = cv2.initUndistortRectifyMap(
            self.K_left, self.D_left, R1,
            P1, img_shape, cv2.CV_32FC1)

        self.rightMapX, self.rightMapY = cv2.initUndistortRectifyMap(
            self.K_right, self.D_right, R2,
            P2, img_shape, cv2.CV_32FC1)

        print('Got camera intrinsic')
        print('Got camera-lidar extrinsics')

    def load_points(self):
        self.Lidar_3D, self.Image_2D,self.Image_2D2, self.Image_3D,self.Camera_XYZ = [],[],[],[],[]
        with open(self.file,'rb') as f:
            self.dataPoinst = pickle.load(f)

        self.N = len(self.dataPoinst)
        print('Got {} data views'.format(self.N))
        #self.N = 1

        for i in range(self.N):
            try:
                dictionary_data = self.dataPoinst[i]
                LiDAR_3D_points = dictionary_data['board_template_inside'] #N x 3

                #pixelsPoints = dictionary_data['pixelsPoints']             #N x 2
                #StereoCam_3D_points = dictionary_data['Camera_XYZ_Stereo'] #N x 3

                pixelsPointsLeft = dictionary_data['pixelsPointsLeft']
                pixelsPointsRight = dictionary_data['pixelsPointsRight']
                StereoCam_3D_points = dictionary_data['_3DreconstructedBoard'] #N x 3


                self.Lidar_3D.append(LiDAR_3D_points)
                self.Image_2D.append(pixelsPointsLeft)
                self.Image_2D2.append(pixelsPointsRight)
                self.Image_3D.append(StereoCam_3D_points)

                if self.chess:
                    self.Camera_XYZ.append(dictionary_data['Camera_XYZ'])
            except:
                #print('Cannot read data')
                pass



        #self.Lidar_3D = np.array(self.Lidar_3D).reshape(-1,3)
        #self.Image_2D = np.array(self.Image_2D).reshape(-1,2)
        #self.Image_3D = np.array( self.Image_3D).reshape(-1,3)
        print('Lidar_3D:{}, Image_2D:{}, Image_2D2:{}, Image_3D:{}'.format(np.shape(self.Lidar_3D),
                                                                                   np.shape(self.Image_2D),np.shape(self.Image_2D2),
                                                                                   np.shape(self.Image_3D)))

    def plotData(self):
        self.fig = plt.figure(figsize=plt.figaspect(0.33))
        self.fig.tight_layout()
        for i in range(self.N):
            print('{}/{}'.format(i+1,self.N))
            ax1 = self.fig.add_subplot(1, 3, 1, projection='3d')
            #ax1.set_title('3D LiDAR')
            ax1.set_xlabel('X', fontsize=8)
            ax1.set_ylabel('Y', fontsize=8)
            ax1.set_zlabel('Z', fontsize=8)


            ax2 = self.fig.add_subplot(1, 3, 2, projection='3d')
            ax2.set_title('3D Stereo cameras')
            ax2.set_xlabel('X', fontsize=8)
            ax2.set_ylabel('Y', fontsize=8)
            ax2.set_zlabel('Z', fontsize=8)


            ax3 = self.fig.add_subplot(1, 3, 3, projection='3d')
            ax3.set_title('2D pixels')
            ax3.set_xlabel('X', fontsize=8)
            ax3.set_ylabel('Y', fontsize=8)
            ax3.set_zlabel('Z', fontsize=8)


            _3d_LIDAR = np.array(self.Lidar_3D[i])
            ax1.scatter(*_3d_LIDAR.T)
            self.axisEqual3D(ax1, _3d_LIDAR)

            _3d_cam = np.array(self.Image_3D[i])
            ax2.scatter(*_3d_cam.T, c='r')
            self.axisEqual3D(ax2,_3d_cam)

            _2d_cam = np.array(self.Image_2D[i])
            ax3.scatter(*_2d_cam.T, c='g')
            self.axisEqual3D(ax3, _2d_cam)

            plt.show()

    def axisEqual3D(self,ax,data):
        extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
        sz = extents[:, 1] - extents[:, 0]
        centers = np.mean(data, axis=0)
        maxsize = max(abs(sz))
        r = maxsize / 2
        for ctr, dim in zip(centers, 'xyz'):
            getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

    def get3D_3D_homography(self, src, dst):  #both or Nx3 matrices
        src_mean = np.mean(src, axis=0)
        dst_mean = np.mean(dst, axis=0)
        # Compute covariance
        try:
            H = reduce(lambda s, (a, b): s + np.outer(a, b), zip(src - src_mean, dst - dst_mean), np.zeros((3, 3)))
            u, s, v = np.linalg.svd(H)
            R = v.T.dot(u.T)  # Rotation
            T = - R.dot(src_mean) + dst_mean  # Translation
            H = np.hstack((R, T[:, np.newaxis]))
            return H,R.T,T
        except:
            print('switch to python 2')

    def calibrate_3D_3D_old(self):
        print('3D-3D ========================================================================================')
        file = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/data/GoodPoints_3D3D_{}.pkl'.format('chess')
        file = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/data/GoodPoints_{}.pkl'.format('chess')

        self.Lidar_3D, self.Image_2D, self.Image_3D, self.Camera_XYZ = [], [], [], []
        with open(file, 'rb') as f:
            self.dataPoinst = pickle.load(f)

        self.N = len(self.dataPoinst)
        print('Got {} data views'.format(self.N))
        for i in range(self.N):
            try:
                dictionary_data = self.dataPoinst[i]
                LiDAR_3D_points = dictionary_data['board_template_inside']  # N x 3
                pixelsPoints = dictionary_data['pixelsPoints']  # N x 2
                StereoCam_3D_points = dictionary_data['Camera_XYZ_Stereo']  # N x 3
                #StereoCam_3D_points = dictionary_data['point3D_trianguate']
                self.Lidar_3D.append(LiDAR_3D_points)
                self.Image_2D.append(pixelsPoints)
                self.Image_3D.append(StereoCam_3D_points)

                if self.chess:
                    self.Camera_XYZ.append(dictionary_data['Camera_XYZ'])
            except:
                print('Cannot read data===================================================')
                break

        print('Lidar_3D:{}, Image_2D:{}, Image_3D:{}'.format(np.shape(self.Lidar_3D),
                                                             np.shape(self.Image_2D),
                                                             np.shape(self.Image_3D)))

        Lidar_3D = np.array(self.Lidar_3D).reshape(-1, 3)
        Image_3D = np.array( self.Image_3D).reshape(-1,3)
        print('Lidar_3D:{}, Image_3D:{}'.format(np.shape(Lidar_3D),np.shape(Image_3D)))
        #-------------------------------------#-------------------------------------
        c_, R_, t_ = self.estimate(Lidar_3D,Image_3D)

        #import superpose3d as super
        #(RMSD, R_, t_, c_) = super.Superpose3D(Lidar_3D, Image_3D)
        #print('RMSD -> {}, t_{},  c_->{}'.format(RMSD, t_, c_))
        # -------------------------------------#-------------------------------------

        def similarity_transform(from_points, to_points):
            assert len(from_points.shape) == 2, \
                "from_points must be a m x n array"
            assert from_points.shape == to_points.shape, \
                "from_points and to_points must have the same shape"

            N, m = from_points.shape

            mean_from = from_points.mean(axis=0)
            mean_to = to_points.mean(axis=0)

            delta_from = from_points - mean_from  # N x m
            delta_to = to_points - mean_to  # N x m

            sigma_from = (delta_from * delta_from).sum(axis=1).mean()
            sigma_to = (delta_to * delta_to).sum(axis=1).mean()

            cov_matrix = delta_to.T.dot(delta_from) / N

            U, d, V_t = np.linalg.svd(cov_matrix, full_matrices=True)
            cov_rank = np.linalg.matrix_rank(cov_matrix)
            S = np.eye(m)

            if cov_rank >= m - 1 and np.linalg.det(cov_matrix) < 0:
                S[m - 1, m - 1] = -1
            elif cov_rank < m - 1:
                raise ValueError("colinearility detected in covariance matrix:\n{}".format(cov_matrix))

            R = U.dot(S).dot(V_t)
            c = (d * S.diagonal()).sum() / sigma_from
            t = mean_to - c * R.dot(mean_from)
            print('R:{},t:{},c:{}'.format(R,t,c))
            return c * R, t
        print('similarity_transform===============================')
        from_points = Lidar_3D
        to_points = Image_3D
        M_ans, t_ans = similarity_transform(from_points, to_points)

        H, R, T = self.get3D_3D_homography(src = Lidar_3D, dst=Image_3D)
        print('H:{}, R:{}, T:{}'.format(np.shape(H), np.shape(R), np.shape(T)))
        print(H)
        self.fig = plt.figure(figsize=plt.figaspect(1.))
        ax1 = self.fig.add_subplot(1, 1, 1, projection='3d')
        #ax1.set_title('3D LiDAR')
        ax1.set_xlabel('X', fontsize=8)
        ax1.set_ylabel('Y', fontsize=8)
        ax1.set_zlabel('Z', fontsize=8)
        ax1.set_axis_off()
        _3d_LIDAR = self.Lidar_3D[0]
        ax1.scatter(*_3d_LIDAR.T, label = 'LiDAR')
        _3d_Image = self.Image_3D[0]
        ax1.scatter(*_3d_Image.T, s=25, label = 'Stereo Cam')
        T = _3d_LIDAR.dot(c_ * R_) + t_
        print('T -> {}'.format(np.shape(T)))
        ax1.scatter(*T.T, marker='x', label='T')

        d2 = distance_matrix(_3d_Image,_3d_Image)
        print('d2:{}'.format(d2))
        print('d2 shape :{}'.format(np.shape(d2)))

        ones = np.ones(len(_3d_LIDAR))[:, np.newaxis]
        transformed_ = np.hstack((_3d_LIDAR,ones))
        transformed = np.dot(H, transformed_.T).T #transformation estimated with SVD

        print(np.shape(transformed))
        ax1.scatter(*transformed.T, s=25, label = 'ICP sol')
        #ax1.set_axis_off()
        primary = Lidar_3D# _3d_LIDAR
        secondary = Image_3D#    _3d_Image

        pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
        unpad = lambda x: x[:, :-1]
        X = pad(primary)
        Y = pad(secondary)

        # Solve the least squares problem X * A = Y # to find our transformation matrix A
        A, res, rank, s = np.linalg.lstsq(X, Y)
        transform = lambda x: unpad(np.dot(pad(x), A))

        #print transform(primary)
        print("Max error:", np.abs(secondary - transform(primary)).max())

        trns2 = transform(_3d_LIDAR) #transformation estimated with LS
        ax1.scatter(*trns2.T, label = 'least square sol')

        to_points = M_ans.dot(_3d_LIDAR.T).T + t_ans
        print('to_points ->{}'.format(np.shape(to_points)))
        ax1.scatter(*to_points.T, label = 'to_points')

        self.axisEqual3D(ax1, transformed)
        ax1.legend()
        plt.show()

        #----------------------------------
        if True:
            img = cv2.imread('/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/data/chess/left/left_4.png')
            img2 = cv2.imread('/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/data/chess/right/right_4.png')
            cloud_file = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/data/chess/cloud_4.npy'
        else:
            img = cv2.imread('/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/data/charuco/left/left_4.png')
            img2 = cv2.imread('/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/data/charuco/right/right_4.png')
            cloud_file = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/data/charuco/cloud_4.npy'

        i = 12
        l = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/cool/left_{}.png'.format(i)
        r = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/cool/right_{}.png'.format(i)
        #img, img2 = cv2.imread(l), cv2.imread(r)
        #cloud_file = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/cool/cloud_{}.npy'.format(i)

        if stereoRectify and True:
            img = cv2.remap(src=img, map1=self.leftMapX, map2=self.leftMapY,
                                      interpolation=cv2.INTER_LINEAR, dst=None, borderMode=cv2.BORDER_CONSTANT)
            img2 = cv2.remap(src=img2, map1=self.rightMapX, map2=self.rightMapY,
                            interpolation=cv2.INTER_LINEAR, dst=None, borderMode=cv2.BORDER_CONSTANT)

        #Points in LiDAR frame
        LiDAR_points3D = np.array(np.load(cloud_file, mmap_mode='r'), dtype=np.float32)[:, :3] #
        print('LiDAR_points3D:{}'.format(np.shape(LiDAR_points3D)))
        #converted in camera frame

        ones = np.ones(len(LiDAR_points3D))[:, np.newaxis]
        transformed_ = np.hstack((LiDAR_points3D, ones))
        Camera_points3D = np.dot(H, transformed_.T).T

        #Camera_points3D = transform(LiDAR_points3D)
        #print('Camera_points3D:{}'.format(np.shape(Camera_points3D)))

        #Camera_points3D = LiDAR_points3D.dot(c_ * R_) + t_
        #Camera_points3D = LiDAR_points3D.dot(R_) + t_
        #Camera_points3D = transform(LiDAR_points3D) #transformation estimated with LS
        print('Camera_points3D -> {}'.format(Camera_points3D))
        rvec, _ = cv2.Rodrigues(np.eye(3))
        tvec = np.zeros(3)

        #Camera_points3D = LiDAR_points3D#.dot(R_) + t_
        #rvec = R_
        #tran = t_
        #tran[0] = -0.02
        #tran[1] = -0.03
        print('rvec -> {}, tvec->{}'.format(np.shape(rvec),np.shape(tvec)))
        print('Camera_points3D -> {}'.format(np.shape(Camera_points3D)))
        # Reproject back into the two cameras
        rvec1, _ = cv2.Rodrigues(np.eye(3).T)  # Change
        rvec2, _ = cv2.Rodrigues(self.R.T)  # Change

        t1 = np.array([[0.], [0.], [0.]])
        t2 = self.T

        p1, _ = cv2.projectPoints(Camera_points3D[:, :3], rvec1, -t1, self.K, distCoeffs=self.D)  # Change
        p2, _ = cv2.projectPoints(Camera_points3D[:, :3], rvec2, -t2, self.K, distCoeffs=self.D)  # Change

        #points2D = [cv2.projectPoints(point, rvec, tvec, self.K, self.D)[0] for point in Camera_points3D[:, :3]]
        points2D, _ = cv2.projectPoints(Camera_points3D[:, :3], np.identity(3), np.array([0., 0., 0.]), self.K, self.D)
        points2D = np.asarray(points2D).squeeze()
        points2D = np.asarray(p1).squeeze()
        print('points2D:{},  img.shape[1]:{}'.format(np.shape(points2D), img.shape[1]))

        inrange = np.where(
            (points2D[:, 0] >= 0) &
            (points2D[:, 1] >= 0) &
            (points2D[:, 0] < img.shape[1]) &
            (points2D[:, 1] < img.shape[0])
        )
        points2D = points2D[inrange[0]].round().astype('int')
        # Draw the projected 2D points
        for i in range(len(points2D)):
            cv2.circle(img, tuple(points2D[i]), 2, (0, 255, 0), -1)
            #cv2.circle(img2, tuple(points2D[i]), 2, (0, 255, 0), -1)

        print('rvec -> {}, tvec->{}'.format(np.shape(rvec),np.shape(tvec)))
        T_01 = np.vstack((np.hstack((np.eye(3), tvec[:,np.newaxis])), [0, 0, 0, 1]))  # from lidar to right camera
        T_12 = np.vstack((np.hstack((self.R, self.T)), [0, 0, 0, 1]))  # between cameras
        T_final = np.dot(T_01,T_12)
        rotation, translation = T_final[:3, :3], T_final[:3, -1]

        points2D = [cv2.projectPoints(point, rotation, translation, self.K, self.D)[0] for point in Camera_points3D[:, :3]]
        points2D = np.asarray(points2D).squeeze()
        points2D = np.asarray(p2).squeeze()
        print('points2D:{},  img.shape[1]:{}'.format(np.shape(points2D), img.shape[1]))

        inrange = np.where(
            (points2D[:, 0] >= 0) &
            (points2D[:, 1] >= 0) &
            (points2D[:, 0] < img.shape[1]) &
            (points2D[:, 1] < img.shape[0])
        )
        points2D = points2D[inrange[0]].round().astype('int')
        # Draw the projected 2D points
        for i in range(len(points2D)):
            cv2.circle(img2, tuple(points2D[i]), 2, (0, 255, 0), -1)


        cv2.imshow('left', cv2.resize(img,None, fx=.4, fy=.4))
        cv2.imshow('right', cv2.resize(img2, None, fx=.4, fy=.4))
        cv2.waitKey()
        cv2.destroyAllWindows()

    def drawCharuco(self, QueryImg):
        points2D = np.array(self.Image_2D[0]).reshape(-1, 2)
        for p in points2D:
            cv2.circle(QueryImg, tuple(p), 4, (0, 0, 255), 5)

        return QueryImg

    def calibrate_3D_2D(self, userRansac = False):
        points3D = np.array(self.Lidar_3D).reshape(-1, 3)
        points2D = np.array(self.Image_2D).reshape(-1,2)
        print('points3D:{}, points2D:{}'.format(np.shape(points3D),np.shape(points2D)))

        # Estimate extrinsics
        if userRansac:
            success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(points3D,
                                                                                       points2D, self.K, self.D,
                                                                                       flags=cv2.SOLVEPNP_ITERATIVE)
            print('success:{},rotation_vector:{},translation_vector:{},inliers:{}'.format(success, np.shape(rotation_vector),
                                                                                          np.shape(translation_vector), np.shape(inliers)))
            # Compute re-projection error.
            points2D_reproj = cv2.projectPoints(points3D, rotation_vector,
                                                translation_vector, self.K, self.D)[0].squeeze(1)

            error = (points2D_reproj - points2D)[inliers]  # Compute error only over inliers.
            error = np.asarray(error).squeeze()
            print('points2D_reproj:{}, points2D:{},error:{}'.format(np.shape(points2D_reproj), np.shape(points2D), np.shape(error)))
            rmse = np.sqrt(np.mean(error[:, 0] ** 2 + error[:, 1] ** 2))
            print('Re-projection error before LM refinement (RMSE) in px: ' + str(rmse))

            # Refine estimate using LM
            if not success:
                print('Initial estimation unsuccessful, skipping refinement')
            elif not hasattr(cv2, 'solvePnPRefineLM'):
                print('solvePnPRefineLM requires OpenCV >= 4.1.1, skipping refinement')
            else:
                assert len(inliers) >= 3, 'LM refinement requires at least 3 inlier points'
                rotation_vector, translation_vector = cv2.solvePnPRefineLM(points3D[inliers],
                                                                           points2D[inliers], self.K, self.D,
                                                                           rotation_vector, translation_vector)


                # Compute re-projection error.
                points2D_reproj = cv2.projectPoints(points3D, rotation_vector,
                                                    translation_vector, self.K, self.D)[0].squeeze(1)
                assert (points2D_reproj.shape == points2D.shape)
                error = (points2D_reproj - points2D)[inliers]  # Compute error only over inliers.
                error = np.array(error).squeeze()
                rmse = np.sqrt(np.mean(error[:, 0] ** 2 + error[:, 1] ** 2))
                print('Re-projection error after LM refinement (RMSE) in px: ' + str(rmse))

            # Convert rotation vector
            #from tf.transformations import euler_from_matrix
            rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
            euler = euler_from_matrix(rotation_matrix)

            # Save extrinsics
            np.savez('extrinsics{}.npz'.format('chess' if self.chess else 'charuco'),euler=euler,Rodrigues=rotation_matrix, R=rotation_vector, T=translation_vector)

            # Display results
            print('Euler angles (RPY):', euler)
            print('Rotation Matrix Rodrigues :', rotation_matrix)
            print('rotation_vector:', rotation_vector)
            print('Translation Offsets:', translation_vector)

            points2D = cv2.projectPoints(points3D, rotation_vector, translation_vector, self.K, self.D)[0].squeeze(1)
            print('========points3D:{}, points2D:{}=================================================='.format(np.shape(points3D),np.shape(points2D)))

        else:
            #-------------------------------------------------------------------------------------------------
            imgp = np.array([points2D], dtype=np.float32).squeeze()
            objp = np.array([points3D], dtype=np.float32).squeeze()

            retval, rvec, tvec = cv2.solvePnP(objp, imgp, self.K, self.D, flags=cv2.SOLVEPNP_ITERATIVE)
            rmat, jac = cv2.Rodrigues(rvec)
            q = Quaternion(matrix=rmat)
            print("Transform from camera to laser")
            print("T = ")
            print(tvec)
            print("R = ")
            print(rmat)
            print("Quaternion = ")
            print(q)

            print("RMSE in pixel = %f" % self.rmse(objp, imgp, self.K, self.D, rvec, tvec))
            result_file = 'solvePnP_extrinsics{}.npz'.format('chess' if self.chess else 'charuco')
            with open(result_file, 'w') as f:
                f.write("%f %f %f %f %f %f %f" % (q.x, q.y, q.z, q.w, tvec[0], tvec[1], tvec[2]))

            print("Result output format: qx qy qz qw tx ty tz")
            #refine results
            print('refine results------------------------------------>')
            rvec, tvec = cv2.solvePnPRefineLM(objp,imgp, self.K, self.D, rvec, tvec)
            rmat, jac = cv2.Rodrigues(rvec)
            q = Quaternion(matrix=rmat)
            print("Transform from camera to laser")
            print("T = ")
            print(tvec)
            print("R = ")
            print(rmat)
            print("Quaternion = ")
            print(q)
            print('Euler angles')

            angles = euler_from_matrix(rmat)
            print(angles)
            print('euler angles ', [(180.0 / math.pi) * i for i in angles])

            print("RMSE in pixel = %f" % self.rmse(objp, imgp, self.K, self.D, rvec, tvec))
            result_file = 'refined_solvePnP_extrinsics{}.npz'.format('chess' if self.chess else 'charuco')
            with open(result_file, 'w') as f:
                f.write("%f %f %f %f %f %f %f" % (q.x, q.y, q.z, q.w, tvec[0], tvec[1], tvec[2]))

    def get_z(self, T_cam_world, T_world_pc, K):
        R = T_cam_world[:3, :3]
        t = T_cam_world[:3, 3]
        proj_mat = np.dot(K, np.hstack((R, t[:, np.newaxis])))
        xyz_hom = np.hstack((T_world_pc, np.ones((T_world_pc.shape[0], 1))))
        xy_hom = np.dot(proj_mat, xyz_hom.T).T
        z = xy_hom[:, -1]
        z = np.asarray(z).squeeze()
        return z

    def callback_solvePnP(self, img, cloud_file):
        #init calibraiton
        calib_file = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/solvePnP_extrinsics{}.npz'.format(
            'chess' if self.chess else 'charuco')

        calib_file_ = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/combined_extrinsics{}.npz'
        with open(calib_file, 'r') as f:
            data = f.read().split()
            #print('data:{}'.format(data))
            qx = float(data[0])
            qy = float(data[1])
            qz = float(data[2])
            qw = float(data[3])
            tx = float(data[4])
            ty = float(data[5])
            tz = float(data[6])

        q = Quaternion(qw, qx, qy, qz).transformation_matrix
        q[0, 3] = tx
        q[1, 3] = ty
        q[2, 3] = tz
        print("Extrinsic parameter - camera to laser")
        print(q)
        tvec = q[:3, 3]
        rot_mat = q[:3, :3]
        rvec, _ = cv2.Rodrigues(rot_mat)
        try:
            objPoints = np.array(np.load(cloud_file, mmap_mode='r'), dtype=np.float32)[:, :3]
            print('objPoints:{}'.format(np.shape(objPoints)))
            Z = self.get_z(q, objPoints, self.K)
            objPoints = objPoints[Z > 0]
            #print('objPoints:{}'.format(objPoints))
            img_points, _ = cv2.projectPoints(objPoints, rvec, tvec, self.K, self.D)

            img_points = np.squeeze(img_points)
            for i in range(len(img_points)):
                try:
                    cv2.circle(img, (int(round(img_points[i][0])), int(round(img_points[i][1]))), 3,
                               (0, 255, 0), 1)
                except OverflowError:
                    continue
            if self.chess:
                cv2.drawChessboardCorners(img, (10, 7), np.array(self.Image_2D).reshape(-1,2), True)
            else:
                self.drawCharuco(img)
        except:
            print('callback_solvePnP - error')
        image = cv2.resize(img, None, fx=.6, fy=.6)
        return image

    def callback_solvePnP_Ransac(self, img, cloud_file):
        points3D  = np.array(np.load(cloud_file, mmap_mode='r'), dtype=np.float32)[:, :3]
        print('points3D:{}'.format(np.shape(points3D)))
        file = np.load('extrinsics{}.npz'.format('chess' if self.chess else 'charuco'))

        euler = np.array(file["euler"])
        rotation_matrix = np.array(file["Rodrigues"])
        rotation_vector = np.array(file["R"])
        translation_vector = np.array(file["T"])
        print('Euler angles (RPY):', euler)
        print('Rotation Matrix Rodrigues :', rotation_matrix)
        print('rotation_vector:', rotation_vector)
        print('Translation Offsets:', translation_vector)

        rvec = rotation_matrix

        #rvec, _ = cv2.Rodrigues(rotation_matrix)
        print('========points3D:{}=================================================='.format(
            np.shape(points3D)))

        #points2D = cv2.projectPoints(points3D, rotation_vector, translation_vector, self.K, self.D)[0].squeeze(1)
        #print('points2D:{}'.format(np.shape(points2D)))

        points2D = [cv2.projectPoints(point, rvec, translation_vector, self.K, self.D)[0] for point in points3D[:, :3]]
        points2D = np.asarray(points2D).squeeze()
        print('points2D:{},  img.shape[1]:{}'.format(np.shape(points2D),img.shape[1]))

        inrange = np.where(
                               (points2D[:, 0] >= 0) &
                               (points2D[:, 1] >= 0) &
                               (points2D[:, 0] < img.shape[1]) &
                               (points2D[:, 1] < img.shape[0])
                               )

        points2D = points2D[inrange[0]].round().astype('int')
        # Draw the projected 2D points
        for i in range(len(points2D)):
            cv2.circle(img, tuple(points2D[i]), 2, (0, 255, 0), -1)

        if self.chess:
            cv2.drawChessboardCorners(img, (10, 7), np.array(self.Image_2D).reshape(-1,2), True)
        else:
            self.drawCharuco(img)

        image = cv2.resize(img, None, fx=.6, fy=.6)
        return image

    def callback(self):
        if self.chess:
            img = cv2.imread('/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/data/chess/left/left_0.png')
            cloud_file = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/data/chess/cloud_0.npy'
        else:
            img = cv2.imread('/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/data/charuco/left/left_0.png')
            cloud_file = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/data/charuco/cloud_0.npy'

        #img = cv2.imread('/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/data/charuco/left/left_0.png')
        #cloud_file = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/data/charuco/cloud_0.npy'
        #img = cv2.imread('/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/data/chess/left/left_0.png')
        #cloud_file = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/data/chess/cloud_0.npy'

        #solvePnP_Ransac_image = self.callback_solvePnP_Ransac(img=img.copy(),cloud_file=cloud_file)
        cv2.imshow('solvePnP_Ransac', cv2.resize(img,None,fx=.4,fy=.4))
        cv2.waitKey()

        solvePnP_image = self.callback_solvePnP(img=img.copy(),cloud_file=cloud_file)
        cv2.imshow('solvePnP', solvePnP_image)

        cv2.waitKey()
        cv2.destroyAllWindows()

    def combine_both_boards_and_train(self):
        #get data from chessboard
        name = 'chess'
        self.file = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/data/GoodPoints_{}.pkl'.format(name)
        self.load_points()
        Lidar_3D, Image_2D, Image_3D = np.array(self.Lidar_3D).reshape(-1,3), np.array(self.Image_2D).reshape(-1,2), np.array(self.Image_3D).reshape(-1,3)

        #get data from charuco
        name = 'charuco'
        self.file = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/data/GoodPoints_{}.pkl'.format(name)
        self.load_points()
        Lidar_3D, Image_2D = np.vstack((Lidar_3D, np.array(self.Lidar_3D).reshape(-1,3))), np.vstack((Image_2D, np.array(self.Image_2D).reshape(-1,2)))
        print('Lidar_3D:->{}, Image_2D:->{}'.format(np.shape(Lidar_3D), np.shape(Image_2D)))

        imgp = np.array([Image_2D], dtype=np.float32).squeeze()
        objp = np.array([Lidar_3D], dtype=np.float32).squeeze()

        retval, rvec, tvec = cv2.solvePnP(objp, imgp, self.K, self.D, flags=cv2.SOLVEPNP_ITERATIVE)
        print('tvec -> {}'.format(tvec.ravel()))
        rmat, jac = cv2.Rodrigues(rvec)
        q = Quaternion(matrix=rmat)
        angles = euler_from_matrix(rmat)
        print(angles)
        print('euler angles ', [(180.0 / math.pi) * i for i in angles])
        print("RMSE in pixel = %f" % self.rmse(objp, imgp, self.K, self.D, rvec, tvec))
        result_file = 'combined_extrinsics{}.npz'
        with open(result_file, 'w') as f:
            f.write("%f %f %f %f %f %f %f" % (q.x, q.y, q.z, q.w, tvec[0], tvec[1], tvec[2]))

        print('Combined calibration done!!!')

    def computeTransformation(self):
        i = 5
        l = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/cool/left_{}.png'.format(i)
        r = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/cool/right_{}.png'.format(i)
        img1 = cv2.imread(l)
        img2 = cv2.imread(r)

        #sift = cv2.SIFT_create()
        sift = cv2.xfeatures2d.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        pts1 = []
        pts2 = []
        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.8 * n.distance:
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)

        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        #F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, cv2.RANSAC, 0.999, 1.0, None)

        print(E)
        points, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K)
        print('R')
        print(R)
        angles = euler_from_matrix(R)
        print('rotation angles: ', [(180.0 / math.pi) * i for i in angles])
        print('t')
        print(t)
        for pt1, pt2 in zip(pts1, pts2):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
            img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)

        cv2.imshow('imgL', cv2.resize(img1, None, fx=.4, fy=.4))
        cv2.imshow('imgR', cv2.resize(img2, None, fx=.4, fy=.4))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def write_ply(self, fn, verts, colors):
        ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        '''
        out_colors = colors.copy()
        verts = verts.reshape(-1, 3)
        verts = np.hstack([verts, out_colors])
        with open(fn, 'wb') as f:
            f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
            np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

    def view(self):
        import glob
        import open3d
        file = glob.glob('/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/*.ply')
        pcda = []
        for i, file_path in enumerate(file):
            print("{} Load a ply point cloud, print it, and render it".format(file_path))
            pcd = open3d.io.read_point_cloud(file_path)
            pcda.append(pcd)
            open3d.visualization.draw_geometries([pcd])

        #o3d.visualization.draw_geometries([pcda[1], pcda[-1]])

    def reproject_on_3D(self, useUnique = True):
        def readCalibrationExtrinsic():
            calib_file = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/solvePnP_extrinsics{}.npz'.format(
                'chess' if self.chess else 'charuco')
            calib_file = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/combined_extrinsics{}.npz'
            with open(calib_file, 'r') as f:
                data = f.read().split()
                #print('data:{}'.format(data))
                qx = float(data[0])
                qy = float(data[1])
                qz = float(data[2])
                qw = float(data[3])
                tx = float(data[4])
                ty = float(data[5])
                tz = float(data[6])

            q = Quaternion(qw, qx, qy, qz).transformation_matrix
            q[0, 3],q[1, 3],q[2, 3] = tx,ty,tz
            tvec = q[:3, 3]
            rot_mat = q[:3, :3]
            #rvec, _ = cv2.Rodrigues(rot_mat)
            rvec = rot_mat
            print('tvec -> {}'.format(tvec))

            return rvec, tvec, q

        rvec, tvec, q = readCalibrationExtrinsic()

        i=1
        i=11
        l = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/cool/left_{}.png'.format(i)
        r = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/cool/right_{}.png'.format(i)
        imgLeft, imgRight = cv2.imread(l),cv2.imread(r)
        cloud_file = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/cool/cloud_{}.npy'.format(i)
        _3DPoints = np.array(np.load(cloud_file, mmap_mode='r'), dtype=np.float32)[:, :3]

        #Left image--------------------------------------------------------------------------------------------
        objPoints_left = _3DPoints.copy()
        Z = self.get_z(q, objPoints_left, self.K)
        objPoints_left = objPoints_left[Z > 0]
        print('objPoints_left:{}'.format(np.shape(objPoints_left)))
        points2D_left, _ = cv2.projectPoints(objPoints_left, rvec, tvec, self.K, self.D)
        points2D_left = np.squeeze(points2D_left)
        print('objPoints_left -> {},  points2D_left -> {},  '.format(np.shape(objPoints_left), np.shape(points2D_left)))

        inrange_left = np.where((points2D_left[:, 0] > 0) & (points2D_left[:, 1] > 0) &
            (points2D_left[:, 0] < imgLeft.shape[1]-1) & (points2D_left[:, 1] < imgLeft.shape[0]-1))
        print('inrange_left : {}'.format(np.shape(inrange_left)))
        points2D_left = points2D_left[inrange_left[0]].round().astype('int')
        print('points2D_left:{},  '.format(np.shape(points2D_left)))

        #Right image ----------------------------------------------------------------------------------------
        objPoints_right = _3DPoints.copy()
        Z = self.get_z(q, objPoints_right, self.K_left)
        objPoints_right = objPoints_right[Z > 0]
        T_01 = np.vstack((np.hstack((rvec, tvec[:, np.newaxis])), [0,0,0,1])) #from lidar to right camera
        T_12 = np.vstack((np.hstack((self.R, self.T)), [0,0,0,1]))            #between cameras
        T_final = np.dot(T_12, T_01)
        rotation, translation = T_final[:3,:3], T_final[:3,-1]
        points2D_right, _ = cv2.projectPoints(objPoints_right, rotation, translation, self.K_left, self.D_left)
        points2D_right = np.squeeze(points2D_right)
        inrange_right = np.where((points2D_right[:, 0] >= 0) &(points2D_right[:, 1] >= 0) &
            (points2D_right[:, 0] < imgRight.shape[1]-1) &(points2D_right[:, 1] < imgRight.shape[0]-1))
        print('points2D_right init ->{}'.format(np.shape(points2D_right)))
        points2D_right = points2D_right[inrange_right[0]].round().astype('int')
        print('points2D_right now ->{}'.format(np.shape(points2D_right)))

        #columns=["X", "Y", "Z","intens","ring"]
        colors = np.array(np.load(cloud_file, mmap_mode='r'))[:, 3]  #
        # Color map for the points
        colors = colors[inrange_left[0]]
        cmap = matplotlib.cm.get_cmap('hsv')
        colors = cmap(colors / np.max(colors))
        print('colors -> {},  min:{}, max:{}'.format(np.shape(colors), np.min(colors), np.max(colors)))

        colorImageLeft,colorImageRight = imgLeft.copy(),imgRight.copy()
        fig, axs = plt.subplots(1, 2)
        fig.set_size_inches(20, 10.5, forward=True)
        axs[0].imshow(imgLeft)
        #axs[0].scatter(points2D_left[:,0],points2D_left[:,1], s=.1, c='green')
        axs[0].scatter(points2D_left[:,0],points2D_left[:,1], s=.3, c=colors)
        axs[0].set_title("Left image")

        axs[1].set_title("Right image")
        axs[1].imshow(imgRight)
        #axs[1].scatter(points2D_right[:,0],points2D_right[:,1], s=.1, c='red')
        # Color map for the points
        colors = np.array(np.load(cloud_file, mmap_mode='r'))[:, 3]  #
        colors = colors[inrange_right[0]]
        colors = cmap(colors / np.max(colors))
        print('points2D_right->{},  colors->{}'.format(np.shape(points2D_right), np.shape(colors)))
        axs[1].scatter(points2D_right[:,0],points2D_right[:,1], s=.1, c=colors)

        fig.tight_layout()
        plt.show()
        points_left = objPoints_left[inrange_left[0]]
        points_right = objPoints_right[inrange_right[0]]

        print('points_left -> {},  colorImageLeft->{}'.format(np.shape(points_left), np.shape(colorImageLeft)))
        print('points_right -> {},  colorImageRight->{}'.format(np.shape(points_right), np.shape(colorImageRight)))

        colors_left = colorImageLeft[points2D_left[:, 1], points2D_left[:, 0], :]
        colors_right = colorImageRight[points2D_right[:, 1], points2D_right[:, 0], :]

        print('colors_left -> {}'.format(np.shape(colors_left)))
        print('colors_right -> {}'.format(np.shape(colors_right)))

        points = np.vstack((points_left,points_right))
        color = np.vstack((colors_left,colors_right))
        print('points->{}, color->{}'.format(np.shape(points), np.shape(color)))
        #plt.show()
        #self.write_ply('Lidar_cam.ply', points, color)
        self.view()
        plt.show()
        def hsv_to_rgb(h, s, v):
            if s == 0.0:
                return v, v, v

            i = int(h * 6.0)
            f = (h * 6.0) - i
            p = v * (1.0 - s)
            q = v * (1.0 - s * f)
            t = v * (1.0 - s * (1.0 - f))
            i = i % 6

            if i == 0:
                return v, t, p
            if i == 1:
                return q, v, p
            if i == 2:
                return p, v, t
            if i == 3:
                return p, q, v
            if i == 4:
                return t, p, v
            if i == 5:
                return v, p, q

        def filterOcclusion(data):
            print('data -> {}'.format(np.shape(data)))

            # ---create a pandas Dataframe with X,Y,Z
            print('Create a DataFrame')
            df = pd.DataFrame(data, columns=['X','Y','Z','X3D','Y3X','Z3D','R','G','B'])

            # ---sort it ascend by Z
            print('Sort by Z')
            df = df.sort_values(by=['Z'],kind='quicksort')

            print('Data point after sorting------------------------------')

            #---For each point create rectangle centered in current point
            xGap,yGap = 20, 50
            xOffset, yOffset = int(xGap / 2), int(yGap / 2)
            def create_rectange(x,y,depth):
                bl = [x-xOffset, y+yOffset] #bottom left
                tr = [x+xOffset, y-yOffset] #top right
                return [bl,tr,depth]

            print('Adding rectangles')

            import time
            #Rectangles = np.array([create_rectange(x=row['X'],y=row['Y'], depth = row['Z']) for index, row in df.iterrows()])
            vfunc = np.vectorize(create_rectange)
            Rectangles = vfunc(df['X'].values, df['Y'].values, df['Z'].values)
            df['Rectangles'] = Rectangles
            #Rectangles = np.asarray(Rectangles.tolist())
            #print('Rectangles -> {}'.format(np.shape(Rectangles)))
            #bl,tr = np.asarray(Rectangles[:,0].tolist()),np.asarray(Rectangles[:,0].tolist())
            # 'bl0 -> {}'.format(np.shape(bl), np.shape(tr))
            #df['bl0'] = bl[:,0]
            #df['bl1'] = bl[:, 1]
            #df['tr0'] = tr[:, 0]
            #df['tr1'] = tr[:, 1]
            # For each point, project it if it does not belong in prev 5 points
            t = .5
            def lies_inside(bl, tr, p, dist): #bottom_left, top_right, poin, distance_left, distance_right
                if (p[0] > bl[0] and p[0] < tr[0] and p[1] < bl[1] and p[1] > tr[1]):
                    if abs(p[-1]-dist)>t:
                        return True
                    else:
                        return False
                else:
                    return False

            def lies_inside_(bl0,bl1, tr0,tr1, p0,p1,p2, dist): #bottom_left, top_right, poin, distance_left, distance_right
                if (p0 > bl0 and p0 < tr0 and p1 < bl1 and p1 > tr1):
                    if abs(p2-dist)>t:
                        return True
                    else:
                        return False
                else:
                    return False

            lies_inside_ = np.vectorize(lies_inside_)
            occluded = np.zeros_like(Z, dtype=bool)
            projected = np.zeros_like(Z, dtype=bool)
            df['occluded'] = occluded
            df['projected'] = projected
            idx = range(len(df))
            df['idx'] = idx
            df = df.set_index(['idx'])

            # for each point check if the prev 5 points belongs to its rectangle -> if yes-> discard it
            print('Compute neighbors')
            from sklearn.neighbors import NearestNeighbors
            X = np.array(df.iloc[:,0:2])
            nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(X)
            distances, indices = nbrs.kneighbors(X)
            print('distances -> {}, indices->{}, df->{}'.format(np.shape(distances), np.shape(indices), np.shape(df)))
            df['nbrs_indices'] = indices[:,1:].tolist()

            print(df.head())
            import time
            start = time.time()
            print('Start projection')

            def soc_iter(i):
                print(i)
                # take the neighbours that are already projected and not occluded
                nbrs = df.iloc[i, -1]
                prev_points = df.iloc[nbrs]  # .query('projected == 1 & occluded == 0') #5.82813405991 s
                condition = (prev_points.projected == True) & (prev_points.occluded == False)
                prev_points = prev_points[condition]  # time = 156.481780052 s

                # print('nbrs -> {}, prev_points->{}, condition1->{}'.format(np.shape(nbrs), np.shape(prev_points), np.shape(condition)))
                if len(prev_points) > 0:
                    p = np.array(df.iloc[i, 0:3])  # current_point
                    # time = 156.481780052 s
                    Rectangles = prev_points['Rectangles']
                    occlusion = [lies_inside(bl=point[0], tr=point[1], p=p, dist=point[-1]) for point in Rectangles]
                    # time = 156.481780052 s
                    #occlusion = lies_inside_(prev_points['bl0'].values, prev_points['bl1'].values, prev_points['tr0'].values, prev_points['tr1'].values, p[0], p[1], p[-1], prev_points['Z'].values)
                    if np.any(occlusion):
                        # print('point {} is occluded'.format(p))
                        df.loc[i, 'occluded'] = True
                df.loc[i, 'projected'] = True
            soc_iter_vect = np.vectorize(soc_iter)

            N = len(df)

            m = np.linspace(start=1, stop=N-1, num=N-1, dtype=int)
            print('m->{}, N:{}'.format(np.shape(m),N))
            soc_iter_vect(m) # uncomment this

            '''for i in range(1,2): #len(df)
                print i
                # take the neighbours that are already projected and not occluded
                nbrs = df.iloc[i, -1]
                prev_points = df.iloc[nbrs]#.query('projected == 1 & occluded == 0') #5.82813405991 s
                condition = (prev_points.projected == True) & (prev_points.occluded == False)
                prev_points = prev_points[condition]  #time = 156.481780052 s

                #print('nbrs -> {}, prev_points->{}, condition1->{}'.format(np.shape(nbrs), np.shape(prev_points), np.shape(condition)))
                if len(prev_points)>0:
                    p = np.array(df.iloc[i, 0:3]) #current_point
                    # time = 303.82229900
                    #occlusion = (p[0] > (prev_points.X-xOffset)) & (p[0] < (prev_points.X+xOffset)) & (p[1] < (prev_points.Y+yOffset)) & (p[1] > (prev_points.Y-yOffset)) & (abs(p[-1] - prev_points.Z) > .3)
                    #time = 156.481780052 s
                    Rectangles = prev_points['Rectangles']
                    occlusion = np.array([lies_inside(bl=point[0], tr=point[1], p=p, dist=point[-1]) for point in Rectangles])
                    if np.any(occlusion):
                        #print('point {} is occluded'.format(p))
                        df.loc[i,'occluded'] = True
                df.loc[i, 'projected'] = True'''
            #soc_iter_vect(1)

            end = time.time()
            print('the publish took {}'.format(end - start))

            print(df.head())
            Points = np.array(df[df['occluded']==False]).squeeze()
            good_points = Points[:,0:2].astype('int')
            distance = Points[:,2]
            _3Dpoint = Points[:,3:6]
            _3Dcolor = Points[:, 6:9]

            MIN_DISTANCE, MAX_DISTANCE = np.min(distance), np.max(distance)
            colours = (distance - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE)
            colours = np.asarray([np.asarray(hsv_to_rgb( c, np.sqrt(1), 1.0)) for c in colours])

            cols = 255 * colours

            return good_points, cols,_3Dpoint, _3Dcolor

        #points left
        Z = np.linalg.norm(points_left, axis=1)[:, np.newaxis]
        data = np.hstack((points2D_left, Z))  # N x 3 (x,y,distance)
        data = np.hstack((data,points_left))  # N x 6
        data = np.hstack((data,colors_left))  # N x 9 (x,y,distance, X,Y,Z,R,G,B)

        good_points, cols,_3Dpoint, _3Dcolor = filterOcclusion(data = data)
        print('good_points->{}, cols->{}, _3Dpoint->{}, _3Dcolor->{}'.format(
            np.shape(good_points), np.shape(cols), np.shape(_3Dpoint), np.shape(_3Dcolor)))
        for i in range(len(good_points)):
            cv2.circle(imgLeft, tuple(good_points[i]), 2, cols[i], -1)

        Z = np.linalg.norm(points_right, axis=1)[:, np.newaxis]
        data = np.hstack((points2D_right, Z))  # N x 3 (x,y,distance)
        data = np.hstack((data,points_right))  # N x 6 (x,y,distance)
        data = np.hstack((data,colors_right))  # N x 9 (x,y,distance, X,Y,Z,R,G,B)

        _good_points, _cols,_3Dpoint_, _3Dcolor_ = filterOcclusion(data=data)
        print('good_points->{}, cols->{},  _3Dpoint->{}'.format(np.shape(good_points), np.shape(cols), np.shape(_3Dpoint)))
        for i in range(len(_good_points)):
            cv2.circle(imgRight, tuple(_good_points[i]), 2, _cols[i], -1)

        cv2.imshow('imgLeft', cv2.resize(imgLeft,None, fx=.4,fy=.4))
        cv2.imshow('imgRight', cv2.resize(imgRight,None, fx=.4,fy=.4))

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        #create a combined pointcloud
        print('_3Dpoint->{},  _3Dpoint_->{}'.format(np.shape(_3Dpoint), np.shape(_3Dpoint_)))
        print('_3Dcolor->{},  _3Dcolor_->{}'.format(np.shape(_3Dcolor), np.shape(_3Dcolor_)))
        points = np.vstack((_3Dpoint, _3Dpoint_))
        color = np.vstack((_3Dcolor, _3Dcolor_))
        #points = np.vstack((_3Dpoint_, _3Dpoint))
        #color = np.vstack((_3Dcolor_, _3Dcolor))
        #points = _3Dpoint #np.vstack((_3Dpoint, _3Dpoint_))
        #color = _3Dcolor #np.vstack((_3Dcolor, _3Dcolor_))
        print('points->{}, color->{}'.format(np.shape(points), np.shape(color)))
        self.write_ply('Lidar_cam_filtered.ply', points, color)
        self.view()

        plt.show()
        print('----------------------------------------------------------------------------------------')
        def occlus(t=.3):
            # columns=["X", "Y", "Z","intens","ring", time]
            _3DPoints = np.array(np.load(cloud_file, mmap_mode='r'), dtype=np.float32) #[:,6] # N x 6
            print('_3DPoints -> {}'.format(np.shape(_3DPoints)))

            # Left image--------------------------------------------------------------------------------------------
            objPoints_left = _3DPoints.copy()
            Z = self.get_z(q, objPoints_left[:,:3], self.K)
            objPoints_left = objPoints_left[Z > 0]
            print('objPoints_left:{}'.format(np.shape(objPoints_left)))
            points2D_left, _ = cv2.projectPoints(np.array(objPoints_left[:,:3]).squeeze(), rvec, tvec, self.K, self.D)
            points2D_left = np.squeeze(points2D_left)
            print('objPoints_left -> {},  points2D_left -> {},  '.format(np.shape(objPoints_left), np.shape(points2D_left)))

            inrange_left = np.where((points2D_left[:, 0] > 0) & (points2D_left[:, 1] > 0) &
                                    (points2D_left[:, 0] < imgLeft.shape[1] - 1) & (points2D_left[:, 1] < imgLeft.shape[0] - 1))
            points2D_left = points2D_left[inrange_left[0]].round().astype('int')
            print('points2D_left:{},  '.format(np.shape(points2D_left)))

            points_left = objPoints_left[inrange_left[0]]
            colors_left = colorImageLeft[points2D_left[:, 1], points2D_left[:, 0], :]
            print('points->{}, color->{}'.format(np.shape(points_left), np.shape(colors_left)))

            distance = np.linalg.norm(points_left[:,:3], axis=1)[:, np.newaxis]

            MIN_DISTANCE, MAX_DISTANCE = np.min(distance), np.max(distance)
            colours = np.asarray((distance - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE)).squeeze()
            colours = np.asarray([hsv_to_rgb(0.75 * c, np.sqrt(1), 1.0) for c in colours])
            cols = 255 * colours
            df = pd.DataFrame(data=points_left, columns=["X", "Y", "Z", "intens", "ring", "time"])
            df['RGB'] = colors_left.tolist()
            df['pixels'] = points2D_left.tolist()
            df['distance'] = distance #.round()
            df['color'] = cols.tolist()
            print(df.head())
            gp = df.groupby('ring')
            keys = gp.groups.keys()
            print('keys -> {}'.format(np.shape(keys)))

            _3Dcolor,_3Dpoint = [],[]
            k=0

            for i in keys:
                group = gp.get_group(i).to_numpy()     # X,Y,Z,intens,ring,time,RGB,pixels,distance,color


                #group = gp.get_group(i+b).to_numpy()
                N = len(group)
                print('Ring ->{}, {}'.format(i, np.shape(group)))
                #take x pixels
                pixels = np.concatenate(group[:,7]).reshape(-1,2)
                sorted_idx = pixels[:, 0].argsort(kind='mergesort') #sort by x pixel
                #sorted_idx = np.linspace(start = 0, stop = N-1, num = N, dtype = int)
                pixels = pixels[sorted_idx]
                distance = group[sorted_idx,8]
                points, colors = np.asarray(group[:, :3]), np.asarray(group[:, 6])
                k+=1
                collours = []
                for j in range(1, N):
                    d = distance[j] - distance[j - 1]
                    s = np.sign(d)
                    if abs(d) > t:
                        if s < 0:
                            col, col_ = 'b', (255, 0, 0)
                            _3Dcolor.append(colors[j])
                            _3Dpoint.append(points[j])
                            size = 3
                            l=2
                        else:
                            col, col_ = 'r', (0, 0, 255)
                            #col, col_ = 'g', (0, 255, 0)
                            size = 2
                            l=-1
                    else:
                        col, col_ = 'g', (0, 255, 0)
                        _3Dcolor.append(colors[j])
                        _3Dpoint.append(points[j])
                        size = 2
                        l = -1
                    collours.append(col)
                    #cv2.circle(imgLeft, tuple(pixels[j]), size, col_, l)
                    cv2.circle(imgLeft, tuple(pixels[j]), size, (0,255,0), l)

                    #plt.scatter(pixels[j,0], distance[j], c=col, s=2)

                #plt.plot(pixels[:, 0], distance, c='blue', alpha=0.2)
                plt.scatter(pixels[:, 0], distance, c=collours, s=.5)
                #if k%3==0:
                #plt.grid()
                #plt.show()

                #distance = group[:,8] #unsorted
                #m = np.linspace(start = 0, stop = N-1, num = N, dtype = int)
                #plt.scatter(m, distance,s=2)
                #plt.grid()
                #plt.show()

                cv2.imshow('imgLeft', cv2.resize(imgLeft, None, fx=.5, fy=.5))
                cv2.waitKey(0)

            cv2.imshow('imgLeft', cv2.resize(imgLeft, None, fx=.5, fy=.5))
            plt.grid()
            plt.show()
            cv2.waitKey(0)

            print('_3Dpoint -> {}, _3Dcolor->{}'.format(np.shape(_3Dpoint), np.shape(_3Dcolor)))
            #self.write_ply('0Lidar_cam_filter2.ply', np.array(_3Dpoint), np.array(_3Dcolor))
            #self.view()
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        #occlus()
        #self.view()

    def DLT(self):
        def vgg_rq(S):
            S = S.T
            Q, U = np.linalg.qr(np.fliplr(np.flipud(S)))
            Q = np.fliplr(np.flipud(Q.T))
            U = np.fliplr(np.flipud(U.T))
            return U, Q

        def vgg_KR_from_P(P, noscale=False):
            N = P.shape[0]
            H = P[:, :N]

            K, R = vgg_rq(H)
            if not noscale:
                K = K / K[N - 1, N - 1]
                if K[0, 0] < 0:
                    D = np.diag(np.hstack((np.array([-1, -1]), np.ones(N - 2))))
                    K = np.dot(K, D)
                    R = np.dot(D, R)

            t = np.linalg.lstsq(-P[:, 0:N], P[:, -1])[0]
            return K, R, t

        def camcalibDLT(Xworld, Xim):
            # Xworld - (8, 4)
            # Xim - (8, 3)
            n, d = np.shape(Xworld)
            zeros_1x4 = np.zeros((1, 4))
            saved_data = []
            for j in range(n):
                world_point = np.array(Xworld[j]).reshape(1, -1)
                image_point = np.array(Xim[j]).reshape(-1, 1)
                image2world = image_point.dot(world_point)  # (3, 4)

                minus_row1 = -(image2world[0]).reshape(1, -1)  # (1, 4)
                minus_row2 = -(image2world[1]).reshape(1, -1)  # (1, 4)
                row3 = image2world[2].reshape(1, -1)  # (1, 4)

                stack = np.stack((zeros_1x4, row3, minus_row2)).reshape(1, -1)  # (1, 12)
                saved_data.append(stack[0])

                stack = np.stack((row3, zeros_1x4, minus_row1)).reshape(1, -1)  # (1, 12)
                saved_data.append(stack[0])

            saved_data = np.array(saved_data)
            print('saved_data ', np.shape(saved_data))

            _, Sigma, V = np.linalg.svd(saved_data)
            P = V[np.argmin(Sigma)].reshape((3, 4))

            return P

        def GET_DATA():
            # get data from chessboard
            '''name = 'chess'
            self.file = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/data/GoodPoints_{}.pkl'.format(name)
            self.load_points()
            Lidar_3D, Image_2D, Image_3D = np.array(self.Lidar_3D).reshape(-1, 3), np.array(self.Image_2D).reshape(-1,
                                                                                                                   2), np.array(
                self.Image_3D).reshape(-1, 3)

            # get data from charuco'''
            name = 'charuco'
            self.file = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/data/GoodPoints_{}.pkl'.format(name)
            self.load_points()
            #Lidar_3D, Image_2D = np.vstack((Lidar_3D, np.array(self.Lidar_3D).reshape(-1, 3))), np.vstack(
            #    (Image_2D, np.array(self.Image_2D).reshape(-1, 2)))
            #print('Lidar_3D:->{}, Image_2D:->{}'.format(np.shape(Lidar_3D), np.shape(Image_2D)))
            Lidar_3D, Image_2D, Image_3D = np.array(self.Lidar_3D).reshape(-1, 3), np.array(self.Image_2D).reshape(-1,
                                                                                                                   2), np.array(
                self.Image_3D).reshape(-1, 3)

            imgp = np.array([Image_2D], dtype=np.float32).squeeze()
            objp = np.array([Lidar_3D], dtype=np.float32).squeeze()
            return objp, imgp

        _3D_points, _2D_points = GET_DATA()
        print('_3D_points->{},  _2D_points->{}'.format(np.shape(_3D_points), np.shape(_2D_points)))

        P1 = camcalibDLT(np.hstack((_3D_points, np.ones((len(_3D_points), 1)))),
                         np.hstack((_2D_points, np.ones((len(_3D_points), 1)))))
        print('P1 -> {}'.format(P1))
        # Check the results by projecting the world points with the estimated P.
        # The projected points should overlap with manually localized points
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
        # plot manually localized
        axes.plot(_2D_points[:,0], _2D_points[:,1], 'c+', markersize=10)

        # plot projected
        pproj1 = np.dot(P1, np.hstack((_3D_points, np.ones((len(_3D_points), 1)))).T)
        for i in range(len(_3D_points)):
            axes.plot(pproj1[0, i] / pproj1[2, i], pproj1[1, i] / pproj1[2, i], 'rx', markersize=12)

        plt.show()

        print('intrinsic camera calibration matrices')
        K1, R1, t1 = vgg_KR_from_P(P1)
        print('K1')
        print(K1)

    def estimate(self, a1=None,a2=None):
        import numpy as np
        import numpy.linalg

        # Relevant links:
        #   - http://stackoverflow.com/a/32244818/263061 (solution with scale)
        #   - "Least-Squares Rigid Motion Using SVD" (no scale but easy proofs and explains how weights could be added)

        # Rigidly (+scale) aligns two point clouds with know point-to-point correspondences
        # with least-squares error.
        # Returns (scale factor c, rotation matrix R, translation vector t) such that
        #   Q = P*cR + t
        # if they align perfectly, or such that
        #   SUM over point i ( | P_i*cR + t - Q_i |^2 )
        # is minimised if they don't align perfectly.
        def umeyama(P, Q):
            assert P.shape == Q.shape
            n, dim = P.shape

            centeredP = P - P.mean(axis=0)
            centeredQ = Q - Q.mean(axis=0)

            C = np.dot(np.transpose(centeredP), centeredQ) / n

            V, S, W = np.linalg.svd(C)
            d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

            if d:
                S[-1] = -S[-1]
                V[:, -1] = -V[:, -1]

            R = np.dot(V, W)

            varP = np.var(a1, axis=0).sum()
            c = 1 / varP * np.sum(S)  # scale factor

            t = Q.mean(axis=0) - P.mean(axis=0).dot(c * R)

            return c, R, t

        # Testing

        np.set_printoptions(precision=3)

        if a1 is None and a2 is None:
            a1 = np.array([
                [0, 0, -1],
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0],
            ])

            a2 = np.array([
                [0, 0, 1],
                [0, 0, 0],
                [0, 0, -1],
                [0, 1, 0],
                [-1, 0, 0],
            ])
            a2 *= 2  # for testing the scale calculation
            a2 += 3  # for testing the translation calculation

        c, R, t = umeyama(a1, a2)
        print "R =\n", R
        print "c =", c
        print "t =\n", t
        print
        print "Check:  a1*cR + t = a2  is", np.allclose(a1.dot(c * R) + t, a2)
        err = ((a1.dot(c * R) + t - a2) ** 2).sum()
        print "Residual error", err

        return c, R, t

    def project3D_2D_onImage(self, imgLeft, _3DPoints):
        def readCalibrationExtrinsic():
            calib_file = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/solvePnP_extrinsics{}.npz'.format(
                'chess' if self.chess else 'charuco')
            calib_file = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/combined_extrinsics{}.npz'
            with open(calib_file, 'r') as f:
                data = f.read().split()
                #print('data:{}'.format(data))
                qx = float(data[0])
                qy = float(data[1])
                qz = float(data[2])
                qw = float(data[3])
                tx = float(data[4])
                ty = float(data[5])
                tz = float(data[6])

            q = Quaternion(qw, qx, qy, qz).transformation_matrix
            q[0, 3],q[1, 3],q[2, 3] = tx,ty,tz
            tvec = q[:3, 3]

            rot_mat = q[:3, :3]
            #rvec, _ = cv2.Rodrigues(rot_mat)
            rvec = rot_mat
            print('tvec -> {}'.format(tvec))

            return rvec, tvec, q

        rvec, tvec, q = readCalibrationExtrinsic()

        #Left image--------------------------------------------------------------------------------------------
        objPoints_left = _3DPoints.copy()
        Z = self.get_z(q, objPoints_left, self.K)
        objPoints_left = objPoints_left[Z > 0]
        points2D_left, _ = cv2.projectPoints(objPoints_left, rvec, tvec, self.K, self.D)
        points2D_left = np.squeeze(points2D_left)

        inrange_left = np.where((points2D_left[:, 0] > 0) & (points2D_left[:, 1] > 0) &
            (points2D_left[:, 0] < imgLeft.shape[1]-1) & (points2D_left[:, 1] < imgLeft.shape[0]-1))
        points2D_left = points2D_left[inrange_left[0]].round().astype('int')
        for i in range(len(points2D_left)):
            cv2.circle(imgLeft, tuple(points2D_left[i]), 2, (0, 255, 0), -1)

        return imgLeft

    def calibrate_3D_3D(self):
        def rot2eul(R):
            beta = -np.arcsin(R[2, 0])
            alpha = np.arctan2(R[2, 1] / np.cos(beta), R[2, 2] / np.cos(beta))
            gamma = np.arctan2(R[1, 0] / np.cos(beta), R[0, 0] / np.cos(beta))
            return np.array((np.rad2deg(alpha), np.rad2deg(beta), np.rad2deg(gamma)))

        print('3D-3D ========================================================================================')
        Lidar_3D = np.array(self.Lidar_3D).reshape(-1, 3)
        Image_3D = np.array(self.Image_3D).reshape(-1, 3)
        print('Lidar_3D:{}, Image_3D:{}'.format(np.shape(Lidar_3D), np.shape(Image_3D)))

        self.fig = plt.figure(figsize=plt.figaspect(1.))
        ax1 = self.fig.add_subplot(1, 1, 1, projection='3d')
        ax1.set_xlabel('X', fontsize=8)
        ax1.set_ylabel('Y', fontsize=8)
        ax1.set_zlabel('Z', fontsize=8)
        ax1.set_xlim([-3, 3])
        ax1.set_ylim([-3, 3])
        ax1.set_zlim([-5, 10])
        #ax1.set_axis_off()
        #plot all data
        #ax1.scatter(*Lidar_3D.T, c='blue', label = 'LiDAR points')
        #ax1.scatter(*Image_3D.T, s=25, c='red', label = 'Stereo Cam points')

        ax1.scatter(*self.Lidar_3D[0].T, c='blue', label='LiDAR points')
        ax1.scatter(*self.Image_3D[0].T, s=25, c='red', label='Stereo Cam points')
        dist_mat = distance_matrix(self.Image_3D[0],self.Image_3D[0])
        print('distance_matrix cam')
        print(dist_mat)
        dist_mat = distance_matrix(self.Lidar_3D[0], self.Lidar_3D[0])
        print('distance_matrix LiDAR')
        print(dist_mat)

        #ax1.legend()
        #plt.show()



        #estimate transformation ====================================================
        c, R, t = self.estimate(Lidar_3D,Image_3D)
        print('t:{}'.format(t))
        angles = rot2eul(R)
        print('angles:{}'.format(angles))
        Camera_points3D = self.Lidar_3D[0].dot(c * R) + t
        #Camera_points3D = self.Lidar_3D[0].dot(R) + t
        ax1.scatter(*Camera_points3D.T, label='Transformed LiDAR')
        ax1.legend()
        plt.show()


        #project on image ===========================================================
        l = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/data/charuco/left/left_4.png'
        img = cv2.imread(l)
        cloud_file = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/data/charuco/cloud_4.npy'

        i = 12
        #l = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/cool/left_{}.png'.format(i)
        #img = cv2.imread(l)
        #cloud_file = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/cool/cloud_{}.npy'.format(i)

        img = cv2.remap(src=img, map1=self.leftMapX, map2=self.leftMapY,interpolation=cv2.INTER_LINEAR, dst=None, borderMode=cv2.BORDER_CONSTANT)
        LiDAR_points3D = np.array(np.load(cloud_file, mmap_mode='r'), dtype=np.float32)[:, :3]  #
        Camera_points3D = LiDAR_points3D.dot(c * R) + t   #LiDAR points in camera frame
        print('LiDAR_points3D:{}, Camera_points3D:{}'.format(np.shape(LiDAR_points3D), np.shape(Camera_points3D)))

        homogen = lambda x: np.array([x[0],x[1],x[2],1])
        invhomogen = lambda x: np.array([x[0]/x[-1], x[1]/x[-1]])
        cam = np.array([homogen(x) for x in Camera_points3D[:, :3]])
        points2D = self.P1.dot(cam.T).T
        points2D = np.array([invhomogen(x) for x in points2D[:]])
        print('points2D -> {}'.format(np.shape(points2D)))
        inrange = np.where(
            (points2D[:, 0] >= 0) & (points2D[:, 1] >= 0) &
            (points2D[:, 0] < img.shape[1]) & (points2D[:, 1] < img.shape[0])
        )
        points2D = points2D[inrange[0]].round().astype('int')

        for i in range(len(points2D)):
            cv2.circle(img, tuple(points2D[i]), 2, (0, 255, 0), -1)


        projection_2D_3D = self.project3D_2D_onImage(cv2.imread(l), LiDAR_points3D)

        #cv2.imshow('3D-3D estimation', cv2.resize(img,None,fx=.4,fy=.4))
        #cv2.imshow('2D-3D estimation', cv2.resize(projection_2D_3D,None,fx=.4,fy=.4))
        cv2.putText(img, '3D-3D', (20, 1200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.putText(projection_2D_3D, '3D-2D', (20, 1200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        scale = .4
        _horizontal = np.hstack(
            (cv2.resize(img, None, fx=scale, fy=scale), cv2.resize(projection_2D_3D, None, fx=scale, fy=scale)))
        cv2.imshow('Estimation', _horizontal)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print('self.P1->{}'.format(np.shape(self.P1)))
        print(self.P1)

        #-------------------------------------------------------------
        l = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/data/chess/left/left_5.png'
        img = cv2.imread(l)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (10, 7), None)
        print('ret ->{}'.format(ret))
        if ret == True:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)

            # Find the rotation and translation vectors.
            success, rvecs, tvecs, inliers = cv2.solvePnPRansac(self.objp, corners2, self.K, self.D)
            print('success->{}  rvecs:{}, tvecs:{}, inliers:{}'.format(success, np.shape(rvecs), np.shape(tvecs), np.shape(inliers)))
            #print(rvecs)
            #print(tvecs)
            rvecs,_ = cv2.Rodrigues(rvecs)

            print('self.objp->{}'.format(np.shape(self.objp)))
            _3Dpoints = self.objp
            # project 3D points to image plane
            _2Dpoints, jac = cv2.projectPoints(_3Dpoints, rvecs, tvecs, self.K, self.D)
            _2Dpoints = np.array(_2Dpoints, dtype=np.float32).squeeze()
            print('_2Dpoints -> {}'.format(np.shape(_2Dpoints)))
            for i in range(len(_2Dpoints)):
                cv2.circle(img, tuple(_2Dpoints[i]), 5, (0, 255, 0), 3)
            _3Dpoints = rvecs.dot(_3Dpoints.T)+tvecs
            _3Dpoints = _3Dpoints.T
            #rvecs, tvecs, _3Dpoints, _2Dpoints
            print('_3Dpoints->{}'.format(np.shape(_3Dpoints)))
            print(_3Dpoints)
            dist_mat = distance_matrix(_3Dpoints,_3Dpoints)
            print('dist_mat')
            print(dist_mat)
            self.fig = plt.figure(figsize=plt.figaspect(1.))
            ax1 = self.fig.add_subplot(1, 1, 1, projection='3d')
            ax1.scatter(*_3Dpoints.T, label='OpenCV')
            #ax1.scatter(*self.Image_3D[0].T, s=25, c='red', label='Stereo Cam points')
            ax1.legend()


        cv2.imshow('img', cv2.resize(img, None, fx=.4, fy=.4))
        cv2.waitKey(0)
        plt.show()
        cv2.destroyAllWindows()

    def doSomePlots(self):
        points3D = np.array(self.Lidar_3D).reshape(-1, 3)
        points2D = np.array(self.Image_2D).reshape(-1, 2)
        print('points3D:{}, points2D:{}'.format(np.shape(points3D), np.shape(points2D)))

        def readCalibrationExtrinsic():
            calib_file = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/solvePnP_extrinsics{}.npz'.format(
                'chess' if self.chess else 'charuco')
            calib_file = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/combined_extrinsics{}.npz'
            with open(calib_file, 'r') as f:
                data = f.read().split()
                #print('data:{}'.format(data))
                qx = float(data[0])
                qy = float(data[1])
                qz = float(data[2])
                qw = float(data[3])
                tx = float(data[4])
                ty = float(data[5])
                tz = float(data[6])

            q = Quaternion(qw, qx, qy, qz).transformation_matrix
            q[0, 3],q[1, 3],q[2, 3] = tx,ty,tz
            tvec = q[:3, 3]
            rot_mat = q[:3, :3]
            #rvec, _ = cv2.Rodrigues(rot_mat)
            rvec = rot_mat
            print('tvec -> {}'.format(tvec))

            return rvec, tvec, q

        #ground truth estimation
        rvec, tvec, q = readCalibrationExtrinsic()
        ground_truth_rotation = euler_from_matrix(rvec)
        ground_truth_translation = np.array(tvec).squeeze()
        ground_truth_rotation = np.array([(180.0 / math.pi) * i for i in ground_truth_rotation]).squeeze()
        print('ground_truth_rotation: ', ground_truth_rotation)
        print('ground_truth_translation: ', ground_truth_translation)

        #randomly select 5%, 10%, 15%, ..., 100% of data points
        #compute the transformation
        #estimate the error between the ground truth
        #save for later plot and plot it
        percentage = np.linspace(2,100,20)
        N = len(points3D) #
        Idx = np.arange(0,len(points3D))
        print('N -> {}'.format(N))
        print('percentage -> {}'.format(percentage))
        rot_, tran_ = [], []
        for i in range(20):
            rot, tran = [], []
            for p in percentage:
                nr_points = int(p*N/100)
                #print(nr_points)
                idx_points = np.random.choice(Idx, nr_points)
                train_lidar = points3D[idx_points]
                train_camera = points2D[idx_points]

                imgp = np.array([train_camera], dtype=np.float32).squeeze()
                objp = np.array([train_lidar], dtype=np.float32).squeeze()

                retval, rvec, tvec = cv2.solvePnP(objp, imgp, self.K, self.D, flags=cv2.SOLVEPNP_ITERATIVE)
                rvec,_ = cv2.Rodrigues(rvec)
                tvec = np.array(tvec).squeeze()
                _rotation = euler_from_matrix(rvec)
                _rotation = np.array([(180.0 / math.pi) * i for i in _rotation])

                err_rotation = np.abs(_rotation-ground_truth_rotation)
                err_translation = np.abs(tvec - ground_truth_translation)
                #print('err_rotation->{}, err_translation->{}'.format(np.shape(err_rotation), np.shape(err_translation)))
                rot.append(err_rotation)
                tran.append(err_translation)

            print('rot->{}, tran->{}'.format(np.shape(rot), np.shape(tran)))
            rot_.append(rot)
            tran_.append(tran)
        print('rot_->{}, tran_->{}'.format(np.shape(rot_), np.shape(tran_)))
        rot_ = np.mean(rot_, axis=0)
        tran_ = np.mean(tran_, axis=0)
        print('rot_->{}, tran_->{}'.format(np.shape(rot_), np.shape(tran_)))
        ticks = percentage * N / 100
        print('ticks -> {}'.format(np.shape(ticks)))
        plt.plot(ticks,rot_[:,0], label='X')
        plt.plot(ticks,rot_[:, 1], label='Y')
        plt.plot(ticks,rot_[:, 2], label='Z')
        plt.legend()
        plt.xlabel("n-points")
        plt.ylabel("mean rotation error (degree)")

        plt.xticks(ticks)
        plt.show()

        plt.plot(ticks,tran_[:, 0], label='X')
        plt.plot(ticks,tran_[:, 1], label='Y')
        plt.plot(ticks,tran_[:, 2], label='Z')
        plt.xlabel("n-points")
        plt.ylabel("mean translation error (m)")
        plt.legend()
        plt.show()

    def do_holy_Final_calibration(self,viewData = False):
        #get data
        self.Lidar_3D = np.array(self.Lidar_3D)[:,-1,:]
        self.Image_3D = np.array(self.Image_3D)[:, -1, :]
        self.Image_2D = np.array(self.Image_2D)[:, -1, :]
        self.Image_2D2 = np.array(self.Image_2D2)[:, -1, :]
        print('self.Lidar_3D ->{}, self.Image_3D->{}'.format(np.shape(self.Lidar_3D), np.shape(self.Image_3D)))

        points3D_Lidar = np.array(self.Lidar_3D, dtype=np.float32).reshape(-1, 3)
        points3D_Camera = np.array(self.Image_3D, dtype=np.float32).reshape(-1, 3)
        points2DLeft = np.array(self.Image_2D, dtype=np.float32).reshape(-1, 2)
        points2DRight = np.array(self.Image_2D2, dtype=np.float32).reshape(-1, 2)

        print('points3D_Lidar:{},points3D_Camera:{}, points2DLeft:{}, points2DRight:{}'.format(np.shape(points3D_Lidar),np.shape(points3D_Camera), np.shape(points2DLeft), np.shape(points2DRight)))

        #visualize the data
        if viewData:
            for i in range(len(self.Lidar_3D)):
                fig = plt.figure()
                ax0 = fig.add_subplot(2, 2, 1, projection='3d')  # Lidar
                ax0.set_title('Lidar points')
                ax1 = fig.add_subplot(2, 2, 2, projection='3d')  # camera 3d
                ax1.set_title('Camera 3D')
                ax2 = fig.add_subplot(2, 2, 3)  # left pixels
                ax2.set_title('Left px')
                ax3 = fig.add_subplot(2, 2, 4)  # right pixels
                ax3.set_title('Right px')
                print(i)
                ax0.clear()
                ax0.scatter(*self.Lidar_3D[i].T)
                ax0.set_title('Lidar points')
                dist_Lidar = distance_matrix(self.Lidar_3D[i],self.Lidar_3D[i])
                print('dist_Lidar---------------------------------------------------------')
                print(dist_Lidar[0,:11])

                ax1.clear()
                ax1 = plt.axes(projection='3d')
                ax1.scatter(*self.Image_3D[i].T, c='k', marker='v', alpha=1)
                ax1.set_title('Camera 3D')
                dist_Cam = distance_matrix(self.Image_3D[i], self.Image_3D[i])
                print('dist_Cam---------------------------------------------------------')
                print(dist_Cam[0,:11])
                data = np.array(self.Image_3D).squeeze()
                #ax1.plot_wireframe(data[i,:,0], data[i,:,1], data[i,:,2], rstride=1, cstride=1)

                ax1.plot_trisurf(data[i,:,0], data[i,:,1], data[i,:,2],
                               alpha=.4, color='grey', shade=False)
                ax1.set_xlabel('X')
                ax1.set_ylabel('Y')
                ax1.set_zlabel('Z')
                ax1.set_xticks([])
                ax1.set_yticks([])
                ax1.set_zticks([])
                ax1.set_axis_off()
                plt.show()

                ax2.clear()
                ax2.scatter(*self.Image_2D[i].T)
                ax2.set_title('Left px')

                ax3.clear()
                ax3.scatter(*self.Image_2D2[i].T)
                ax3.set_title('Right px')

                plt.show()
                break
        #Calibrate LiDAR3d-Camera3D
        self.fig = plt.figure(figsize=plt.figaspect(1.))
        ax1 = self.fig.add_subplot(1, 1, 1, projection='3d')
        ax1.set_xlabel('X', fontsize=8)
        ax1.set_ylabel('Y', fontsize=8)
        ax1.set_zlabel('Z', fontsize=8)
        ax1.set_xlim([-3, 3])
        ax1.set_ylim([-3, 3])
        ax1.set_zlim([-3, 3])
        # ax1.set_axis_off()

        ax1.scatter(*self.Lidar_3D[0].T, c='blue', label='LiDAR points')
        ax1.scatter(*self.Image_3D[0].T, s=25, c='red', label='Stereo Cam points')

        #ax1.scatter(*points3D_Lidar.T, c='blue', label='LiDAR points2')
        #ax1.scatter(*points3D_Camera.T, s=25, c='red', label='Stereo Cam points2')

        # estimate transformation ====================================================
        c, R, t = self.estimate(points3D_Lidar, points3D_Camera)
        pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
        unpad = lambda x: x[:, :-1]
        # Solve the least squares problem X * A = Y # to find our transformation matrix A
        A, res, rank, s = np.linalg.lstsq(pad(points3D_Lidar),  pad(points3D_Camera))
        transform = lambda x: unpad(np.dot(pad(x), A))

        #Camera_points3D = transform(np.array(self.Lidar_3D[0]))  # transformation estimated with LS
        #ax1.scatter(*Camera_points3D.T, label='least square sol')

        print('t:{}'.format(t))
        angles = euler_from_matrix(R)
        print('euler angles ', [(180.0 / math.pi) * i for i in angles])
        Camera_points3D = self.Lidar_3D[0].dot(c * R) + t
        #Camera_points3D = self.Lidar_3D[0].dot(R) + t
        ax1.scatter(*Camera_points3D.T, label='SVD')
        ax1.legend()
        plt.show()
        left_src = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/data/chess/left/left_0.png'
        left_src = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/data/charuco/left/left_4.png'
        left_img = cv2.imread(left_src)
        cloud_file = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/data/chess/cloud_0.npy'
        cloud_file = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/data/charuco/cloud_4.npy'
        _3DPoints = np.array(np.load(cloud_file, mmap_mode='r'), dtype=np.float32)[:, :3]

        # Left image--------------------------------------------------------------------------------------------
        objPoints_left = _3DPoints.copy()
        objPoints_left = objPoints_left.dot(c * R) + t
        #objPoints_left  = np.array(transform(_3DPoints), dtype=np.float32).squeeze()  # transformation estimated with LS
        #objPoints_left = Camera_points3D
        print('objPoints_left ->{}'.format(np.shape(objPoints_left)))
        print(objPoints_left)
        points2D_left, _ = cv2.projectPoints(objPoints_left, np.eye(3), np.zeros(3), self.K_left, self.D_left)
        points2D_left = np.squeeze(points2D_left)
        print('objPoints_left -> {},  points2D_left -> {},  '.format(np.shape(objPoints_left), np.shape(points2D_left)))
        inrange_left = np.where((points2D_left[:, 0] > 0) & (points2D_left[:, 1] > 0) &
                                (points2D_left[:, 0] < left_img.shape[1] - 1) & (
                                            points2D_left[:, 1] < left_img.shape[0] - 1))
        points2D_left = points2D_left[inrange_left[0]].round().astype('int')
        for i in range(len(points2D_left)):
            cv2.circle(left_img, tuple(points2D_left[i]), 2, (0, 255, 0), -1)

        cv2.imshow('left_img 3D-3D estimation', cv2.resize(left_img, None, fx=.4, fy=.4))
        cv2.waitKey(0)
        # cv2.destroyAllWindows()



        #calibrate Lidar-> left camera
        print('Calibrate LiDAR->Left camera===============================================================')
        imgp = np.array([points2DLeft], dtype=np.float32).squeeze()
        objp = np.array([points3D_Lidar], dtype=np.float32).squeeze()
        print('imgp->{},objp->{}'.format(np.shape(imgp), np.shape(objp)))
        retval, rvec, tvec = cv2.solvePnP(objp, imgp, self.K, self.D, flags=cv2.SOLVEPNP_ITERATIVE)
        #success, rvec, tvec, inliers = cv2.solvePnPRansac(objp,imgp, self.K, self.D,flags=cv2.SOLVEPNP_ITERATIVE)
        rvec, tvec = cv2.solvePnPRefineLM(objp, imgp, self.K, self.D, rvec, tvec)
        rvec, jac = cv2.Rodrigues(rvec)
        print("RMSE in pixel = %f" % self.rmse(objp, imgp, self.K_left, self.D_left, rvec, tvec))
        print("T = ")
        print(tvec)
        print('Euler angles')
        angles = euler_from_matrix(rvec)
        self.Lidar_left_tvec = tvec
        self.Lidar_left_rvec = rvec
        print('euler angles ', [(180.0 / math.pi) * i for i in angles])

        #test calibration LiDAR->Left camera
        left_src = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/data/chess/left/left_0.png'
        left_src = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/data/charuco/left/left_4.png'
        left_img = cv2.imread(left_src)
        cloud_file = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/data/chess/cloud_0.npy'
        cloud_file = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/data/charuco/cloud_4.npy'
        _3DPoints = np.array(np.load(cloud_file, mmap_mode='r'), dtype=np.float32)[:, :3]

        #Left image--------------------------------------------------------------------------------------------
        objPoints_left = _3DPoints.copy()
        points2D_left, _ = cv2.projectPoints(objPoints_left, rvec, tvec, self.K_right, self.D_right)
        points2D_left = np.squeeze(points2D_left)
        print('objPoints_left -> {},  points2D_left -> {},  '.format(np.shape(objPoints_left), np.shape(points2D_left)))
        inrange_left = np.where((points2D_left[:, 0] > 0) & (points2D_left[:, 1] > 0) &
                                (points2D_left[:, 0] < left_img.shape[1] - 1) & (points2D_left[:, 1] < left_img.shape[0] - 1))
        points2D_left = points2D_left[inrange_left[0]].round().astype('int')
        for i in range(len(points2D_left)):
            cv2.circle(left_img, tuple(points2D_left[i]), 2, (0,255,0), -1)

        cv2.imshow('left_img',cv2.resize(left_img,None,fx=.4,fy=.4))
        cv2.waitKey(0)
        #cv2.destroyAllWindows()





        #=======================================================================================
        # calibrate Lidar-> right camera
        print('Calibrate LiDAR->right camera===============================================================')
        imgp = np.array([points2DRight], dtype=np.float32).squeeze()
        objp = np.array([points3D_Lidar], dtype=np.float32).squeeze()
        print('imgp->{},objp->{}'.format(np.shape(imgp), np.shape(objp)))
        retval, rvec, tvec = cv2.solvePnP(objp, imgp, self.K, self.D, flags=cv2.SOLVEPNP_ITERATIVE)
        #success, rvec, tvec, inliers = cv2.solvePnPRansac(objp,imgp, self.K, self.D,flags=cv2.SOLVEPNP_ITERATIVE)
        rvec, tvec = cv2.solvePnPRefineLM(objp, imgp, self.K, self.D, rvec, tvec)
        rmat, jac = cv2.Rodrigues(rvec)
        print("RMSE in pixel = %f" % self.rmse(objp, imgp, self.K, self.D, rvec, tvec))
        print("T = ")
        print(tvec)
        print('Euler angles')
        self.Lidar_right_tvec = tvec
        self.Lidar_right_rvec = rmat
        angles = euler_from_matrix(rmat)
        print('euler angles ', [(180.0 / math.pi) * i for i in angles])
        print("Quaternion = ")
        q = Quaternion(matrix=rmat).transformation_matrix
        #tvec[2] = -.59
        q[0, 3], q[1, 3], q[2, 3] = tvec[0], tvec[1], tvec[2]
        # test calibration LiDAR->Left camera
        src = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/data/chess/right/right_0.png'
        src = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/data/charuco/right/right_4.png'
        img = cv2.imread(src)
        cloud_file = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/data/chess/cloud_0.npy'
        cloud_file = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/data/charuco/cloud_4.npy'
        _3DPoints = np.array(np.load(cloud_file, mmap_mode='r'), dtype=np.float32)[:, :3]

        # Left image--------------------------------------------------------------------------------------------
        objPoints_left = _3DPoints.copy()
        Z = self.get_z(q, objPoints_left, self.K)
        objPoints_left = objPoints_left[Z > 0]
        points2D_left, _ = cv2.projectPoints(objPoints_left, rvec, tvec, self.K_right, self.D_right)
        points2D_left = np.squeeze(points2D_left)
        print('objPoints_left -> {},  points2D_left -> {},  '.format(np.shape(objPoints_left), np.shape(points2D_left)))
        inrange_left = np.where((points2D_left[:, 0] > 0) & (points2D_left[:, 1] > 0) &
                                (points2D_left[:, 0] < left_img.shape[1] - 1) & (
                                            points2D_left[:, 1] < left_img.shape[0] - 1))
        points2D_left = points2D_left[inrange_left[0]].round().astype('int')
        for i in range(len(points2D_left)):
            cv2.circle(img, tuple(points2D_left[i]), 2, (0, 255, 0), -1)

        cv2.imshow('right_img', cv2.resize(img, None, fx=.4, fy=.4))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print('=============================================================')
        #test stereo calibration based on lidar extrinsics
        stere_tvec = np.array([-0.96, 0., 0.12])[:, np.newaxis]
        angles = euler_from_matrix(self.R)
        stereo_angles = np.array([(180.0 / math.pi) * i for i in angles])
        print('Stereo camera calibration extrinsics')
        print('angles -> {}'.format(stereo_angles))
        print('tvec -> {}'.format(stere_tvec.ravel()))

        T_lidar_leftCam = np.vstack((np.hstack((self.Lidar_left_rvec, self.Lidar_left_tvec)), np.array([0, 0, 0, 1])[:,np.newaxis].T))
        T_lidar_rightCam = np.vstack((np.hstack((self.Lidar_right_rvec, self.Lidar_right_tvec)), np.array([0, 0, 0, 1])[:,np.newaxis].T))

        #T left cam to right cam is T1^-1 * T2
        T_leftCam_rightCam = np.dot(T_lidar_rightCam,np.linalg.inv(T_lidar_leftCam))
        rvec, tvec = T_leftCam_rightCam[:3, :3], T_leftCam_rightCam[:3, -1]
        angles = euler_from_matrix(rvec)
        angles = np.array([(180.0 / math.pi) * i for i in angles])
        print('')
        print('Lidar based camera calibration extrinsics')
        print('angles -> {}'.format(angles))
        print('tvec -> {}'.format(tvec))


if __name__ == '__main__':
    collect_Data = True
    collect_Data = False

    if collect_Data:
        getData(chess=False)
    else:
        name = 'chess' #works for both
        name = 'charuco' #works for both
        file = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/data/GoodPoints_{}.pkl'.format(name)
        file = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/data/GoodPoints2_{}.pkl'.format(name)

        chess = True if name == 'chess' else False
        calibrator = LiDAR_Camera_Calibration(file=file, chess = chess)
        calibrator.load_points()
        #calibrator.computeTransformation()

        #calibrator.plotData()
        #calibrator.calibrate_3D_3D()

        #calibrator.estimate()
        #3D-2D calibration & results
        #calibrator.calibrate_3D_2D(userRansac=False)
        #calibrator.calibrate_3D_2D(userRansac=True)
        #calibrator.callback()

        #calibrator.combine_both_boards_and_train()
        #calibrator.reproject_on_3D()
        #calibrator.doSomePlots()
        #calibrator.DLT()

        #Calibreate Lidar->left camera,  Lidar->right camera, Lidar->3D points
        calibrator.do_holy_Final_calibration(False)