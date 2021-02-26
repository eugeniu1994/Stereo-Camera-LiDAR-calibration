#!/usr/bin/env python2.7

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import cv2
import numpy as np
import yaml
from sensor_msgs.msg import CameraInfo
import cv2.aruco as aruco
import math
import pickle
import open3d

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.pyplot as plt
import mpl_toolkits
from mpl_toolkits.mplot3d import Axes3D

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
        return [X_img_plane, X_triangle, X_center1, X_center2, X_center3, X_center4, X_frame1, X_frame2, X_frame3]
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
    # util function
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

def visualize_views(camera_matrix, rvecs, tvecs,
                    board_width, board_height, square_size,
                    cam_width=64 / 2, cam_height=48 / 2,
                    scale_focal=40, patternCentric=False,
                    figsize=(8, 8), save_dir=None):
    i = 0
    extrinsics = np.zeros((len(rvecs), 6))
    for rot, trans in zip(rvecs, tvecs):
        rospy.loginfo('extrinsics:{}, rot:{}, tran:{}'.format(np.shape(extrinsics), np.shape(rot.flatten()), np.shape(trans.flatten())))
        extrinsics[i] = np.append(rot.flatten(), trans.flatten())
        i += 1
    # The extrinsics  matrix is of shape (N,6) (No default)
    # Where N is the number of board patterns
    # the first 3  columns are rotational vectors
    # the last 3 columns are translational vectors

    fig = plt.figure(figsize=figsize)
    ax = fig.gca(projection='3d')

    ax.set_aspect("auto")

    min_values, max_values = _draw_camera_boards(ax, camera_matrix, cam_width, cam_height,
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
    else:
        ax.set_title('Camera Centric View')
    plt.show()

class CameraNode(object):
    def __init__(self, Stereo = False, image_row='', pub_img='',left_image_row=None, right_image_row=None, flipVertically=True, display_pattern = True, totalMarkers=27):
        self.totalMarkers = totalMarkers
        self.flipVertically = flipVertically
        self.img_counter = 0
        self.img_counterL = 42
        self.img_counterR = 42
        self.display_pattern = True #display_pattern

        self.Stereo = Stereo
        self.image = None
        self.left_image = None
        self.right_image = None
        self.bridge = CvBridge()
        self.rate = rospy.Rate(10)

        #subscribers
        if self.Stereo: #subscribe to both cameras
            cam_left  = message_filters.Subscriber(left_image_row, Image)
            cam_right = message_filters.Subscriber(right_image_row, Image)

            # Synchronize the topics by time
            ats = message_filters.ApproximateTimeSynchronizer(
                [cam_left, cam_right], queue_size=5, slop=0.01)

            ats.registerCallback(self.StereoCallback)

            rospy.loginfo('Subscribed to left and right camera')
        else:#subscribe to single cam
            rospy.Subscriber(image_row, Image, self.callback)
            rospy.loginfo('Subscribed to single camera')

        # Publishers
        self.pub = rospy.Publisher(pub_img, Image, queue_size=10)

        #self.flags = cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FILTER_QUADS
        self.flags = 0  # cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        self.image_width = 1936
        self.image_height = 1216

    def setBoardParams(self, charuco=True, squaresY = 6, squaresX = 9, squareLength = .045, markerLength = 0.034,totalMarkers=0):
        '''charuco = True -> display charuco else display chessboard'''
        self.pattern_columns = squaresX
        self.pattern_rows = squaresY
        self.distance_in_world_units = squareLength
        self.charuco = charuco
        if charuco:
            #self.ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_1000)
            self.ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_4X4_1000)
            # Create constants to be passed into OpenCV and Aruco methods
            self.CHARUCO_BOARD = aruco.CharucoBoard_create(
                squaresX=squaresX, squaresY=squaresY,
                squareLength=squareLength,
                markerLength=markerLength,
                dictionary=self.ARUCO_DICT)

        self.totalMarkers = totalMarkers

    def showImageCV(self,img, fx=0.5, fy=0.5):
        gray = cv2.cvtColor(np.asarray(img, dtype=np.uint8), cv2.COLOR_BGR2GRAY)
        img_copy = np.asarray(img, dtype=np.uint8)
        detected_corners = 0
        position = (20, 30)
        if self.display_pattern:
            if self.charuco:
                corners, ids, rejected = aruco.detectMarkers(image=gray, dictionary=self.ARUCO_DICT)
                found = True if len(corners) > 1 else False
            else:
                found, corners = cv2.findChessboardCorners(gray, (self.pattern_columns, self.pattern_rows), flags=self.flags)

            if found==False:
                #rospy.loginfo('NO MARKER DETECTED')
                resized = cv2.resize(img_copy, None, fx=fx, fy=fy)
            else:
                #rospy.loginfo('MARKER WAS DETECTED')
                if self.charuco:
                    img_copy = aruco.drawDetectedMarkers(image=img_copy, corners=corners)
                else:
                    img_copy = cv2.drawChessboardCorners(img_copy, (self.pattern_columns, self.pattern_rows), corners, found)
                detected_corners = len(corners)
                resized = cv2.resize(img_copy, None, fx=fx, fy=fy)
        else:
            resized = cv2.resize(img_copy, None, fx=fx, fy=fy)

        if self.flipVertically:
            resized = cv2.flip(resized, -1)

        cv2.putText(resized, "Corners:  {}/{}".format(detected_corners, self.totalMarkers),  # text
                    position, cv2.FONT_HERSHEY_SIMPLEX, 1,  # font size
                    (0, 0, 255, 0) if detected_corners > 1 else (255, 0, 0, 0), 2)  # font stroke
        cv2.imshow('Camera image', resized)
        k = cv2.waitKey(1)
        if k % 256 == 32 and detected_corners>1: # SPACE pressed
            img_name = "Mono_img_{}.png".format(self.img_counter)
            cv2.imwrite(img_name, gray)
            self.img_counter += 1
            print("{} written!".format(img_name))
        elif k == ord('i'):
            img_name = "Inside_{}.png".format(self.img_counterL)
            cv2.imwrite(img_name, gray)
            self.img_counterL += 1
            print("{} written!".format(img_name))
        elif k == ord('o'):
            img_name = "Outside_{}.png".format(self.img_counterR)
            cv2.imwrite(img_name, gray)
            self.img_counterR += 1
            print("{} written!".format(img_name))

    def callback(self,msg):
        try:
            self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError, e:
            rospy.logerr(e)

    def StereoCallback(self, cam_left, cam_right):
        try:
            self.left_image = self.bridge.imgmsg_to_cv2(cam_left, 'bgr8')
            self.right_image = self.bridge.imgmsg_to_cv2(cam_right, 'bgr8')
        except CvBridgeError as e:
            rospy.logerr('Error at reading stereo images: {}'.format(e))
            return

    def showImageCV_stereo(self,img_left, img_right, fx=0.45, fy=0.5):
        gray_left = cv2.cvtColor(np.asarray(img_left, dtype=np.uint8), cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(np.asarray(img_right, dtype=np.uint8), cv2.COLOR_BGR2GRAY)

        img_left_copy = np.asarray(img_left, dtype=np.uint8)
        img_right_copy = np.asarray(img_right, dtype=np.uint8)

        detected_corners_left, detected_corners_right = 0, 0
        position = (20, 30)
        cam_left_resized, cam_right_resized = None, None
        if self.display_pattern:
            #Left camera detection------------------------------------------------------------------------
            if self.charuco:
                corners, ids, rejected = aruco.detectMarkers(image=gray_left, dictionary=self.ARUCO_DICT)
                found = True if len(corners) > 1 else False
            else:
                found, corners = cv2.findChessboardCorners(gray_left, (self.pattern_columns, self.pattern_rows), flags=self.flags)

            if found==False:
                resized = cv2.resize(img_left_copy, None, fx=fx, fy=fy)
            else:
                if self.charuco:
                    img_copy = aruco.drawDetectedMarkers(image=img_left_copy, corners=corners)
                else:
                    img_copy = cv2.drawChessboardCorners(img_left_copy, (self.pattern_columns, self.pattern_rows), corners, found)
                detected_corners_left = len(corners)
                resized = cv2.resize(img_copy, None, fx=fx, fy=fy)

            cam_left_resized = resized
            # Right camera detection----------------------------------------------------------------------
            if self.charuco:
                corners, ids, rejected = aruco.detectMarkers(image=gray_right, dictionary=self.ARUCO_DICT)
                found = True if len(corners) > 1 else False
            else:
                found, corners = cv2.findChessboardCorners(gray_right, (self.pattern_columns, self.pattern_rows),flags=self.flags)

            if found == False:
                resized = cv2.resize(img_right_copy, None, fx=fx, fy=fy)
            else:
                if self.charuco:
                    img_copy = aruco.drawDetectedMarkers(image=img_right_copy, corners=corners)
                else:
                    img_copy = cv2.drawChessboardCorners(img_right_copy, (self.pattern_columns, self.pattern_rows),
                                                         corners, found)
                detected_corners_right = len(corners)
                resized = cv2.resize(img_copy, None, fx=fx, fy=fy)

            cam_right_resized = resized

        else:
            resized = cv2.resize(img_right_copy, None, fx=fx, fy=fy)
            cam_right_resized = resized
            resized = cv2.resize(img_left_copy, None, fx=fx, fy=fy)
            cam_left_resized = resized

        if self.flipVertically:
            cam_right_resized = cv2.flip(cam_right_resized, -1)
            cam_left_resized = cv2.flip(cam_left_resized, -1)

        cv2.putText(cam_right_resized, "R-Corners:  {}/{}".format(detected_corners_right, self.totalMarkers),  # text
                    position, cv2.FONT_HERSHEY_SIMPLEX, 1,  # font size
                    (0, 0, 255, 0) if (detected_corners_right) > 1 else (255, 0, 0, 0) , 2)  # font stroke
        cv2.putText(cam_left_resized, "L-Corners:  {}/{}".format(detected_corners_left, self.totalMarkers),  # text
                    position, cv2.FONT_HERSHEY_SIMPLEX, 1,  # font size
                    (0, 0, 255, 0) if (detected_corners_left) > 1 else (255, 0, 0, 0), 2)  # font stroke

        im_h = cv2.hconcat([cam_left_resized, cam_right_resized])
        cv2.imshow('Stereo camera', im_h)
        #cv2.imshow('Left camera', cam_left_resized)
        #cv2.imshow('Right camera', cam_right_resized)
        k = cv2.waitKey(1)
        if k % 256 == 32 and detected_corners_right > 1 and detected_corners_left>1: # SPACE pressed
            img_left_name = "left_{}.png".format(self.img_counter)
            img_right_name = "right_{}.png".format(self.img_counter)
            cv2.imwrite(img_left_name, gray_left)
            cv2.imwrite(img_right_name, gray_right)
            self.img_counter += 1
            print("Both images {} has been written!".format(self.img_counter))

    def start(self):
        rospy.loginfo("Start publishing")
        if self.Stereo:
            while not rospy.is_shutdown():
                if self.left_image is not None and self.right_image is not None:
                    self.showImageCV_stereo(self.left_image, self.right_image)

                self.left_image = None
                self.right_image = None
                self.rate.sleep()
        else:
            while not rospy.is_shutdown():
                if self.image is not None:
                    self.showImageCV(self.image)
                self.image = None
                self.rate.sleep()

    def depth_map(self, imgL, imgR):
        """ Depth map calculation. Works with SGBM and WLS. Need rectified images, returns depth map ( left to right disparity ) """
        # SGBM Parameters -----------------
        window_size = 3  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        numDisparities = 5
        left_matcher = cv2.StereoSGBM_create(
            minDisparity=-1,
            numDisparities=numDisparities * 16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
            blockSize=window_size,
            P1=8 * 3 * window_size,
            # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
            P2=32 * 3 * window_size,
            disp12MaxDiff=12,
            uniquenessRatio=10,
            speckleWindowSize=50,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
        # FILTER Parameters
        lmbda = 80000
        sigma = 1.3

        #lmbda = 50000
        #sigma = 1.

        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
        wls_filter.setLambda(lmbda)

        wls_filter.setSigmaColor(sigma)
        displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
        dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
        displ = np.int16(displ)
        dispr = np.int16(dispr)
        #filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!
        filteredImg = wls_filter.filter(displ, imgL, imgR, dispr)

        filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
        filteredImg = np.uint8(filteredImg)

        return filteredImg

    def CamDetection_old(self, leftFrame):
        gray = cv2.cvtColor(leftFrame, cv2.COLOR_BGR2GRAY)
        QueryImg = leftFrame.copy()

        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.ARUCO_DICT)
        corners, ids, rejectedImgPoints, recoveredIds = aruco.refineDetectedMarkers(
            image=gray, board=self.board, detectedCorners=corners, detectedIds=ids,
            rejectedCorners=rejectedImgPoints, cameraMatrix=self.cameraMatrix, distCoeffs=self.distCoeffs)

        imboard = self.board.draw((1000, 800))
        #cv2.imshow('CharucoBoard target', imboard)
        #cv2.waitKey(0)
        #ids:[2 3 1 0]
        if np.all(ids != None):
            self.rvec, self.tvec = None, None
            self.retval, self.rvec, self.tvec = aruco.estimatePoseBoard(corners, ids, self.board, self.cameraMatrix,self.distCoeffs, self.rvec, self.tvec)

            if self.retval:
                QueryImg = aruco.drawAxis(QueryImg, self.cameraMatrix, self.distCoeffs, self.rvec, self.tvec, 0.3)
                self.dst, jacobian = cv2.Rodrigues(self.rvec)

                #QueryImg = aruco.drawDetectedMarkers(QueryImg, corners, ids, borderColor=(0, 0, 255))
                QueryImg = aruco.drawDetectedMarkers(QueryImg, corners, borderColor=(0, 0, 255))
                '''rotation_vectors, translation_vectors, _objPoints = aruco.estimatePoseSingleMarkers(corners, 0.233, self.cameraMatrix, self.distCoeffs)
                a, circle_tvec, circle_rvec, b = .28, [],[], .8
                """for i,(rvec,tvec) in enumerate(zip(rotation_vectors,translation_vectors)):
                    self.dst, jacobian = cv2.Rodrigues(rvec)
                    if ids[i]==0:
                        QueryImg = aruco.drawAxis(QueryImg, self.cameraMatrix, self.distCoeffs, rvec, tvec, 0.1)
                        circle_tvec.append(np.asarray(tvec).squeeze() + np.dot(self.dst, np.asarray([a,-a, 0])))
    
                    elif ids[i]==29:
                        QueryImg = aruco.drawAxis(QueryImg, self.cameraMatrix, self.distCoeffs, rvec, tvec, 0.1)
                        circle_tvec.append(np.asarray(tvec).squeeze() + np.dot(self.dst, np.asarray([-a,+a, 0])))
                    elif ids[i]==9:
                        QueryImg = aruco.drawAxis(QueryImg, self.cameraMatrix, self.distCoeffs, rvec, tvec, 0.1)
                        circle_tvec.append(np.asarray(tvec).squeeze() + np.dot(self.dst, np.asarray([-a,-a, 0])))
                    elif ids[i]==19:
                        QueryImg = aruco.drawAxis(QueryImg, self.cameraMatrix, self.distCoeffs, rvec, tvec, 0.1)
                        circle_tvec.append(np.asarray(tvec).squeeze() + np.dot(self.dst, np.asarray([a,a, 0])))"""
                a = .33
                circle_tvec.append(np.asarray(self.tvec).squeeze() + np.dot(self.dst, np.asarray([a,a, 0])))
                if len(circle_tvec)>0:
                    circle_rvec = self.rvec
                    circle_tvec = np.mean(circle_tvec, axis=0)
                    QueryImg = aruco.drawAxis(QueryImg, self.cameraMatrix, self.distCoeffs, circle_rvec, circle_tvec, 0.2)

                    #pts = np.float32([[0, .79, 0],[.755, 0, 0],[0, -.705, 0],[-.755, 0, 0]])

                    """visualize_views(camera_matrix=self.cameraMatrix,
                                    rvecs=[np.array(self.rvec).squeeze()],
                                    tvecs=[np.array(self.tvec).squeeze()],
                                    board_width=10,
                                    board_height=10,
                                    square_size=10,
                                    cam_width=10,
                                    cam_height=10,
                                    scale_focal=1,
                                    patternCentric=False
                                    )"""

                    b = .705
                    pts = np.float32([[0, b, 0],[b, 0, 0],[0, -b, 0],[-b, 0, 0]])

                    pt_dict = {}
                    imgpts, _ = cv2.projectPoints(pts, circle_rvec, circle_tvec, self.cameraMatrix, self.distCoeffs)
                    for i in range(len(pts)):
                        pt_dict[tuple(pts[i])] = tuple(imgpts[i].ravel())
                    top_right = pt_dict[tuple(pts[0])]
                    bot_right = pt_dict[tuple(pts[1])]
                    bot_left = pt_dict[tuple(pts[2])]
                    top_left = pt_dict[tuple(pts[3])]
                    cv2.circle(QueryImg,top_right,4,(0,0,255),5)
                    cv2.circle(QueryImg,bot_right,4,(0,0,255),5)
                    cv2.circle(QueryImg, bot_left, 4, (0, 0, 255), 5)
                    cv2.circle(QueryImg, top_left, 4, (0, 0, 255), 5)

                    QueryImg = cv2.line(QueryImg, top_right, bot_right, (0, 255, 0), 4)
                    QueryImg = cv2.line(QueryImg, bot_right, bot_left, (0, 255, 0), 4)
                    QueryImg = cv2.line(QueryImg, bot_left, top_left, (0, 255, 0), 4)
                    QueryImg = cv2.line(QueryImg, top_left, top_right, (0, 255, 0), 4)

                    polygons = np.array([
                        [bot_left, bot_right, top_right, top_left]
                    ])

                    mask = np.zeros_like(QueryImg)
                    cv2.fillPoly(mask, np.int32([polygons]), (255,255,255))

                    segment = cv2.bitwise_and(QueryImg, mask)
                    #cv2.imshow("segment circle " , cv2.resize(segment,None, fx=.4,fy=.4))
                    # Blur the image to reduce noise
                    segment = cv2.cvtColor(segment,cv2.COLOR_BGR2GRAY)
                    img_blur = cv2.medianBlur(segment, 5)
                    # Apply hough transform on the image
                    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, QueryImg.shape[0] / 64, param1=300, param2=50,
                                               minRadius=25, maxRadius=200)
                    # Draw detected circles
                    if circles is not None:
                        circles = np.uint16(np.around(circles))
                        max_r,idx_max = 0, 0
                        for index, i in enumerate(circles[0, :]):
                            if i[2] > max_r:
                                max_r = i[2]
                                idx_max = index
                        i = circles[0, idx_max]
                        cv2.circle(QueryImg, (i[0], i[1]), i[2], (0, 255, 0), 2)
                        cv2.circle(QueryImg, (i[0], i[1]), 3, (255, 0, 0), 3)'''

        return QueryImg

    def CamDetection(self, leftFrame):
        gray = cv2.cvtColor(leftFrame, cv2.COLOR_BGR2GRAY)
        QueryImg = leftFrame.copy()

        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.ARUCO_DICT)
        corners, ids, rejectedImgPoints, recoveredIds = aruco.refineDetectedMarkers(
            image=gray, board=self.board, detectedCorners=corners, detectedIds=ids,
            rejectedCorners=rejectedImgPoints, cameraMatrix=self.cameraMatrix, distCoeffs=self.distCoeffs)

        #imboard = self.board.draw((1000, 800))
        #cv2.imshow('CharucoBoard target', imboard)
        # cv2.waitKey(0)

        if np.all(ids != None):
            self.rvec, self.tvec = None, None
            self.retval, self.rvec, self.tvec = aruco.estimatePoseBoard(corners, ids, self.board, self.cameraMatrix,
                                                                        self.distCoeffs, self.rvec, self.tvec)

            if self.retval:
                QueryImg = aruco.drawAxis(QueryImg, self.cameraMatrix, self.distCoeffs, self.rvec, self.tvec, 0.3)
                self.dst, jacobian = cv2.Rodrigues(self.rvec)

                QueryImg = aruco.drawDetectedMarkers(QueryImg, corners, ids, borderColor=(0, 0, 255))
                #QueryImg = aruco.drawDetectedMarkers(QueryImg, corners, borderColor=(0, 0, 255))
                rotation_vectors, translation_vectors, _objPoints = aruco.estimatePoseSingleMarkers(corners, 0.233, self.cameraMatrix, self.distCoeffs)
                a, circle_tvec, circle_rvec, b = .49, [],[], .8
                circle_tvec.append(np.asarray(self.tvec).squeeze() + np.dot(self.dst, np.asarray([a,a, 0])))
                if len(circle_tvec)>0:
                    circle_rvec = self.rvec
                    circle_tvec = np.mean(circle_tvec, axis=0)
                    QueryImg = aruco.drawAxis(QueryImg, self.cameraMatrix, self.distCoeffs, circle_rvec, circle_tvec, 0.2)

                    b = 1
                    t = -.01
                    pts = np.float32([[0, b, 0],[b, b, 0],[b, 0, 0],[t, t, 0]])

                    pt_dict = {}
                    imgpts, _ = cv2.projectPoints(pts, self.rvec, self.tvec, self.cameraMatrix, self.distCoeffs)
                    for i in range(len(pts)):
                        pt_dict[tuple(pts[i])] = tuple(imgpts[i].ravel())
                    top_right = pt_dict[tuple(pts[0])]
                    bot_right = pt_dict[tuple(pts[1])]
                    bot_left = pt_dict[tuple(pts[2])]
                    top_left = pt_dict[tuple(pts[3])]
                    cv2.circle(QueryImg,top_right,4,(0,0,255),5)
                    cv2.circle(QueryImg,bot_right,4,(0,0,255),5)
                    cv2.circle(QueryImg, bot_left, 4, (0, 0, 255), 5)
                    cv2.circle(QueryImg, top_left, 4, (0, 0, 255), 5)

                    QueryImg = cv2.line(QueryImg, top_right, bot_right, (0, 255, 0), 4)
                    QueryImg = cv2.line(QueryImg, bot_right, bot_left, (0, 255, 0), 4)
                    QueryImg = cv2.line(QueryImg, bot_left, top_left, (0, 255, 0), 4)
                    QueryImg = cv2.line(QueryImg, top_left, top_right, (0, 255, 0), 4)

                    polygons = np.array([
                        [bot_left, bot_right, top_right, top_left]
                    ])

                    mask = np.zeros_like(QueryImg)
                    cv2.fillPoly(mask, np.int32([polygons]), (255,255,255))

                    segment = cv2.bitwise_and(QueryImg, mask)
                    #cv2.imshow("segment circle " , cv2.resize(segment,None, fx=.4,fy=.4))
                    # Blur the image to reduce noise
                    segment = cv2.cvtColor(segment,cv2.COLOR_BGR2GRAY)
                    img_blur = cv2.medianBlur(segment, 5)
                    # Apply hough transform on the image
                    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, QueryImg.shape[0] / 64, param1=300, param2=50,
                                               minRadius=25, maxRadius=200)
                    # Draw detected circles
                    if circles is not None:
                        circles = np.uint16(np.around(circles))
                        max_r,idx_max = 0, 0
                        for index, i in enumerate(circles[0, :]):
                            if i[2] > max_r:
                                max_r = i[2]
                                idx_max = index
                        i = circles[0, idx_max]
                        cv2.circle(QueryImg, (i[0], i[1]), i[2], (0, 255, 0), 2)
                        cv2.circle(QueryImg, (i[0], i[1]), 3, (255, 0, 0), 3)

        return QueryImg

    def depthMap(self,leftFrame,rightFrame, fx=0.35, fy=0.35):
        inter = cv2.INTER_CUBIC
        #inter = cv2.INTER_LANCZOS4
        #inter = cv2.INTER_LINEAR

        #detect the board and color it

        left_rectified = cv2.remap(leftFrame, self.leftMapX, self.leftMapY, inter)
        right_rectified = cv2.remap(rightFrame, self.rightMapX, self.rightMapY, inter)
        #left_rectified = cv2.remap(leftFrame, self.leftMapX, self.leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        #right_rectified = cv2.remap(rightFrame, self.rightMapX, self.rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

        #out = right_rectified.copy()
        #out[:, :, 0] = left_rectified[:, :, 0]
        #out[:, :, 1] = left_rectified[:, :, 1]
        #out[:, :, 2] = right_rectified[:, :, 2]

        #----------------------------------------------------------------
        QueryImg = self.CamDetection(leftFrame)
        cv2.imshow('QueryImg', cv2.resize(QueryImg, None, fx=.6, fy=.6))
        #----------------------------------------------------------------

        gray_left = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)

        disp = self.depth_map(gray_left, gray_right)
        disparity_image = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
        out = disparity_image
        disparity_image = cv2.applyColorMap(disparity_image, cv2.COLORMAP_JET)

        img_top = cv2.hconcat([cv2.resize(left_rectified, None, fx=fx, fy=fy),cv2.resize(right_rectified, None, fx=fx, fy=fy)])
        img_bot = cv2.hconcat([cv2.resize(out, None, fx=fx, fy=fy), cv2.resize(disparity_image, None, fx=fx, fy=fy)])
        Fin_img = cv2.vconcat([img_top, img_bot])
        #cv2.imshow('Result',Fin_img)

        k = cv2.waitKey(1)
        if k % 256 == 32: #space
            rospy.loginfo('Space pressed')
            img = disp
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
            out_fn = 'out.ply'
            self.write_ply(out_fn, out_points, out_colors)
            rospy.loginfo('%s saved' % out_fn)
            pcd = open3d.io.read_point_cloud(out_fn)
            open3d.visualization.draw_geometries([pcd])

    def showImageCV_stereo_depth(self, img_left, img_right, fx=0.45, fy=0.5):
        #img_l = cv2.cvtColor(np.asarray(img_left, dtype=np.uint8), cv2.COLOR_BGR2GRAY)
        #img_r = cv2.cvtColor(np.asarray(img_right, dtype=np.uint8), cv2.COLOR_BGR2GRAY)

        img_l = np.asarray(img_left, dtype=np.uint8)
        img_r = np.asarray(img_right, dtype=np.uint8)

        #img_l = np.asarray(img_right, dtype=np.uint8)
        #img_r = np.asarray(img_left, dtype=np.uint8)

        if self.flipVertically:
            img_l = cv2.flip(img_l, -1)
            img_r = cv2.flip(img_r, -1)

        #self.img_shape = img_l.shape[::-1]
        self.depthMap(img_l,img_r)

    def startValidation(self,charuco):
        rospy.loginfo("Start validation")
        self.charuco = charuco
        if self.Stereo:
            with open('/home/eugeniu/Desktop/my_data/saved_files/camera_model.pkl', 'rb') as f:
                camera_model = pickle.load(f)

            with open('/home/eugeniu/Desktop/my_data/saved_files/camera_model_rectify.pkl', 'rb') as f:
                camera_model_rectify = pickle.load(f)

            self.K_left = camera_model['K_left']
            self.K_right = camera_model['K_right']
            self.D_left = camera_model['D_left']
            self.D_right = camera_model['D_right']
            self.r_left = camera_model['r_left']
            self.r_right = camera_model['r_right']
            self.t_left = camera_model['t_left']
            self.t_right = camera_model['t_right']
            self.R = camera_model['R']
            self.T = camera_model['T']
            self.Q = camera_model_rectify['Q']

            self.leftMapX, self.leftMapY = camera_model_rectify['leftMapX'], camera_model_rectify['leftMapY']
            self.rightMapX, self.rightMapY = camera_model_rectify['rightMapX'], camera_model_rectify['rightMapY']


            '''aruco_dict = aruco.custom_dictionary(0, 4, 1)
            aruco_dict.bytesList = np.empty(shape=(4, 2, 4), dtype=np.uint8)
            # add new marker(s)
            mybits = np.array([[1, 0, 1, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 1, 0], ], dtype=np.uint8)
            aruco_dict.bytesList[0] = aruco.Dictionary_getByteListFromBits(mybits)

            mybits = np.array([[1, 1, 0, 0], [1, 1, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0]], dtype=np.uint8)
            aruco_dict.bytesList[1] = aruco.Dictionary_getByteListFromBits(mybits)

            mybits = np.array([[0, 1, 1, 1], [0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 1, 1]], dtype=np.uint8)
            aruco_dict.bytesList[2] = aruco.Dictionary_getByteListFromBits(mybits)

            mybits = np.array([[0, 0, 1, 1], [0, 1, 0, 0], [0, 1, 1, 0], [1, 1, 1, 1]], dtype=np.uint8)
            aruco_dict.bytesList[3] = aruco.Dictionary_getByteListFromBits(mybits)
            self.ARUCO_DICT = aruco_dict
            self.board = aruco.GridBoard_create(
                markersX=2,markersY=2,
                markerLength=0.233,markerSeparation=0.233,
                dictionary=self.ARUCO_DICT)'''

            # define an empty custom dictionary for markers of size 5
            N = 5
            aruco_dict = aruco.custom_dictionary(0, N, 1)

            # add empty bytesList array to fill with 2 markers later
            aruco_dict.bytesList = np.empty(shape=(4, N-1, N-1), dtype=np.uint8)
            # add new marker(s)

            A = np.array([[0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 1, 0, 1, 0], [0, 1, 1, 1, 0], [0, 1, 0, 1, 0]], dtype=np.uint8)
            aruco_dict.bytesList[0] = aruco.Dictionary_getByteListFromBits(A)
            R = np.array([[1, 1, 1, 1, 0], [1, 0, 0, 1, 0], [1, 1, 1, 0, 0], [1, 0, 0, 1, 0], [1, 0, 0, 0, 1]],dtype=np.uint8)
            aruco_dict.bytesList[1] = aruco.Dictionary_getByteListFromBits(R)
            V = np.array([[1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0]],dtype=np.uint8)
            O = np.array([[0, 1, 1, 1, 0], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1],[1, 0, 0, 0, 1], [0, 1, 1, 1, 0]],dtype=np.uint8)
            aruco_dict.bytesList[2] = aruco.Dictionary_getByteListFromBits(O)
            aruco_dict.bytesList[3] = aruco.Dictionary_getByteListFromBits(V)
            self.ARUCO_DICT = aruco_dict
            self.board = aruco.GridBoard_create(
                markersX=2, markersY=2,
                markerLength=0.126, markerSeparation=0.74,
                dictionary=self.ARUCO_DICT)

            self.cameraMatrix = self.K_right
            self.distCoeffs = self.D_right

            rospy.loginfo('Data loaded')
            while not rospy.is_shutdown():
                if self.left_image is not None and self.right_image is not None:

                    self.showImageCV_stereo_depth(self.left_image, self.right_image)

                self.left_image = None
                self.right_image = None
                self.rate.sleep()
        else:
            while not rospy.is_shutdown():
                if self.image is not None:
                    self.poseEstimation(QueryImg = self.image,charuco=charuco)
                self.rate.sleep()

    def draw(self, img, corners, imgpts):
        corner = tuple(corners[0].ravel())
        img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
        img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
        img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
        return img

    def poseEstimation(self, QueryImg,charuco=True, cameraMatrix = None, distCoeffs = None):
        gray = cv2.cvtColor(QueryImg, cv2.COLOR_BGR2GRAY)
        originalImg = QueryImg.copy()
        cameraMatrix = np.array([[1365.2428, 0., 944.7252],
                                 [0., 1365.0823, 613.1538],
                                 [0., 0., 1.]])
        distCoeffs = np.array([-0.15, 0.1191, -0.0005, -0.0005, -0.0343])

        if charuco:
            # Constant parameters used in Aruco methods
            ARUCO_PARAMETERS = aruco.DetectorParameters_create()
            ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_4X4_1000)

            gridboard = aruco.GridBoard_create(
                markersX=1,markersY=1,markerLength=0.27,markerSeparation=0.01,dictionary=ARUCO_DICT)

            # Detect Aruco markers
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
            # Eliminates markers not part of our board, adds missing markers to the board
            n = len(corners)
            corners, ids, rejectedImgPoints, recoveredIds = aruco.refineDetectedMarkers(
                image=gray,board=gridboard,detectedCorners=corners,detectedIds=ids,
                rejectedCorners=rejectedImgPoints,cameraMatrix=cameraMatrix,distCoeffs=distCoeffs)
            if len(corners) != n:
                rospy.logerr('recovered some points')

            QueryImg = aruco.drawDetectedMarkers(QueryImg, corners, borderColor=(0, 0, 255))
            #draw camera origin on the frame
            QueryImg = aruco.drawAxis(QueryImg, cameraMatrix, distCoeffs, np.zeros(3), np.zeros(3), 3)
            if ids is not None and len(ids) > 0:
                pose, rvec, tvec = aruco.estimatePoseBoard(corners, ids, gridboard, cameraMatrix, distCoeffs, None, None)
                #pose, rvec, tvec = aruco.estimatePoseSingleMarkers(corners, 0.27, cameraMatrix, distCoeffs)
                QueryImg = aruco.drawAxis(QueryImg, cameraMatrix, distCoeffs, rvec, tvec, .3)
        else:

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
            objp = np.zeros((7 * 10, 3), np.float32)
            objp[:, :2] = np.mgrid[0:10, 0:7].T.reshape(-1, 2)
            axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

            img = QueryImg
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (10, 7), None)
            if ret == True:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                # Find the rotation and translation vectors.
                ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, cameraMatrix, distCoeffs)
                # project 3D points to image plane
                imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, cameraMatrix, distCoeffs)
                img = self.draw(img, corners2, imgpts)

                QueryImg = img

        resized = cv2.resize(QueryImg, None, fx=.6, fy=.6)
        resized = cv2.flip(resized, -1)
        cv2.imshow('Camera image', resized)
        k = cv2.waitKey(1)
        if k % 256 == 32:  # SPACE pressed
            img_name = "img_{}.png".format(self.img_counter)
            print('-----------------------------------')
            #print('rvec')
            #print('{}'.format(rvec))
            print('tvecs')
            print('{}'.format(tvecs))
            if tvecs is not None:
                tx, ty, tz = tvecs
                print('tx:{}, ty:{},tz:{}'.format(tx, ty, tz))
                d = math.sqrt(tx * tx + ty * ty + tz * tz)
                d_ = d * 10
                print('distance:{} -> {} cm'.format(d, d_))
                cv2.imwrite(img_name, originalImg)
                print('{} was saved '.format(img_name))
                self.img_counter +=1
            print('-----------------------------------')

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

        verts = verts.reshape(-1, 3)
        colors = colors.reshape(-1, 3)
        verts = np.hstack([verts, colors])
        with open(fn, 'wb') as f:
            f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
            np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


if __name__ == '__main__':
    rospy.init_node('testNode', anonymous=True)
    rospy.loginfo('---Node data collection started---')
    rospy.loginfo('---Press space to save the image---')

    collect_data = False
    image_row = '/image_raw'
    left_image_row = '/left/image_raw'
    right_image_row = '/right/image_raw'

    left_image_row = '/camera/left/image_raw'
    right_image_row = '/camera/right/image_raw'
    change_sides = False
    if change_sides:
        right_image_row = '/camera/left/image_raw'
        left_image_row = '/camera/right/image_raw'

    right_image_row = '/camera_IDS_Left_4103423533/image_raw'
    left_image_row = '/camera_IDS_Right_4103423537/image_raw'
    #image_row = '/camera/left/image_raw'
    pub_img = 'forearm/image_rect'
    camNode = CameraNode(Stereo = True, image_row=image_row, pub_img=pub_img,left_image_row=left_image_row,right_image_row=right_image_row, flipVertically=False, display_pattern = True)

    if collect_data: #used to collect data
        camNode.setBoardParams(charuco=True, totalMarkers=27)
        #camNode.setBoardParams(charuco=True, squaresY=6, squaresX=15, squareLength=.027, markerLength=0.0205,
        #                       totalMarkers=45)
        #camNode.setBoardParams(charuco=True, squaresY=9, squaresX=12, squareLength=.06, markerLength=0.045, totalMarkers=54)
        #camNode.setBoardParams(charuco=False, squaresY = 7, squaresX = 10, squareLength = .1, totalMarkers=70)
        camNode.start()
    else: #estimate pose
        camNode.startValidation(charuco=True)

    def Shutdown():
        rospy.loginfo('------------------Shutdown----------------------')
        cv2.destroyAllWindows()
    rospy.on_shutdown(Shutdown)


