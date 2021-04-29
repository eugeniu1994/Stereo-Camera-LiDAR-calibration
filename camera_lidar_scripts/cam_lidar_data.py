#!/usr/bin/env python2.7

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


import rospy
from sensor_msgs.msg import Image,CameraInfo, PointCloud2
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import cv2
import numpy as np
import ros_numpy
import cv2.aruco as aruco
import math
import pickle
import open3d
import std_msgs.msg
import sensor_msgs.point_cloud2 as pcl2

import pcl

from pynput.keyboard import Key, Listener

class CamLiDAR(object):
    def __init__(self, image_row='', velodyne_points='', pub_LiDAR='', second_cam = None):
        self.second_cam = second_cam
        self.debug = False #display debug messages
        self.image = None
        self.points = None
        self.bridge = CvBridge()
        self.rate = rospy.Rate(10)

        # Subscribe to topics
        self.image_sub = message_filters.Subscriber(image_row, Image)
        if self.second_cam is not  None:
            self.image_sub2 = message_filters.Subscriber(second_cam, Image)
        self.velodyne_sub = message_filters.Subscriber(velodyne_points, PointCloud2)
        # Publishers
        self.Lidar_pub = rospy.Publisher(pub_LiDAR, PointCloud2, queue_size=10)

        self.image_width = 1936
        self.image_height = 1216
        self.display_image_pattern = False# True

        # Synchronize the topics by time
        self.ats = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.velodyne_sub] if self.second_cam is None else [self.image_sub, self.velodyne_sub,self.image_sub2],
            queue_size=5, slop=2)
        self.ats.registerCallback(self.callback, self.Lidar_pub) if self.second_cam is None else self.ats.registerCallback(self.callback2, self.Lidar_pub)

        self.setType(chess=False, charuco1 = False)
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        self.axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
        self.objp = np.zeros((7*10,3), np.float32)
        self.objp[:,:2] = np.mgrid[0:10,0:7].T.reshape(-1,2)

        #publish intermedier solutions
        self.pub_sol = rospy.Publisher('fitting_solutions', PointCloud2, queue_size=5)
        self.idx = 6
        self.readData = True
        def on_press(key):
            try:
                print('alphanumeric key {0} pressed'.format(key.char))
                if key.char == 's':
                    print('Save data----------------------------------------')

                    with open('/home/eugeniu/cool/cloud_{}.npy'.format(self.idx), 'wb') as f:
                        np.save(f, self.points)
                    cv2.imwrite('/home/eugeniu/cool/left_{}.png'.format(self.idx), self.image)
                    if self.second_cam is not None:
                        cv2.imwrite('/home/eugeniu/cool/right_{}.png'.format(self.idx),
                                    np.asarray(self.image2, dtype=np.uint8))

                    print('data {} saved'.format(self.idx))
                    self.idx += 1
                elif key.char == 't':
                    self.readData = not self.readData

            except AttributeError:
                print('special key {0} pressed'.format(key))

        listener = Listener(on_press=on_press)
        listener.start()

        try:
            rospy.spin()
        except rospy.ROSInterruptException:
            rospy.loginfo('Shutting down')
        cv2.destroyAllWindows()

    def setType(self, chess = False, charuco1=False):
        with open('/home/eugeniu/Desktop/my_data/saved_files/camera_model.pkl', 'rb') as f:
            camera_model = pickle.load(f)
        self.K_left = camera_model['K_left']
        self.K_right = camera_model['K_right']
        self.D_left = camera_model['D_left']
        self.D_right = camera_model['D_right']

        self.K = self.K_right
        self.D = self.D_right

        self.chess = chess
        self.cahruco1 = charuco1
        if self.chess == False:
            if self.cahruco1:
                aruco_dict = aruco.custom_dictionary(0, 4, 1)
                aruco_dict.bytesList = np.empty(shape=(4, 2, 4), dtype=np.uint8)
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
                    markersX=2, markersY=2,
                    markerLength=0.233, markerSeparation=0.233,
                    dictionary=self.ARUCO_DICT)
            else:
                N = 5
                aruco_dict = aruco.custom_dictionary(0, N, 1)
                aruco_dict.bytesList = np.empty(shape=(4, N - 1, N - 1), dtype=np.uint8)
                A = np.array([[0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 1, 0, 1, 0], [0, 1, 1, 1, 0], [0, 1, 0, 1, 0]],dtype=np.uint8)
                aruco_dict.bytesList[0] = aruco.Dictionary_getByteListFromBits(A)
                R = np.array([[1, 1, 1, 1, 0], [1, 0, 0, 1, 0], [1, 1, 1, 0, 0], [1, 0, 0, 1, 0], [1, 0, 0, 0, 1]],dtype=np.uint8)
                aruco_dict.bytesList[1] = aruco.Dictionary_getByteListFromBits(R)
                V = np.array([[1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0]],dtype=np.uint8)
                O = np.array([[0, 1, 1, 1, 0], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [0, 1, 1, 1, 0]],dtype=np.uint8)
                aruco_dict.bytesList[2] = aruco.Dictionary_getByteListFromBits(O)
                aruco_dict.bytesList[3] = aruco.Dictionary_getByteListFromBits(V)

                self.ARUCO_DICT = aruco_dict
                self.board = aruco.GridBoard_create(
                    markersX=2, markersY=2,
                    markerLength=0.126, markerSeparation=0.74,
                    dictionary=self.ARUCO_DICT)

            rospy.loginfo('charuco data loaded')

    def callback(self, image_msg, velodyne_msg, Lidar_pub=None):
        # Extract points data
        try:
            self.points = ros_numpy.point_cloud2.pointcloud2_to_array(velodyne_msg)
            self.points = np.asarray(self.points.tolist())  # X Y Z intensity ring (5)
            # Filter points in front of camera
            inrange = np.where(
                                 (self.points[:, 2] > -1) & (self.points[:, 2] < 2)
                                 & (self.points[:, 0] > 2) & (self.points[:, 2] < 3.5)
                                 & (np.abs(self.points[:, 1]) < 2.5)
                               )

            inrange = np.where(
                (self.points[:, 2] > -2) & (self.points[:, 2] < 3)
                & (self.points[:, 0] > 1) & (self.points[:, 2] < 15.5)
                & (np.abs(self.points[:, 1]) < 3.5)
            )
            self.points = self.points[inrange[0]]

        except Exception, e:
            rospy.logerr(e)
        self.point_cloud_fields = velodyne_msg.fields
        self.publishPoints(velodyne_msg.fields, Lidar_pub,  fitPlane = False)

        #Extract images
        try:
            self.image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        except CvBridgeError, e:
            rospy.logerr(e)
        self.showImageCV(self.image, display_pattern=self.display_image_pattern)

        if self.debug:
            rospy.loginfo('Image:{},  Points:{}'.format(np.shape(self.image), np.shape(self.points)))
            rospy.loginfo('Cam: {}'.format(image_msg.header))
            rospy.loginfo('Lidar: {}'.format(velodyne_msg.fields))
            rospy.loginfo('')

    def callback2(self, image_msg, velodyne_msg, image_msg2, Lidar_pub=None):
        # Extract points data
        try:
            #self.points = ros_numpy.point_cloud2.pointcloud2_to_array(velodyne_msg)
            #self.points = np.asarray(self.points.tolist())  # X Y Z intensity ring (5)

            lidar = pcl2.read_points(velodyne_msg)
            self.points = np.array(list(lidar))

            # Filter points in front of camera
            inrange = np.where(
                                 (self.points[:, 2] > -1.2) & (self.points[:, 2] < 2.2)
                                 & (self.points[:, 1] > 1.5) & (self.points[:, 1] < 10)
                                 & (np.abs(self.points[:, 0]) < 2.5)
                               )

            #used to collect points to project on the whole image
            inrange = np.where(
                (self.points[:, 2] > -2.5) & (self.points[:, 2] < 5.5)
                & (self.points[:, 1] > 1.)
                & (np.abs(self.points[:, 0]) < 35)
            )

            '''inrange = np.where(
                (self.points[:, 2] > -2) & (self.points[:, 2] < 3)
                & (self.points[:, 0] > 1) & (self.points[:, 2] < 15.5)
                & (np.abs(self.points[:, 1]) < 3.5)
            )'''
            self.points = self.points[inrange[0]]
            self.cloud_points = self.points

        except Exception, e:
            rospy.logerr(e)
        self.point_cloud_fields = velodyne_msg.fields
        self.publishPoints(velodyne_msg.fields, Lidar_pub,  fitPlane = False)

        #Extract images
        try:
            self.image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            self.image2 = self.bridge.imgmsg_to_cv2(image_msg2, "bgr8")
        except CvBridgeError, e:
            rospy.logerr(e)
        self.showImageCV(display_pattern=self.display_image_pattern)

        if self.debug:
            rospy.loginfo('Image:{},  Points:{}'.format(np.shape(self.image), np.shape(self.points)))
            rospy.loginfo('Cam: {}'.format(image_msg.header))
            rospy.loginfo('Lidar: {}'.format(velodyne_msg.fields))
            rospy.loginfo('')

    def showImageCV(self, fx=0.4, fy=0.4, display_pattern = False):
        img = np.asarray(self.image, dtype=np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if display_pattern:
            QueryImg = img.copy()
            if self.chess:
                ret, corners = cv2.findChessboardCorners(gray, (10, 7), None)
                if ret == True:
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                    cv2.drawChessboardCorners(QueryImg, (10, 7), corners, ret)

                    ret, self.rvecs, self.tvecs = cv2.solvePnP(self.objp, corners2, self.K, self.D)

                    imgpts, jac = cv2.projectPoints(self.axis, self.rvecs, self.tvecs, self.K, self.D)
                    QueryImg = self.draw(QueryImg, corners2, imgpts)
            else:
                corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.ARUCO_DICT)
                corners, ids, rejectedImgPoints, recoveredIds = aruco.refineDetectedMarkers(
                    image=gray, board=self.board, detectedCorners=corners, detectedIds=ids,
                    rejectedCorners=rejectedImgPoints, cameraMatrix=self.K, distCoeffs=self.D)
                if np.all(ids != None):
                    if len(ids)>2:
                        self.retval, self.rvec, self.tvec = aruco.estimatePoseBoard(corners, ids, self.board, self.K,self.D, None, None)
                        if self.retval:
                            QueryImg = aruco.drawAxis(QueryImg, self.K, self.D, self.rvec, self.tvec, 0.3)
                            QueryImg = aruco.drawDetectedMarkers(QueryImg, corners, ids, borderColor=(0, 0, 255))
                            self.dst, jacobian = cv2.Rodrigues(self.rvec)
                            if self.cahruco1:
                                a,circle_tvec = .33,[]
                                circle_tvec.append(np.asarray(self.tvec).squeeze() + np.dot(self.dst, np.asarray([a, a, 0])))
                                circle_tvec = np.mean(circle_tvec, axis=0)
                                QueryImg = aruco.drawAxis(QueryImg, self.K, self.D, self.rvec,circle_tvec, 0.2)
                                b = .705
                                pts = np.float32([[0, b, 0], [b, 0, 0], [0, -b, 0], [-b, 0, 0]])
                                imgpts, _ = cv2.projectPoints(pts, self.rvec, circle_tvec, self.K, self.D)
                            else:
                                a, circle_tvec, b = .49, [], 1
                                circle_tvec.append(np.asarray(self.tvec).squeeze() + np.dot(self.dst, np.asarray([a, a, 0])))
                                circle_tvec = np.mean(circle_tvec, axis=0)
                                QueryImg = aruco.drawAxis(QueryImg, self.K, self.D, self.rvec,circle_tvec, 0.2)
                                pts = np.float32([[0, b, 0], [b, b, 0], [b, 0, 0], [0, 0, 0]])
                                imgpts, _ = cv2.projectPoints(pts, self.rvec, self.tvec, self.K, self.D)
                            pt_dict = {}
                            for i in range(len(pts)):
                                pt_dict[tuple(pts[i])] = tuple(imgpts[i].ravel())
                            top_right = pt_dict[tuple(pts[0])]
                            bot_right = pt_dict[tuple(pts[1])]
                            bot_left = pt_dict[tuple(pts[2])]
                            top_left = pt_dict[tuple(pts[3])]
                            cv2.circle(QueryImg, top_right, 4, (0, 0, 255), 5)
                            cv2.circle(QueryImg, bot_right, 4, (0, 0, 255), 5)
                            cv2.circle(QueryImg, bot_left, 4, (0, 0, 255), 5)
                            cv2.circle(QueryImg, top_left, 4, (0, 0, 255), 5)

                            QueryImg = cv2.line(QueryImg, top_right, bot_right, (0, 255, 0), 4)
                            QueryImg = cv2.line(QueryImg, bot_right, bot_left, (0, 255, 0), 4)
                            QueryImg = cv2.line(QueryImg, bot_left, top_left, (0, 255, 0), 4)
                            QueryImg = cv2.line(QueryImg, top_left, top_right, (0, 255, 0), 4)
            resized = cv2.resize(QueryImg, None, fx=fx, fy=fy)
        else:
            resized = cv2.resize(gray, None, fx=fx, fy=fy)

        cv2.imshow('Camera image', resized)
        if self.second_cam is not None:
            resized2 = cv2.resize(np.asarray(self.image2, dtype=np.uint8), None, fx=fx, fy=fy)
            cv2.imshow('Camera image 2', resized2)
        k = cv2.waitKey(1)
        if k % 256 == 32:  # SPACE pressed
            print('pressed space')
            with open('/home/eugeniu/cool/cloud_{}.npy'.format(self.idx), 'wb') as f:
                np.save(f, self.cloud_points)
            cv2.imwrite('/home/eugeniu/cool/left_{}.png'.format(self.idx), img)
            if self.second_cam is not None:
                cv2.imwrite('/home/eugeniu/cool/right_{}.png'.format(self.idx), np.asarray(self.image2, dtype=np.uint8))

            print('data {} saved'.format(self.idx))
            self.idx += 1

        elif k == ord('q'):
            cv2.destroyAllWindows()

    def publishPoints(self, fields, Lidar_pub, fitPlane = False):
        if fitPlane:
            # fit plane to pcl
            p = pcl.PointCloud(np.array(self.points[:, :3], dtype=np.float32))
            # RANSAC Plane Segmentation
            inlier, outliner, indices = self.do_ransac_plane_segmentation(p, pcl.SACMODEL_PLANE, pcl.SAC_RANSAC, 0.05)
            #inlier, outliner = np.array(inlier), np.array(outliner)
            #inliers, outliers, indices = self.do_ransac_plane_normal_segmentation(p, 0.1)
            #inlier, outliner = np.array(inliers), np.array(outliers)

            self.cloud_points = self.points[indices]
        else:
            self.cloud_points = self.points

        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'VLS128_Center'

        scaled_polygon_pcl = pcl2.create_cloud(header=header,fields=fields,points=self.cloud_points)
        Lidar_pub.publish(scaled_polygon_pcl)

    def draw(self, img, corners, imgpts):
        corner = tuple(corners[0].ravel())
        cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
        cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
        cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
        return img

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

        indices, coefficients = seg.segment()
        inlier_object = pcl_data.extract(indices, negative=False)
        outlier_object = pcl_data.extract(indices, negative=True)
        return inlier_object, outlier_object, indices

    def do_ransac_plane_normal_segmentation(self, point_cloud, input_max_distance):
        segmenter = point_cloud.make_segmenter_normals(ksearch=50)
        segmenter.set_optimize_coefficients(True)
        segmenter.set_model_type(pcl.SACMODEL_NORMAL_PLANE)  # pcl_sac_model_plane
        segmenter.set_normal_distance_weight(0.1)
        segmenter.set_method_type(pcl.SAC_RANSAC)  # pcl_sac_ransac
        segmenter.set_max_iterations(200)
        segmenter.set_distance_threshold(input_max_distance)  # 0.03)  #max_distance
        indices, coefficients = segmenter.segment()

        print('Model coefficients: ' + str(coefficients[0]) + ' ' + str(
            coefficients[1]) + ' ' + str(coefficients[2]) + ' ' + str(coefficients[3]))

        print('Model inliers: ' + str(len(indices)))

        inliers = point_cloud.extract(indices, negative=False)
        outliers = point_cloud.extract(indices, negative=True)

        return inliers, outliers, indices

if __name__ == '__main__':
    rospy.init_node('testNode', anonymous=True)
    rospy.loginfo('---Node data collection started---')
    rospy.loginfo('---Press space to save the data---')

    right_cam, velodyne_points, pub_LiDAR = '/camera_IDS_Left_4103423533/image_raw', '/lidar_VLS128_Center_13202202695611/velodyne_points', '/syncronizedPoints'
    left_cam = '/camera_IDS_Right_4103423537/image_raw'
    #right_cam = None
    #velodyne_points = '/lidar_VLP16_FrontLeft_AF16713576/velodyne_points'
    node = CamLiDAR(image_row=left_cam, velodyne_points = velodyne_points, pub_LiDAR=pub_LiDAR, second_cam = right_cam)

    def myhook():
        print('ShutDown!!!')
        cv2.destroyAllWindows()
    rospy.on_shutdown(myhook)
    cv2.destroyAllWindows()