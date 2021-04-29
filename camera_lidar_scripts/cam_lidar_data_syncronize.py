

'''#####!/usr/bin/env python2.7'''
try:
    import rospy
    from sensor_msgs.msg import Image, CameraInfo, PointCloud2
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
    import matplotlib.pyplot as plt
    from collections import deque
    from sensor_msgs import point_cloud2
    from sklearn import preprocessing
    from scipy.spatial.distance import euclidean
    from fastdtw import fastdtw
    from testNode.msg import extras
except:
    print('Change python version')

class Camera_LiDAR_syncronizer:
    def __init__(self, display=False):
        self.display = display
        self.bridge = CvBridge()
        self.fig, (self.ax, self.ax2) = plt.subplots(2, 1)
        if display:
            self.ppsLine, = self.ax.plot([0], 'b', label='left cam')

        self.plot_data_left = deque(maxlen=100)
        self.ax.grid()
        self.idxleft = 0
        self.ppsCount = []
        self.camera_image_rostimestamp, self.lidar_rostimestamp = 0, 0
        self.LiDAR_msg, self.leftCam_msg = [], []
        self.Lidar_msg_Modulo = []
        self.extrasTime = []
        # Publishers
        self.Syncronized_Lidar_pub = rospy.Publisher('/syncronized_lidar_VLS128', PointCloud2, queue_size=50)
        self.Syncronized_Cam_pub = rospy.Publisher('/syncronized_left_cam', Image, queue_size=50)
        self.k = 0
        self.useMeanTime = True

    def _callback_left_cam_extras(self, msg):
        self.k += 1
        # print('extras time --->  {}'.format(np.double(msg.timestamp)/ (1e+6)))
        self.pps = int(msg.pps)
        if self.pps == 1:
            if len(self.LiDAR_msg) > 0:
                self.syncronize_msgs()
                self.plotDTW()  # camera timeSeries:->self.ppsCount (microseconds), #lidar timeseries - >self.LiDAR_meanTime (fmod 2 seconds)
                self.LiDAR_msg, self.leftCam_msg, self.extrasTime, self.Lidar_msg_Modulo, self.ppsCount = [], [], [], [], []
                self.idxleft = 0
        else:
            self.extrasTime.append(msg.timestamp)
        if self.display:
            self.plot_data_left.append(self.pps)
            if self.k % 10 == 0:
                self.ppsLine.remove()
                self.plot_data_left.append(self.pps)
                self.ppsLine, = self.ax.plot(self.plot_data_left, 'b', label='left cam')
                self.fig.canvas.draw_idle()
                plt.pause(0.001)

    def syncronize_msgs(self):
        print('pps -> {}, LiDAR_msg:{}, leftCam_msg:{}'.format(self.pps, np.shape(self.LiDAR_msg),
                                                               np.shape(self.leftCam_msg)))
        self.Camera_meanTime = np.array([msg.header.stamp.to_sec() for msg in self.leftCam_msg])
        print('Camera_meanTime:{} -> {}, ppsCount->{}'.format(np.shape(self.Camera_meanTime), self.Camera_meanTime[0],
                                                              np.shape(self.ppsCount)))
        print('ppsCount(s) -> {}'.format(self.ppsCount))
        print('')
        if self.useMeanTime:
            self.LiDAR_meanTime = np.array([np.mean(cloud[:, -1]) for cloud in self.LiDAR_msg])  # each cloud is #Nx6
            print('LiDAR_meanTime:{}  -> {}'.format(np.shape(self.LiDAR_meanTime), self.LiDAR_meanTime))
            self.LiDAR_meanTime = np.array([math.fmod(t, 2.0) for t in self.LiDAR_meanTime])  # each cloud is #Nx6
            print('LiDAR_meanTime mod :{}'.format(self.LiDAR_meanTime))
        else:
            print
        # DTW
        _, self.path = fastdtw(self.LiDAR_meanTime, np.array(self.ppsCount).squeeze(), dist=euclidean)
        self.path = np.array(self.path)
        cleared_path = np.unique(self.path[:, 0], return_index=True)
        self.path = self.path[cleared_path[1]]
        for (x1, x2) in self.path:
            cloud_synchronized = self.LiDAR_msg[x1]
            image_synchronized = self.leftCam_msg[x2]
            self.publishSyncronized_msg(camera_left=image_synchronized, lidar=cloud_synchronized)
        print('------------------------------------------------------------------------')

    def _callback_left_cam_image(self, msg):
        # print('stamp->{}, to_sec->{},  secs->{},  nsecs->{}'.format(msg.header.stamp,msg.header.stamp.to_sec(), msg.header.stamp.secs, msg.header.stamp.nsecs))
        self.leftCam_msg.append(msg)
        self.ppsCount.append(self.idxleft)
        # self.idxleft += (0.05 *  (1e+6)) #seconds converted to microseconds
        self.idxleft += 0.05  # seconds

        '''try:
            #self.camera_image_rostimestamp = np.double(msg.header.stamp.secs)
            #self.left_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
           
            # Camera_meanTime = np.array([np.double(msg.header.stamp.secs)/ (1e+6) for msg in self.leftCam_msg])

        except CvBridgeError, e:
            rospy.logerr(e)'''

    def _callback_lidar(self, msg):
        self.fields = msg.fields
        lidar = pcl2.read_points(msg)
        lidar = np.array(list(lidar)).squeeze()  # Nx6
        # convert lidar timestamp for eah point to microsecond (sec to microsec  (1e+6))
        # lidar[:, -1] *= (1e+6)
        # self.Lidar_msg_Modulo.append([math.fmod(point[-1], 1000000) for point in lidar]) #msg x N x 1
        self.LiDAR_msg.append(lidar)

    def publishSyncronized_msg(self, camera_left, lidar):
        try:
            # publish lidar
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = 'VLS128_Center'

            _pcl = pcl2.create_cloud(header=header, fields=self.fields, points=lidar)
            self.Syncronized_Lidar_pub.publish(_pcl)
            self.Syncronized_Cam_pub.publish(camera_left)

        except Exception as e:
            rospy.logerr(e)

    def plotDTW(self):
        if self.display:
            self.ax2.clear()
            x = self.LiDAR_meanTime  # resulted of fmod 2 seconds for each point
            y = np.array(self.ppsCount).squeeze()  # 0.05, 0.1, ... seconds converted in microseconds

            # x = np.array(preprocessing.normalize([x])).squeeze()
            # y = np.array(preprocessing.normalize([y])).squeeze()

            # print('x:{}'.format(x))
            # print('y:{}'.format(y))

            # distance, self.path = fastdtw(x, y, dist=euclidean)
            # self.ax2.set_xlim([-1, max(len(x), len(y)) + 1])

            offset = 3
            self.ax2.plot(y + offset, c='b', label='Cam-{}'.format(len(y)), linewidth=2)
            self.ax2.plot(x, c='r', label='LiDAR-{}'.format(len(x)), linewidth=2)

            # print(self.path)
            for (x1, x2) in self.path:
                self.ax2.plot([x1, x2], [x[x1], y[x2] + offset], c='k', alpha=.5, linewidth=1)

            self.ax2.legend()


def runNode():
    display = False
    display = True
    rospy.init_node('testNode')
    vis = Camera_LiDAR_syncronizer(display=display)
    left_cam_topic_extras = '/camera_IDS_Left_4103423533/extras'
    # left_cam_topic_extras = '/camera_IDS_Right_4103423537/extras'
    left_Cam_extras = rospy.Subscriber(left_cam_topic_extras, extras, vis._callback_left_cam_extras,queue_size=10000)
    left_cam_topic = '/camera_IDS_Left_4103423533/image_raw'
    # left_cam_topic = '/camera_IDS_Right_4103423537/image_raw'
    left_Cam_image = rospy.Subscriber(left_cam_topic, Image, vis._callback_left_cam_image,queue_size=10000)

    lidar_topic = '/lidar_VLS128_Center_13202202695611/velodyne_points'
    lidar_sub = rospy.Subscriber(lidar_topic, PointCloud2, vis._callback_lidar,queue_size=10000)

    if not display:
        try:
            rospy.spin()
        except rospy.ROSInterruptException:
            rospy.loginfo('Shutting down')
    else:
        plt.show(block=True)


def readBagFiles():
    #import bagpy
    #from bagpy import bagreader

    import pandas as pd

    print('worked')
    '''b = bagreader('/home/eugeniu/arvo/lidar_to_camera_calibration_lidar_pointclouds.bag')
    # replace the topic name as per your need
    LASER_MSG = b.message_by_topic('/lidar_VLS128_Center_13202202695611/velodyne_points')
    print('LASER_MSG--->:{}'.format(LASER_MSG))

    df_laser = pd.read_csv(LASER_MSG)
    print('df_laser:{}'.format(np.shape(df_laser)))'''


#runNode()

#readBagFiles()
'''
rospy.loginfo("image timestamp: %d ns" % image.header.stamp.to_nsec())
rospy.loginfo("scan timestamp: %d ns" % scan.header.stamp.to_nsec())
diff = abs(image.header.stamp.to_nsec() - scan.header.stamp.to_nsec())
rospy.loginfo("diff: %d ns" % diff)
'''
