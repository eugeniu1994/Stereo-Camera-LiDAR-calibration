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


try:
    import rospy
    from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
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
    from pyquaternion import Quaternion
    import matplotlib
    import pandas as pd
except:
    print('Change python version')

from termcolor import colored
from scipy.spatial import distance_matrix
import struct
import rosbag



def getRGBfromI(RGBint):
    blue =  RGBint & 255
    green = (RGBint >> 8) & 255
    red =   (RGBint >> 16) & 255
    #return red, green, blue
    return blue, green, red #return BGR

def getIfromRGB(rgb):
    red = rgb[0]
    green = rgb[1]
    blue = rgb[2]
    #print red, green, blue
    RGBint = (red<<16) + (green<<8) + blue
    return RGBint
def load_obj(name):
    with open('/home/eugeniu/Desktop/my_data/CameraCalibration/data/saved_files/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
def get_z(T_cam_world, T_world_pc, K):
    R = T_cam_world[:3, :3]
    t = T_cam_world[:3, 3]
    proj_mat = np.dot(K, np.hstack((R, t[:, np.newaxis])))
    xyz_hom = np.hstack((T_world_pc, np.ones((T_world_pc.shape[0], 1))))
    xy_hom = np.dot(proj_mat, xyz_hom.T).T
    z = xy_hom[:, -1]
    z = np.asarray(z).squeeze()
    return z
def readCalibration():
    name = 'inside'
    # name = 'outside'
    camera_model = load_obj('{}_combined_camera_model'.format(name))
    camera_model_rectify = load_obj('{}_combined_camera_model_rectify'.format(name))

    K_left = camera_model['K_left']
    K_right = camera_model['K_right']
    D_left = camera_model['D_left']
    D_right = camera_model['D_right']

    K = K_right
    D = D_right

    calib_file = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/combined_extrinsics{}.npz'
    calib_file = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/solvePnP_extrinsicscharuco.npz'

    with open(calib_file, 'r') as f:
        data = f.read().split()
        # print('data:{}'.format(data))
        qx = float(data[0])
        qy = float(data[1])
        qz = float(data[2])
        qw = float(data[3])
        tx = float(data[4])
        ty = float(data[5])
        tz = float(data[6])

    q = Quaternion(qw, qx, qy, qz).transformation_matrix
    q[0, 3], q[1, 3], q[2, 3] = tx, ty, tz
    tvec = q[:3, 3]
    rot_mat = q[:3, :3]
    rvec, _ = cv2.Rodrigues(rot_mat)

    return rvec, tvec,q, K,D
rvec, tvec,q, K, D = readCalibration()


rospy.init_node('testNode', anonymous=True)
global myI
myI = 10
bridge = CvBridge()

display = True
display = False
plotTimeLine = False

useColor = True
#useColor = False

useMeanTime = True  #used the mean time for each laser scan
#useMeanTime = False

velodyne_points = '/lidar_VLS128_Center_13202202695611/velodyne_points'
left_cam = '/camera_IDS_Left_4103423533/image_raw'
left_cam_topic_extras = '/camera_IDS_Left_4103423533/extras'

left_cam = '/camera_IDS_Right_4103423537/image_raw'
left_cam_topic_extras = '/camera_IDS_Right_4103423537/extras'

right_cam = '/camera_IDS_Left_4103423533/image_raw'

Syncronized_Lidar_pub = rospy.Publisher(velodyne_points, PointCloud2, queue_size=200)
Syncronized_Cam_pub = rospy.Publisher(left_cam, Image, queue_size=200)

r,g,b, a = int(0 * 255.0),int(1 * 255.0),int(0 * 255.0), 255
rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
r,g,b, a = int(1 * 255.0),int(0 * 255.0),int(0 * 255.0), 255
rgb_red = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
cmap = matplotlib.cm.get_cmap('hsv')

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

removeShadow = True
#removeShadow = False
global STOP, cloud_points_save, left_image_Save, right_image_Save
def publishSyncronized_msg(cloud_synchronized,image_synchronized, image_synchronized2=None, pixel_opacity = 1):
    try:
        # publish lidar
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'VLS128_Center'
        # columns=["X", "Y", "Z",'rgb',"intens","ring","time"]
        if useColor:
            objPoints_left = cloud_synchronized[:,:3]
            Z = get_z(q, objPoints_left, K)
            cloud_synchronized = cloud_synchronized[Z > 0]
            objPoints_left = objPoints_left[Z > 0]
            points2D, _ = cv2.projectPoints(objPoints_left, rvec, tvec, K, D)
            points2D = np.squeeze(points2D)
            image = bridge.imgmsg_to_cv2(image_synchronized, "bgr8")
            inrange = np.where(
                (points2D[:, 0] >= 0) &
                (points2D[:, 1] >= 0) &
                (points2D[:, 0] < image.shape[1] - 1) &
                (points2D[:, 1] < image.shape[0] - 1)
            )
            points2D = points2D[inrange[0]].round().astype('int')
            cloud_synchronized = cloud_synchronized[inrange[0]]
            #filter again here -> save the closest to the camera

            distance = np.linalg.norm(cloud_synchronized[:, :3], axis=1)

            if removeShadow:
                '''sort points by distance'''
                idx_sorted = np.argsort(distance) #ascending
                idx_sorted = idx_sorted[::-1]     #descending
                cloud_synchronized = cloud_synchronized[idx_sorted]
                points2D = points2D[idx_sorted]
                distance = distance[idx_sorted]


            #cv2.imshow('image ',image)
            #cv2.waitKey(1)

            MIN_DISTANCE, MAX_DISTANCE = np.min(distance), np.max(distance)
            colours = (distance - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE)
            colours = np.asarray([np.asarray(hsv_to_rgb(0.75 * c, np.sqrt(pixel_opacity), 1.0)) for c in colours])
            cols = pixel_opacity * 255 * colours
            # print('colours:{}, cols:{}'.format(np.shape(colours), np.shape(cols)))

            colors_left = image[points2D[:, 1], points2D[:, 0], :]
            colors_left = np.array([getIfromRGB(col) for col in colors_left]).squeeze()

            #greenColors = np.ones(len(cloud_synchronized))*rgb
            cloud_synchronized[:, 3]=colors_left
            image = cv2.Canny(image, 100, 200)
            #for i in range(len(points2D)):
                #cv2.circle(image, tuple(points2D[i]), 2, (0, 255, 0), 1)
                #cv2.circle(image, tuple(points2D[i]), 2, cols[i], -1)

            _pcl = pcl2.create_cloud(header=header, fields=fields, points=cloud_synchronized)
            Syncronized_Lidar_pub.publish(_pcl)
            # publish camera
            Syncronized_Cam_pub.publish(bridge.cv2_to_imgmsg(image))
        else:
            _pcl = pcl2.create_cloud(header=header, fields=fields, points=cloud_synchronized)
            Syncronized_Lidar_pub.publish(_pcl)

            #publish camera
            Syncronized_Cam_pub.publish(image_synchronized)
            cloud_points_save = cloud_synchronized
            left_image_Save = image_synchronized
            right_image_Save = image_synchronized2

    except Exception as e:
        rospy.logerr(e)


def do_job(path, lidar_msgs, cam, cam_right):
    global myI
    print('cam_right -> {}'.format(np.shape(cam_right)))
    print('got path:{}, lidar_msgs:{}, cam:{}'.format(np.shape(path), np.shape(lidar_msgs), np.shape(cam)))
    if useMeanTime:
        for (x1, x2) in path:
            cloud_synchronized = lidar_msgs[x1]
            image_synchronized = cam[x2]
            try:
                image_synchronized2 = cam_right[x2]
            except:
                image_synchronized2 = cam_right[x2-1]
            # print('cloud_synchronized:{}, image_synchronized:{}'.format(np.shape(cloud_synchronized), np.shape(image_synchronized)))
            publishSyncronized_msg(cloud_synchronized,image_synchronized, image_synchronized2)

            l=bridge.imgmsg_to_cv2(image_synchronized, "bgr8")
            r=bridge.imgmsg_to_cv2(image_synchronized2, "bgr8")
            cv2.imshow('left', cv2.resize(l,None,fx=.4,fy=.4))
            cv2.imshow('right', cv2.resize(r,None,fx=.4,fy=.4))

            k = cv2.waitKey(1)
            if k==ord('s'):
                print('Sve cv2')
                print('Saved {}, {}'.format(np.shape(l), np.shape(r)))
                cv2.imwrite('/home/eugeniu/left_{}.png'.format(myI), l)
                cv2.imwrite('/home/eugeniu/right_{}.png'.format(myI), r)
                with open('/home/eugeniu/cloud_{}.npy'.format(myI), 'wb') as f:
                    np.save(f, cloud_synchronized)
                myI+=1

    else:
        _lidar_synchro = []
        lidar_msgs = np.vstack(lidar_msgs)
        print('lidar_msgs -> {}'.format(np.shape(lidar_msgs)))
        '''
        vstack the lidar msg list
        for all unique idx in camera in path
            -take all lidar points that belongs to it
                -publish them toghether
        '''
        unique_cam, indices = np.unique(path[:,1], return_index=True)
        for i,u in enumerate(unique_cam):
            inrange = np.where(path[:,1]==u)  #take all lidar points that belongs to this cam msg
            cloud_synchronized = lidar_msgs[inrange[0]]
            image_synchronized = cam[i]
            #print('cloud_synchronized:{}, image_synchronized:{}'.format(np.shape(cloud_synchronized), np.shape(image_synchronized)))
            publishSyncronized_msg(cloud_synchronized,image_synchronized)

    print(colored('Data published','green'))
    cv2.destroyAllWindows()


k = 0
plot_data_left = deque(maxlen=200)
LiDAR_msg, LeftCam_msg, LeftCam_TimeSeries = [],[],[]
RightCam_msg = []
chessBag  = '/home/eugeniu/chessboard_Lidar_Camera.bag'
charucoBag = '/home/eugeniu/charuco_LiDAR_Camera.bag'
bag = rosbag.Bag(charucoBag)

if display:
    if plotTimeLine:
        fig, (ax, ax2, ax3) = plt.subplots(3, 1)
    else:
        fig, (ax,ax2) = plt.subplots(2, 1)
    ax.grid()
    ax2.grid()
    ax3.grid()
    ppsLine, = ax.plot([0], 'b', label='PPS pulse')
    ax.legend()

import time
import threading

skip = 1
history = []
from pynput.keyboard import Key, Listener

global STOP, cloud_points_save, left_image_Save, right_image_Save
STOP = False
def on_press(key):
    try:
        print('alphanumeric key {0} pressed'.format(key.char))
        if key.char == 's':
            print('Save data----------------------------------------')


        else:
            global STOP
            STOP = True
    except AttributeError:
        print('special key {0} pressed'.format(key))

listener = Listener(on_press=on_press)
listener.start()

for topic, msg, t in bag.read_messages(topics=[left_cam, left_cam_topic_extras, velodyne_points, right_cam]):
    #print('topic -> {}, msg->{}  t->{}'.format(topic, np.shape(msg),t))
    if topic == left_cam_topic_extras:# check pps and apply synchronization
        pps = int(msg.pps)
        m = 'pps->{},  LiDAR->{}, Cam->{}'.format(pps, np.shape(LiDAR_msg), np.shape(LeftCam_msg))
        if pps == 1:
            print(colored(m, 'red'))
            lidar,cam = np.copy(LiDAR_msg),np.copy(LeftCam_msg) #get the copy of current buffer
            cam_right = np.copy(RightCam_msg)
            Cam_Time_series = np.copy(LeftCam_TimeSeries)
            k=0.
            LiDAR_msg, LeftCam_msg, LeftCam_TimeSeries = [], [], [] #clear the storage
            RightCam_msg = []
            #synchronize them
            print('Cam_Time_series -> {}'.format(Cam_Time_series))
            #lidar_msgs = np.array([np.array(list(pcl2.read_points(cloud_msg))).squeeze()[::skip,:] for cloud_msg in lidar]).ravel() # Msg x N x 6
            lidar_msgs = lidar
            LiDAR_Time_series = []

            if useMeanTime:
                for cloud in lidar_msgs:
                    LiDAR_Time_series = np.hstack((LiDAR_Time_series, np.mean(cloud[:, -1])))
            else:
                for cloud in lidar_msgs:
                    LiDAR_Time_series = np.hstack((LiDAR_Time_series, np.asarray(cloud[:,-1]).squeeze()))
            LiDAR_Time_series = np.array(LiDAR_Time_series).squeeze()
            LiDAR_Time_series = np.array([math.fmod(t, 2.0) for t in LiDAR_Time_series])
            print('LiDAR_Time_series -> {}'.format(np.shape(LiDAR_Time_series)))
            print('LiDAR_Time_series -> {}'.format(LiDAR_Time_series[:20]))

            #DTW alignment
            #_, path = fastdtw(LiDAR_Time_series, Cam_Time_series, dist=euclidean)
            #path = np.array(path)

            dist_mat = distance_matrix(LiDAR_Time_series[:, np.newaxis], Cam_Time_series[:, np.newaxis])
            neighbours = np.argsort(dist_mat, axis=1)[:, 0]       #for each lidar msg ge the closest neighbour from camera
            path = np.array([np.linspace(start = 0, stop = len(LiDAR_Time_series)-1, num = len(LiDAR_Time_series), dtype = int), neighbours]).T
            if LiDAR_Time_series[-1] < LiDAR_Time_series[-2]:
                print(colored('delete last element', 'red'))
                path = path[:-2]

            print('path -> {}'.format(np.shape(path)))
            pulish = True
            pulish = False
            start = time.time()
            if pulish:
                do_job(path, lidar_msgs, cam)
                #_thread = threading.Thread(target=do_job,args=(path, lidar_msgs, cam))
                #_thread.daemon = True
                #_thread.start()
            else:
                history.append([path, lidar_msgs, cam, cam_right])

            end = time.time()
            print('the publish took {}'.format(end-start))
            if display:
                ax2.clear()
                x,y = LiDAR_Time_series,Cam_Time_series
                offsetY,offsetX = 3,int(len(path)/2)
                ax2.plot(y + offsetY, c='b', label='Cam-{}'.format(len(y)), linewidth=2)
                ax2.plot(np.linspace(start = 0, stop = len(x)-1, num = len(x), dtype = int)+offsetX,x, c='r', label='LiDAR-{}'.format(len(x)), linewidth=2)
                for (x1, x2) in path:
                    ax2.plot([x1+offsetX, x2], [x[x1], y[x2] + offsetY], c='k', alpha=.5, linewidth=1)
                ax2.grid()
                ax2.legend()

        #else:
            #print(m)
        if display:
            plot_data_left.append(pps)
            ppsLine.remove()
            ppsLine, = ax.plot(plot_data_left, 'b', label='left cam')
            fig.canvas.draw_idle()
            plt.pause(0.001)

    elif topic == left_cam:         #store the camera msgs
        LeftCam_msg.append(msg)
        LeftCam_TimeSeries.append(k)
        k += 0.05  # seconds
    elif topic == velodyne_points:  #store the LiDAR msgs
        fields = msg.fields
        if useColor:
            fields.append(PointField('rgb', 12, PointField.UINT32, 1))


        #LiDAR_msg.append(msg)

        m = np.array(list(pcl2.read_points(msg))).squeeze() #N x 6
        inrange = np.where(m[:, 1] > 1.5)
        LiDAR_msg.append(m[inrange[0]])
    elif topic == right_cam:
        RightCam_msg.append(msg)

        #columns=["X", "Y", "Z","intens","ring","time"]
    if STOP:
        print('Break ---------------------------')
        break

bag.close()
print('Start publishing')
plt.close()
for item in history:
    path, lidar_msgs, cam, cam_right = item
    do_job(path, lidar_msgs, cam, cam_right)
































