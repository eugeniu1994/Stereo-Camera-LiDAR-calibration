import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance_matrix

np.set_printoptions(precision=3, suppress=True)


def axisEqual3D(ax, centers):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    # centers = np.mean(extents, axis=1) if centers is None
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def load_obj(name):
    with open('/home/eugeniu/Desktop/my_data/CameraCalibration/data/saved_files/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


name = 'inside'
# name = 'outside'

# read left and right image
# i = 12
i = 10
i = 14
l = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/data/chess/left/left_{}.png'.format(i)
r = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/data/chess/right/right_{}.png'.format(i)
cloud_file = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/data/chess/cloud_{}.npy'.format(i)

scale = .4
imgLeft, imgRight = cv2.imread(l), cv2.imread(r)
gray_left, gray_right = cv2.cvtColor(imgLeft, cv2.COLOR_BGR2GRAY), cv2.cvtColor(imgRight, cv2.COLOR_BGR2GRAY)
img_shape = (1936, 1216)


# function to rectify images
def rectify_function():
    img_shape = (1936, 1216)
    K_left, D_left, K_right, D_right, R, T = readIntrinsics(name)
    R1, R2, P11, P22, Q, roi_left, roi_right = cv2.stereoRectify(K_left, D_left, K_right, D_right, img_shape, R, T,
                                                                 flags=cv2.CALIB_ZERO_DISPARITY,
                                                                 alpha=-1
                                                                 # alpha=0
                                                                 )

    leftMapX, leftMapY = cv2.initUndistortRectifyMap(
        K_left, D_left, R1,
        P11, img_shape, cv2.CV_32FC1)

    rightMapX, rightMapY = cv2.initUndistortRectifyMap(
        K_right, D_right, R2,
        P22, img_shape, cv2.CV_32FC1)

    # imgL = cv2.remap(imgLeft, leftMapX, leftMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    # imgR = cv2.remap(imgRight, rightMapX, rightMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return leftMapX, leftMapY, rightMapX, rightMapY


def getP(K_left, D_left, K_right, D_right, R, T):
    img_shape = (1936, 1216)
    R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(K_left, D_left, K_right, D_right,
                                                               imageSize=img_shape,
                                                               R=R, T=T,
                                                               flags=cv2.CALIB_ZERO_DISPARITY,
                                                               alpha=-1
                                                               # alpha=0
                                                               )

    return P1, P2


# read intrinsics and extrinsics for inside and outside case
def readIntrinsics(name='inside'):
    camera_model = load_obj('{}_combined_camera_model'.format(name))
    camera_model_rectify = load_obj('{}_combined_camera_model_rectify'.format(name))
    D_left = camera_model['D_left']
    D_right = camera_model['D_right']
    R = camera_model['R']
    T = camera_model['T']
    # T[1:] = 0
    # T[0] = 1.18
    if name == 'inside':
        K_left = np.array([[1366.5, 0, 965.5],
                           [0, 1369.7, 602.0],
                           [0, 0, 1]])
        K_right = np.array([[1369.1, 0, 988.8],
                            [0, 1376.0, 691.3],
                            [0, 0, 1]])
    elif name == 'outside':
        K_left = np.array([[1367.3, 0, 966.2],
                           [0, 1367.4, 604.2],
                           [0, 0, 1]])
        K_right = np.array([[1367.5, 0, 953.],
                            [0, 1367.4, 610.5],
                            [0, 0, 1]])

    return K_left, D_left, K_right, D_right, R, T


# left and right are switched
# K_right, D_right, K_left, D_left, R, T = readIntrinsics(name)
# K_left, D_left, K_right, D_right, R, T = readIntrinsics(name)
# Global variables preset
camera_model = load_obj('{}_combined_camera_model'.format(name))
camera_model_rectify = load_obj('{}_combined_camera_model_rectify'.format(name))
K_left, K_right = camera_model['K_left'], camera_model['K_right']
D_left, D_right = camera_model['D_left'], camera_model['D_right']
R = camera_model['R']
T = camera_model['T']
Q = camera_model_rectify['Q']
print('Translation {}'.format(T[0, 0]))
fx, fy, cx, cy = K_left[0, 0], K_left[1, 1], K_left[0, 2], K_left[1, 2]
undistortAgain = True
if undistortAgain:
    R1, R2, P11, P22, Q, roi_left, roi_right = cv2.stereoRectify(K_left, D_left, K_right, D_right, img_shape, R, T,
                                                                 flags=cv2.CALIB_ZERO_DISPARITY,
                                                                 alpha=-1
                                                                 # alpha=0
                                                                 )

    leftMapX, leftMapY = cv2.initUndistortRectifyMap(
        K_left, D_left, R1,
        P11, img_shape, cv2.CV_32FC1)

    rightMapX, rightMapY = cv2.initUndistortRectifyMap(
        K_right, D_right, R2,
        P22, img_shape, cv2.CV_32FC1)

    imgLeft = cv2.remap(imgLeft, leftMapX, leftMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    imgRight = cv2.remap(imgRight, rightMapX, rightMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


# function to get points left and right
def getPoints(gray_left, gray_right, imgLeft, imgRight):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, (10, 7), None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, (10, 7), None)
    if ret_left is False or ret_right is False:
        print('Cannot detect board in both image')
        return

    corners2_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
    corners2_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)

    cv2.drawChessboardCorners(imgLeft, (10, 7), corners2_left, True)
    cv2.drawChessboardCorners(imgRight, (10, 7), corners2_right, True)

    x_left = np.asarray(corners2_left).squeeze()  # .astype(int)
    x_right = np.asarray(corners2_right).squeeze()  # .astype(int)
    print('x_left->{},  x_right->{}'.format(np.shape(x_left), np.shape(x_right)))

    return x_left, x_right


# function to triangulate points
def triangulate(rectify=False):
    imgLeft, imgRight = cv2.imread(l), cv2.imread(r)
    if rectify:
        leftMapX, leftMapY, rightMapX, rightMapY = rectify_function(K_left, D_left, K_right, D_right, R, T)
        imgLeft = cv2.remap(src=imgLeft, map1=leftMapX, map2=leftMapY,
                            interpolation=cv2.INTER_LINEAR, dst=None, borderMode=cv2.BORDER_CONSTANT)
        imgRight = cv2.remap(src=imgRight, map1=rightMapX, map2=rightMapY,
                             interpolation=cv2.INTER_LINEAR, dst=None, borderMode=cv2.BORDER_CONSTANT)
        gray_left, gray_right = cv2.cvtColor(imgLeft, cv2.COLOR_BGR2GRAY), cv2.cvtColor(imgRight, cv2.COLOR_BGR2GRAY)
        x_left, x_right = getPoints(gray_left, gray_right, imgLeft, imgRight)
    else:
        gray_left, gray_right = cv2.cvtColor(imgLeft, cv2.COLOR_BGR2GRAY), cv2.cvtColor(imgRight, cv2.COLOR_BGR2GRAY)
        x_left, x_right = getPoints(gray_left, gray_right, imgLeft, imgRight)

    R1 = np.array([
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]
    ])
    R2 = R
    t1 = np.array([[0.], [0.], [0.]])
    t2 = T

    P1 = np.hstack([R1.T, -R1.T.dot(t1)])
    P2 = np.hstack([R2.T, -R2.T.dot(t2)])

    P1 = K_left.dot(P1)
    P2 = K_right.dot(P2)

    # P1, P2 = getP(K_left, D_left, K_right, D_right, R, T)

    # Triangulate
    _3d_points = []
    for i, point in enumerate(x_left):
        # print('x_left[i] : {}, x_right[i]: {}'.format(x_left[i],x_right[i]))
        point3D = cv2.triangulatePoints(P1, P2, np.array(x_left[i])[:, np.newaxis],
                                        np.array(x_right[i])[:, np.newaxis]).T
        point3D = point3D[:, :3] / point3D[:, 3:4]

        _3d_points.append(point3D)
    _3d_points = np.array(_3d_points).squeeze()  # ,dtype = np.float32
    print('Triangulate _3d_points -> {}'.format(np.shape(_3d_points)))
    # Reproject back into the two cameras
    rvec1, _ = cv2.Rodrigues(R1.T)  # Change
    rvec2, _ = cv2.Rodrigues(R2.T)  # Change

    p1, _ = cv2.projectPoints(_3d_points[:, :3], rvec1, -t1, K_left, distCoeffs=D_left)
    p2, _ = cv2.projectPoints(_3d_points[:, :3], rvec2, -t2, K_right, distCoeffs=D_right)
    err_left = np.linalg.norm(x_left - p1[0, :])
    err_right = np.linalg.norm(x_right - p2[0, :])
    print('triangulate reprojection err_left:{}, err_right:{}'.format(err_left, err_right))

    p1 = np.array(p1).squeeze().astype(int)
    p2 = np.array(p2).squeeze().astype(int)
    for p in p1:
        cv2.circle(imgLeft, (p[0], p[1]), 5, (0, 255, 0), 5)

    _horizontal = np.hstack(
        (cv2.resize(imgLeft, None, fx=scale, fy=scale), cv2.resize(imgRight, None, fx=scale, fy=scale)))
    cv2.imshow('Triangulin', _horizontal)

    return _3d_points


# points_from_triangulin = triangulate(rectify=False)
# print('points_from_triangulin -> {}'.format(np.shape(points_from_triangulin)))
# dist = distance_matrix(points_from_triangulin, points_from_triangulin)*100
# print('points_from_triangulin dist:{}'.format(dist))

# function to compute depth from disparity
def _3D_fromDisparity():
    gray_left, gray_right = cv2.cvtColor(imgLeft, cv2.COLOR_BGR2GRAY), cv2.cvtColor(imgRight, cv2.COLOR_BGR2GRAY)
    x_left, x_right = getPoints(gray_left, gray_right, imgLeft, imgRight)

    fx, fy, cx, cy = K_left[0, 0], K_left[1, 1], K_left[0, -1], K_left[1, -1]
    baseline = abs(T[0])
    disparity = np.sum(np.sqrt((x_left - x_right) ** 2), axis=1)
    depth = baseline * fx / disparity
    print('disparity -> {}, depth- >{} '.format(np.shape(disparity), np.shape(depth)))
    _3DPoints = []
    for i, pixel in enumerate(x_left):
        u, v = pixel.ravel()
        z = depth[i]
        pt = np.array([u, v, z])
        pt[0] = z * (pt[0] - cx) / fx
        pt[1] = z * (pt[1] - cy) / fy
        _3DPoints.append(pt)
    _3DPoints = np.array(_3DPoints).squeeze()

    # Reproject back
    # Reproject back into the two cameras
    rvec1, _ = cv2.Rodrigues(np.eye(3).T)  # Change
    rvec2, _ = cv2.Rodrigues(R.T)  # Change
    t1 = np.array([[0.], [0.], [0.]])
    t2 = T

    p1, _ = cv2.projectPoints(_3DPoints[:, :3], rvec1, -t1, K_left, distCoeffs=D_left)
    p2, _ = cv2.projectPoints(_3DPoints[:, :3], rvec2, -t2, K_right, distCoeffs=D_right)
    err_left = np.linalg.norm(x_left - p1[0, :])
    err_right = np.linalg.norm(x_right - p2[0, :])
    print('Disparity reprojection err_left:{}, err_right:{}'.format(err_left, err_right))

    p1 = np.array(p1).squeeze().astype(int)
    p2 = np.array(p2).squeeze().astype(int)
    for p in p1:
        cv2.circle(imgLeft, (p[0], p[1]), 5, (0, 255, 0), 5)

    _horizontal = np.hstack(
        (cv2.resize(imgLeft, None, fx=scale, fy=scale), cv2.resize(imgRight, None, fx=scale, fy=scale)))
    cv2.imshow('Disparity', _horizontal)
    return _3DPoints


points_from_disparity = _3D_fromDisparity()
print('points_from_disparity -> {}'.format(np.shape(points_from_disparity)))
dist = distance_matrix(points_from_disparity, points_from_disparity) * 100
print('points_from_disparity dist')
print(dist)

# plot all toghether and compute the distnce between points
fig = plt.figure(figsize=plt.figaspect(1))
ax = plt.axes(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_zlim([-5, 10])
ax.scatter(*points_from_disparity.T, color='b', marker='o', alpha=1, s=10, label='points_from_disparity')
# ax.scatter(*points_from_triangulin.T, color='r', marker='x', alpha=1, s=10, label='points_from_triangulin')

ax.legend()
# axisEqual3D(ax, np.mean(points_from_disparity+points_from_triangulin, axis=0))
cv2.waitKey(0)
plt.show()
cv2.destroyAllWindows()
