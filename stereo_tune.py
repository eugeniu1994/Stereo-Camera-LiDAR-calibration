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

import cv2
import os
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
import json
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

def depth_map_(imgL, imgR):
    window_size = 15  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=-1,
        numDisparities=20 * 16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
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

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)

    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg)

    return filteredImg

def load_obj(name):
    import pickle
    with open('/home/eugeniu/Desktop/my_data/CameraCalibration/data/saved_files/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def write_ply(fn, verts, colors):
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

def view():
    import glob
    import open3d
    file = glob.glob('/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/data/*.ply')
    file = glob.glob('/home/eugeniu/Desktop/my_data/CameraCalibration/scripts/*.ply')
    for i, file_path in enumerate(file):
        print("{} Load a ply point cloud, print it, and render it".format(file_path))
        pcd = open3d.io.read_point_cloud(file_path)
        open3d.visualization.draw_geometries([pcd])

# Global variables preset
img_shape = (1936, 1216)
name = 'outside'
camera_model = load_obj('{}_combined_camera_model'.format(name))
camera_model_rectify = load_obj('{}_combined_camera_model_rectify'.format(name))
K_left, K_right = camera_model['K_left'], camera_model['K_right']
D_left, D_right = camera_model['D_left'], camera_model['D_right']
R = camera_model['R']
T = camera_model['T']
Q = camera_model_rectify['Q']
print('Translation {}'.format(T[0, 0]))
fx, fy, cx, cy = K_left[0, 0], K_left[1, 1], K_left[0, 2], K_left[1, 2]

def create_point_cloud(depth_image, colors):
    shape = depth_image.shape
    rows = shape[0]
    cols = shape[1]
    h, w, d = np.shape(colors)
    points = np.zeros((rows * cols, 3), np.float32);

    bytes_to_units = (1.0 / 256.0);
    # Linear iterator for convenience
    i = 0
    # For each pixel in the image...
    for r in range(0, rows):
        for c in range(0, cols):
            # Get the depth in bytes
            depth = depth_image[r, c];  # depth_image[r, c, 0];

            # If the depth is 0x0 or 0xFF, its invalid.
            # By convention it should be replaced by a NaN depth.
            if (depth > 0 and depth < 255):
                # The true depth of the pixel in units
                z = depth * bytes_to_units;

                # Get the x, y, z coordinates in units of the pixel
                points[i, 0] = (c - cx) / fx * z;
                points[i, 1] = (r - cy) / fy * z;
                points[i, 2] = z
            else:
                # Invalid points have a NaN depth
                points[i, 2] = np.nan;
            i = i + 1

    points = points.reshape(h, w, d)
    print('points:{}, colors:{}'.format(np.shape(points), np.shape(colors)))

    out_fn = 'create_point_cloud.ply'
    # filter by min disparity
    mask = disparity > disparity.min()
    out_points = points[mask]
    out_colors = colors[mask]
    idx = np.fabs(out_points[:, 0]) < 15  # 10.5 # filter by dimension
    out_points = out_points[idx]
    out_colors = out_colors.reshape(-1, 3)
    out_colors = out_colors[idx]
    write_ply(out_fn, out_points, out_colors)
    return points

def point_cloud(depth, colors):
    """Transform a depth image into a point cloud with one point for each
    pixel in the image, using the camera transform for a camera
    centred at cx, cy with field of view fx, fy.

    depth is a 2-D ndarray with shape (rows, cols) containing
    depths from 1 to 254 inclusive. The result is a 3-D array with
    shape (rows, cols, 3). Pixels with invalid depth in the input have
    NaN for the z-coordinate in the result.

    """
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    valid = (depth > 0) & (depth < 255)
    z = np.where(valid, depth / 256.0, np.nan)
    x = np.where(valid, z * (c - cx) / fx, 0)
    y = np.where(valid, z * (r - cy) / fy, 0)

    points = np.dstack((x, y, z))

    print('points:{}, colors:{}'.format(np.shape(points), np.shape(colors)))

    reflect_matrix = np.identity(3)  # reflect on x axis
    reflect_matrix[0] *= -1
    points = np.matmul(points, reflect_matrix)

    out_fn = 'point_cloud.ply'
    # filter by min disparity
    mask = disparity > disparity.min()
    out_points = points[mask]
    out_colors = colors[mask]
    idx = np.fabs(out_points[:, -1]) < 50  # 10.5 # filter by dimension
    print('out_points:{}'.format(np.shape(out_points)))

    out_points = out_points[idx]
    out_colors = out_colors.reshape(-1, 3)
    out_colors = out_colors[idx]
    write_ply(out_fn, out_points, out_colors)

    # reproject on the image -----------------------------------
    reflected_pts = np.matmul(out_points, reflect_matrix)
    projected_img, _ = cv2.projectPoints(reflected_pts, np.identity(3), np.array([0., 0., 0.]), K_left, D_left)
    projected_img = projected_img.reshape(-1, 2)

    blank_img = np.zeros(colors.shape, 'uint8')
    img_colors = colors[mask][idx].reshape(-1, 3)
    for i, pt in enumerate(projected_img):
        pt_x = int(pt[0])
        pt_y = int(pt[1])
        if pt_x > 0 and pt_y > 0:
            # use the BGR format to match the original image type
            col = (int(img_colors[i, 2]), int(img_colors[i, 1]), int(img_colors[i, 0]))
            cv2.circle(blank_img, (pt_x, pt_y), 1, col)

    return blank_img, out_points


l = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/left_0.png'
r = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/right_0.png'
i=11
l = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/cool/left_{}.png'.format(i)
r = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/cool/right_{}.png'.format(i)
imgLeft = cv2.imread(l, 0)
imgRight = cv2.imread(r, 0)

print('imgLeft:{}, imgRight:{}'.format(np.shape(imgLeft), np.shape(imgRight)))
image_shape = np.shape(imgLeft)
print('image_shape:{}'.format(image_shape))
undistortAgain = True# False
if undistortAgain:
    R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(K_left, D_left, K_right, D_right, img_shape, R, T,
                                                               flags=cv2.CALIB_ZERO_DISPARITY,
                                                               # alpha=-1
                                                               alpha=0
                                                               )

    leftMapX, leftMapY = cv2.initUndistortRectifyMap(
        K_left, D_left, R1,
        P1, img_shape, cv2.CV_32FC1)

    rightMapX, rightMapY = cv2.initUndistortRectifyMap(
        K_right, D_right, R2,
        P2, img_shape, cv2.CV_32FC1)

    width_left, height_left = imgLeft.shape[:2]
    width_right, height_right = imgRight.shape[:2]

    imgL = cv2.remap(imgLeft, leftMapX, leftMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    imgR = cv2.remap(imgRight, rightMapX, rightMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
else:
    imgL = imgLeft
    imgR = imgRight
# Depth map function
global minDisparity, numDisparities, blockSize, P1, P2, disp12MaxDiff, uniquenessRatio, speckleWindowSize, speckleRange, preFilterCap, loading_settings
minDisparity, numDisparities, blockSize, P1, P2, disp12MaxDiff, uniquenessRatio, speckleWindowSize, speckleRange, preFilterCap = -1, 10, 15, 8, 32, 12, 10, 50, 32, 63
lmbda, sigma = 80000, 1.3


def stereo_depth_map(imgL, imgR):
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=minDisparity,
        numDisparities=numDisparities * 16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=blockSize,
        P1=P1 * 3 * blockSize,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=P2 * 3 * blockSize,
        disp12MaxDiff=disp12MaxDiff,
        uniquenessRatio=uniquenessRatio,
        speckleWindowSize=speckleWindowSize,
        speckleRange=speckleRange,
        preFilterCap=preFilterCap,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)

    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)
    #filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg)

    #stereo = cv2.StereoBM_create(numDisparities=numDisparities * 16, blockSize=blockSize)
    #filteredImg = stereo.compute(imgL,imgR)

    return filteredImg


tune = True
if tune:
    cv2.imshow('left', cv2.resize(imgL, None, fx=.4, fy=.4))
    cv2.imshow('right', cv2.resize(imgR, None, fx=.4, fy=.4))
    cv2.waitKey()
    cv2.destroyAllWindows()

    # Draw left image and depth map
    axcolor = 'lightgoldenrodyellow'
    fig = plt.subplots(1, 2)
    plt.subplots_adjust(left=0.1, bottom=0.45)
    plt.subplot(1, 2, 1)
    dmObject = plt.imshow(imgL, 'gray')
    plt.xticks([]), plt.yticks([])
    saveax = plt.axes([0.3, 0.41, 0.15, 0.04])  # stepX stepY width height
    buttons = Button(saveax, 'Save settings', color=axcolor, hovercolor='0.975')

    def save_map_settings(event):
        buttons.label.set_text("Saving...")
        print('Saving to file...')
        result = json.dumps(
            {'preFilterCap': preFilterCap, 'speckleRange': speckleRange, 'speckleWindowSize': speckleWindowSize, \
             'uniquenessRatio': uniquenessRatio, 'disp12MaxDiff': disp12MaxDiff, 'P2': P2, \
             'minDisparity': minDisparity, 'lmbda': lmbda, 'sigma': sigma, \
             'P1': P1, 'blockSize': blockSize, 'numDisparities': numDisparities}, \
            sort_keys=True, indent=4, separators=(',', ':'))
        fName = '3dmap_set.txt'
        f = open(str(fName), 'w')
        f.write(result)
        f.close()
        buttons.label.set_text("Save to file")
        print('Settings saved to file ' + fName)


    buttons.on_clicked(save_map_settings)

    loadax = plt.axes([0.5, 0.41, 0.15, 0.04])  # stepX stepY width height
    buttonl = Button(loadax, 'Load settings', color=axcolor, hovercolor='0.975')


    def load_map_settings(event):
        global loading_settings
        loading_settings = 1
        fName = '3dmap_set.txt'
        print('Loading parameters from file...')
        buttonl.label.set_text("Loading...")
        f = open(fName, 'r')
        data = json.load(f)

        sminDisparity.set_val(data['minDisparity'])
        snumDisparities.set_val(data['numDisparities'])
        sblockSize.set_val(data['blockSize'])
        sP1.set_val(data['P1'])
        sP2.set_val(data['P2'])
        sdisp12MaxDiff.set_val(data['disp12MaxDiff'])
        suniquenessRatio.set_val(data['uniquenessRatio'])
        sspeckleWindowSize.set_val(data['speckleWindowSize'])
        sspeckleRange.set_val(data['speckleRange'])
        spreFilterCap.set_val(data['preFilterCap'])

        f.close()
        buttonl.label.set_text("Load settings")
        print('Parameters loaded from file ' + fName)
        print('Redrawing depth map with loaded parameters...')
        loading_settings = 0
        update(0)
        print('Done!')


    buttonl.on_clicked(load_map_settings)

    # Building Depth Map for the first time
    disparity = stereo_depth_map(imgL, imgR)

    plt.subplot(1, 2, 2)
    dmObject = plt.imshow(disparity, aspect='equal', cmap='gray')  # , cmap='jet'
    plt.xticks([]), plt.yticks([])
    plt.colorbar(dmObject)

    SWSaxe = plt.axes([0.15, 0.01, 0.7, 0.025], facecolor=axcolor)  # stepX stepY width height
    PFSaxe = plt.axes([0.15, 0.05, 0.7, 0.025], facecolor=axcolor)  # stepX stepY width height
    PFCaxe = plt.axes([0.15, 0.09, 0.7, 0.025], facecolor=axcolor)  # stepX stepY width height
    MDSaxe = plt.axes([0.15, 0.13, 0.7, 0.025], facecolor=axcolor)  # stepX stepY width height
    NODaxe = plt.axes([0.15, 0.17, 0.7, 0.025], facecolor=axcolor)  # stepX stepY width height
    TTHaxe = plt.axes([0.15, 0.21, 0.7, 0.025], facecolor=axcolor)  # stepX stepY width height
    URaxe = plt.axes([0.15, 0.25, 0.7, 0.025], facecolor=axcolor)  # stepX stepY width height
    SRaxe = plt.axes([0.15, 0.29, 0.7, 0.025], facecolor=axcolor)  # stepX stepY width height
    SPWSaxe = plt.axes([0.15, 0.33, 0.7, 0.025], facecolor=axcolor)  # stepX stepY width height
    preFilterCapaxe = plt.axes([0.15, 0.37, 0.7, 0.025], facecolor=axcolor)  # stepX stepY width height

    sminDisparity = Slider(SWSaxe, 'minDisparity', -10, 100, valinit=minDisparity)
    snumDisparities = Slider(PFSaxe, 'numDisparities', 2, 512, valinit=numDisparities)
    sblockSize = Slider(PFCaxe, 'blockSize', 3, 128, valinit=blockSize)
    sP1 = Slider(MDSaxe, 'P1', 8, 64, valinit=P1)
    sP2 = Slider(NODaxe, 'P2', 8, 64, valinit=P2)
    sdisp12MaxDiff = Slider(TTHaxe, 'disp12MaxDiff', 4, 256, valinit=disp12MaxDiff)
    suniquenessRatio = Slider(URaxe, 'uniquenessRatio', 5, 100, valinit=uniquenessRatio)
    sspeckleWindowSize = Slider(SRaxe, 'speckleWindowSize', 4, 256, valinit=speckleWindowSize)
    sspeckleRange = Slider(SPWSaxe, 'speckleRange', 8, 256, valinit=speckleRange)
    spreFilterCap = Slider(preFilterCapaxe, 'preFilterCap', 8, 256, valinit=preFilterCap)


    def update(val):
        global minDisparity, numDisparities, blockSize, P1, P2, disp12MaxDiff, uniquenessRatio, speckleWindowSize, speckleRange, preFilterCap, loading_settings
        minDisparity = int(sminDisparity.val)
        numDisparities = int(snumDisparities.val)
        blockSize = int(sblockSize.val)
        P1 = int(sP1.val)
        P2 = int(sP2.val)
        disp12MaxDiff = int(sdisp12MaxDiff.val)
        uniquenessRatio = int(suniquenessRatio.val)
        speckleWindowSize = int(sspeckleWindowSize.val)
        speckleRange = int(sspeckleRange.val)
        preFilterCap = int(spreFilterCap.val)
        if (loading_settings == 0):
            print('Rebuilding depth map')
            disparity = stereo_depth_map(imgL, imgR)
            dmObject.set_data(disparity)
            print('Redraw depth map')
            plt.draw()


    sminDisparity.on_changed(update)
    snumDisparities.on_changed(update)
    sblockSize.on_changed(update)
    sP1.on_changed(update)
    sP2.on_changed(update)
    sdisp12MaxDiff.on_changed(update)
    suniquenessRatio.on_changed(update)
    sspeckleWindowSize.on_changed(update)
    sspeckleRange.on_changed(update)
    spreFilterCap.on_changed(update)
    plt.show()

print('create disparity map')
# disparity = depth_map(imgL, imgR)
disparity = stereo_depth_map(imgL, imgR)
if undistortAgain:
    color_img = cv2.remap(cv2.imread(l), leftMapX, leftMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
else:
    l = '/home/eugeniu/catkin_ws/src/testNode/CAMERA_CALIBRATION/cool/left_{}.png'.format(i)
    color_img = cv2.imread(l)

colors = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
# points = create_point_cloud(disparity.copy(),colors)
img, out_points = point_cloud(disparity.copy(), colors)
print('out_points:{}'.format(np.shape(out_points)))

cv2.imshow("disparity SGBM", cv2.resize(disparity, None, fx=.4, fy=.4))
# cv2.imshow("left", cv2.resize(imgL, None, fx = .4, fy = .4))
# cv2.imshow("right", cv2.resize(imgR, None, fx = .4, fy = .4))
cv2.imshow("color_img", cv2.resize(color_img, None, fx=.4, fy=.4))
cv2.imshow("reprojected", cv2.resize(img, None, fx=.4, fy=.4))

cv2.waitKey(0)

skip = 250
data = out_points[::skip, :]
print('data:{}'.format(np.shape(data)))

'''fig = plt.figure(figsize=plt.figaspect(1))
ax = plt.axes(projection='3d')
#ax.set_axis_off()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# Color map for the points
cmap = matplotlib.cm.get_cmap('hsv')
colors = cmap(data[:, -1] / np.max(data[:, -1]))
ax.scatter(*data.T, c = colors)
#ax.scatter(data[:,0],data[:,1],data[:,2])
plt.show()'''

view()
cv2.destroyAllWindows()
