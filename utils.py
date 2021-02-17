
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
import cv2
from matplotlib import cm
import matplotlib.pyplot as plt
import mpl_toolkits
from mpl_toolkits.mplot3d import Axes3D
import os
import pickle

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
        if save_dir:
            plt.savefig(os.path.join(save_dir, "pattern_centric_view.png"))
    else:
        ax.set_title('Camera Centric View')
        if save_dir:
            plt.savefig(os.path.join(save_dir, "camera_centric_view.png"))
    # plt.show()


def save_obj(obj, name):
    with open('/home/eugeniu/Desktop/my_data/CameraCalibration/data/saved_files/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, protocol=2)
    print('{}.pkl Object saved'.format(name))

def save_csv(obj, name):
    obj.to_csv('/home/eugeniu/Desktop/my_data/CameraCalibration/data/saved_files/{}.csv'.format(name), index=False, header=True)
    print('{}.csv Object saved'.format(name))

def load_obj(name):
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
    with open('/home/eugeniu/Desktop/my_data/CameraCalibration/data/saved_files/'+fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')
