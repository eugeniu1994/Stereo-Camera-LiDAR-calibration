import matplotlib.pyplot as plt
import numpy as np
from camera_models import *


def axisEqual3D(ax, data):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(data, axis=0)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

DECIMALS = 2

def world_cam():
    world_origin = np.zeros(3)
    dx, dy, dz = np.eye(3)
    t = np.array([3, -4, 2])
    world_frame = ReferenceFrame(
        origin=world_origin,
        dx=dx,
        dy=dy,
        dz=dz,
        name="World",
    )
    camera_frame = ReferenceFrame(
        origin=t,
        dx=dx,
        dy=dy,
        dz=dz,
        name="Camera",
    )
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    world_frame.draw3d()
    camera_frame.draw3d()
    draw3d_arrow(world_origin, t, color="k", name="t")
    set_xyzlim3d(-3, 3)
    # ax.set_title(f"Camera Translation (t = {t})")
    plt.tight_layout()
    plt.show()

def pin_hole():
    F = 3.0  # focal length
    PX = 2.0  # principal point x-coordinate
    PY = 1.0  # principal point y-coordinate
    THETA_X = np.pi / 2  # roll angle
    THETA_Z = np.pi  # yaw angle
    C = np.array([3, -5, 2])  # camera centre
    IMAGE_HEIGTH = 4
    IMAGE_WIDTH = 6
    R = get_rotation_matrix(theta_x=THETA_X, theta_z=THETA_Z)
    world_origin = np.zeros(3)
    dx, dy, dz = np.eye(3)
    world_frame = ReferenceFrame(
        origin=world_origin,
        dx=dx,
        dy=dy,
        dz=dz,
        name="World",
    )
    camera_frame = ReferenceFrame(
        origin=C,
        dx=R @ dx,
        dy=R @ dy,
        dz=R @ dz,
        name="Camera",
    )
    Z = PrincipalAxis(
        camera_center=camera_frame.origin,
        camera_dz=camera_frame.dz,
        f=F,
    )
    image_frame = ReferenceFrame(
        origin=Z.p - camera_frame.dx * PX - camera_frame.dy * PY,
        dx=R @ dx,
        dy=R @ dy,
        dz=R @ dz,
        name="Image",
    )
    image_plane = ImagePlane(
        origin=image_frame.origin,
        dx=image_frame.dx,
        dy=image_frame.dy,
        heigth=IMAGE_HEIGTH,
        width=IMAGE_WIDTH,
    )
    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca(projection='3d')
    ax.text(*C, 'C')
    world_frame.draw3d()
    camera_frame.draw3d()
    image_frame.draw3d()
    Z.draw3d()
    image_plane.draw3d()
    ax.view_init(elev=30.0, azim=30.0)
    ax.set_title('Pinhole Camera Geometry')
    plt.tight_layout()
    plt.show()

    X = np.array([-1, 2, 3])
    G = GenericPoint(X, name='X')
    L = get_plucker_matrix(C, X)
    X1 = image_frame.origin
    X2 = X1 + image_frame.dx
    X3 = X1 + image_frame.dy
    pi = get_plane_from_three_points(X1, X2, X3)
    x = to_inhomogeneus(L @ pi)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca(projection='3d')
    ax.text(*C, 'C')
    world_frame.draw3d()
    camera_frame.draw3d()
    image_frame.draw3d()
    Z.draw3d()
    image_plane.draw3d()
    G.draw3d(pi, C=C)
    ax.view_init(elev=30.0, azim=30.0)
    plt.tight_layout()
    plt.show()

#pin_hole()

def stereo_Cams():
    tx, ty,tz = -0.965, 0.01, 0 #0.208 #m
    Rx,Ry,Rz = 22.66, -1.0, 0.42  #degree
    world_origin = np.zeros(3)
    dx, dy, dz = np.eye(3)
    t = np.array([3, -4, 2])
    t = -np.array([tx, ty, tz])*10
    left_frame = ReferenceFrame(
        origin=world_origin,
        dx=dx*2,
        dy=-dz*2,
        dz=dy*2,
        name="Left camera",
    )
    right_frame = ReferenceFrame(
        origin=t,
        dx=dx*2,
        dy=-dz*2,
        dz=dy*2,
        name="Right camera",
    )
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    left_frame.draw3d()
    right_frame.draw3d()
    draw3d_arrow(world_origin, t, color="k", alpha=.3)
    #set_xyzlim3d(-3, 3)

    F = 3.0  # focal length
    PX = 2.0  # principal point x-coordinate
    PY = 1.0  # principal point y-coordinate
    THETA_X = np.pi / 2  # roll angle
    THETA_Z = np.pi  # yaw angle
    C = t/2# np.array([3, -5, 2])  # camera centre
    IMAGE_HEIGTH = 6# 4
    IMAGE_WIDTH =  10

    R = get_rotation_matrix(theta_x=THETA_X, theta_z=THETA_Z)
    Z = PrincipalAxis(
        camera_center=left_frame.origin,
        camera_dz=left_frame.dz,
        f=F,
    )
    #Z.draw3d()
    image_plane = ImagePlane(
        origin=np.array([0,18,IMAGE_HEIGTH/2]),
        dx=dx,
        dy=-dz,
        heigth=IMAGE_HEIGTH,
        width=IMAGE_WIDTH,
    )
    image_plane.draw3d()
    image_frame = ReferenceFrame(
        origin=image_plane.origin,
        dx=-R @ dx,
        dy=-R @ dy,
        dz=R @ dz,
        name="Image plane",
    )
    image_frame.draw3d()

    X = np.array([3, 25, -2])
    G = GenericPoint(X, name='X')
    L = get_plucker_matrix(left_frame.origin, X)
    X1 = image_frame.origin
    X2 = X1 + image_frame.dx
    X3 = X1 + image_frame.dy
    pi = get_plane_from_three_points(X1, X2, X3)
    x = to_inhomogeneus(L @ pi)
    G.draw3d(pi, C=left_frame.origin)
    G.draw3d(pi, C=right_frame.origin)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_axis_off()

    #plt.gca().set_xticks([])
    #plt.gca().set_yticks([])
    #plt.gca().set_zticks([])
    axisEqual3D(ax, [t/2])
    plt.show()

#world_cam()
#stereo_Cams()


def car_shape():
    def data_for_cylinder_along_z(center_x, center_y, radius,z_0, height_z):
        z = np.linspace(z_0, height_z, 50)
        theta = np.linspace(0, 2 * np.pi, 50)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = radius * np.cos(theta_grid) + center_x
        y_grid = radius * np.sin(theta_grid) + center_y
        return x_grid, y_grid, z_grid

    import json
    from mpl_toolkits.mplot3d import Axes3D
    import glob
    #files = glob.glob('/home/eugeniu/Desktop/my_data/CameraCalibration/car_models_json/*.json')
    files = glob.glob('/home/eugeniu/Desktop/my_data/CameraCalibration/car_models_json/*leikesasi.json')
    files = glob.glob('/home/eugeniu/Desktop/my_data/CameraCalibration/car_models_json/Skoda_Fabia-2011.json')

    #leikesasi.json
    file = files[0]

    plt.figure(figsize=(20, 10))
    ax = plt.axes(projection='3d')

    Xc, Yc, Zc = data_for_cylinder_along_z(center_x=0, center_y=-2, radius=.5, z_0=3.4, height_z=4)
    ax.plot_surface(Xc, Yc, Zc, alpha=1)

    skip=1
    scale = 5
    with open(file) as json_file:
        data = json.load(json_file)
        print('data -> {}'.format(np.shape(data)))
        vertices = np.array(data['vertices'])*scale
        triangles = np.array(data['faces']) - 1
        print('vertices -> {},  triangles->{}'.format(np.shape(vertices), np.shape(triangles)))


        ax.set_xlim([-15, 15])
        ax.set_ylim([-15, 15])
        ax.set_zlim([-1, 15])
        # ax.plot_trisurf(vertices[::skip, 0], vertices[::skip, 2], triangles[::skip], -vertices[::skip, 1], shade=True, color='grey')
        ax.plot_trisurf(vertices[::skip, 0], vertices[::skip, 2], triangles[::skip], -vertices[::skip, 1], shade=True,
                        color='grey', alpha=.2)

    world_origin = np.zeros(3)
    dx, dy, dz = np.eye(3)
    left_frame = ReferenceFrame(
        origin=world_origin+ [-2.5,2.5,2.5],
        dx=dx * 1.5,
        dy=-dz * 1.5,
        dz=dy * 1.5,
        name="Left cam",
    )
    left_frame.draw3d()
    right_frame = ReferenceFrame(
        origin=world_origin+ [2.5,2.5,2.5],
        dx=dx* 1.5,
        dy=-dz* 1.5,
        dz=dy* 1.5,
        name="Right cam",
    )
    right_frame.draw3d()
    lidar_frame = ReferenceFrame(
        origin=world_origin + [0, -2, 3.8],
        dx=dx * 2,
        dy=dy * 2,
        dz=dz * 2,
        name="LiDAR",
    )
    lidar_frame.draw3d()

    THETA_X = np.pi / 2  # roll angle
    THETA_Z = np.pi  # yaw angle
    C = np.array([3, -5, 2])  # camera centre
    IMAGE_HEIGTH = 10
    IMAGE_WIDTH = 16

    R = get_rotation_matrix(theta_x=THETA_X, theta_z=THETA_Z)
    image_plane = ImagePlane(
        origin=np.array([0, 18, IMAGE_HEIGTH / 2]),
        dx=dx,
        dy=-dz,
        heigth=IMAGE_HEIGTH,
        width=IMAGE_WIDTH,
    )
    #image_plane.draw3d()

    '''X = np.array([1, 20, 5])
    G = GenericPoint(X, name='X')
    X1 = image_plane.origin
    X2 = X1 + image_plane.dx
    X3 = X1 + image_plane.dy
    pi = get_plane_from_three_points(X1, X2, X3)

    G.draw3d(pi, C=left_frame.origin)
    G.draw3d(pi, C=right_frame.origin)
    G.draw3d(pi, C=lidar_frame.origin)'''


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #ax.axis('auto')
    #ax.set_axis_off()
    axisEqual3D(ax, [world_origin])
    plt.show()

    print('here2')

car_shape()

