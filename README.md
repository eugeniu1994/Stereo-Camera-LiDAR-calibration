<img src="icon.png" align="right" />

# Mono,Stereo Camera-LiDAR Calibration README [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome#readme)
>A curated list of instructions

This is the Camera-LiDAR calibration master thesis project.
Contains:
- Mono and Stereo camera calibration
- Camera-based 3D reconstruction
- Camera-LiDAR extrinsic calibration
- Camera-LiDAR occlusion handling
- Sensors synchronisation and fusion

## Camera-LiDAR extrinsic calibration
- Data collection, & pointcloud filtering using RANSAC, see cam_lidar_data.py
- Extract point cloud correspondences - see Fit_Points.py script, collect 3D-2D and 3D-3D data points. Estimate transformation between LiDAR-image pixels and LiDAR-3d stereo points.
- see synchronization.py for camera-Lidar synchronization and fusion

## Mono and Stereo camera calibration & 3D reconstruction

- Mono camera calibration using chessboard, see MonoChess.py, example how to run main.py Chess_MonoCalibration() function.

- Mono camera calibration using ChAruco, see MonoCharuo.py, example how to run main.py Charuco_MonoCalibration() function.

- Stereo camera calibration using chessboard, see StereoChess.py, example how to run main.py Chess_StereoCalibration() function.

- Stereo camera calibration using ChAruco, see StereoChess.py, example how to run main.py Charuco_MonoCalibration() function.

- 3D reconstruction using points collected with ChAruco + chessboard, see Charuco_Chess_combine.py script, example to run see combinedChess_and_Charuco() function in the main.py script
