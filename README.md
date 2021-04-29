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


## License

[![CC0](https://licensebuttons.net/p/zero/1.0/88x31.png)](https://creativecommons.org/publicdomain/zero/1.0/)

CONFIDENTIAL

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
