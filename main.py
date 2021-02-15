'''
    Software License Agreement (BSD License)
 *
 *  Copyright (c) 2021, Eugeniu Vezeteu
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Eugeniu Vezeteu nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.

'''

import glob

from CameraCalibration.scripts.Charuco_Chess_combine import MonoCombinedCalibration, CombinedCalibration
from CameraCalibration.scripts.MonoCharuco import MonoCharuco_Calibrator
from CameraCalibration.scripts.StereoCharuco import StereoCharuco_Calibrator
from utils import  *
from CameraCalibration.scripts.MonoChess import MonoChess_Calibrator
from CameraCalibration.scripts.StereoChess import StereoChess_Calibrator

def Chess_MonoCalibration():
    def calibrate_(leftCam=False, outside=False):
        calibrator = MonoChess_Calibrator(pattern_type="chessboard", pattern_rows=10,
                                          pattern_columns=7, distance_in_world_units=10,  # square is 10 cm
                                          figsize=(14, 10))
        if leftCam:
            if outside:
                images = glob.glob(
                    '/home/eugeniu/Desktop/my_data/CameraCalibration/data/car_cam_data/chess/outside/Left/*.png')
                calibrator.name = 'chess_outside_left'
            else:
                images = glob.glob(
                    '/home/eugeniu/Desktop/my_data/CameraCalibration/data/car_cam_data/chess/inside/Left/*.png')
                calibrator.name = 'chess_inside_left'
        else:
            if outside:
                images = glob.glob(
                    '/home/eugeniu/Desktop/my_data/CameraCalibration/data/car_cam_data/chess/outside/Right/*.png')
                calibrator.name = 'chess_outside_right'
            else:
                images = glob.glob(
                    '/home/eugeniu/Desktop/my_data/CameraCalibration/data/car_cam_data/chess/inside/Right/*.png')
                calibrator.name = 'chess_inside_right'

        calibrator.doStuff(images=images, project=True)
        return calibrator

    # ---------Work with right camera ----------------------------------------------
    # calibrate right camera outside
    print('=== Calibrate right camera outside ===')
    calibrator = calibrate_(leftCam=False, outside=True)

    # calibrate right camera inside
    print('=== Calibrate right camera inside ===')
    calibrator = calibrate_(leftCam=False, outside=False)

    ideal_calibration_right = load_obj(name='chess_outside_right')
    calibrator.adjustCalibration(images=calibrator.images, K=ideal_calibration_right['K'],
                                 D=ideal_calibration_right['D'])

    # ---------Work with left camera -----------------------------------------------
    # calibrate left camera outside
    print('=== Calibrate left camera outside ===')
    calibrator = calibrate_(leftCam=True, outside=True)

    # calibrate left camera inside
    print('=== Calibrate left camera inside ===')
    calibrator = calibrate_(leftCam=True, outside=False)

    ideal_calibration_left = load_obj(name='chess_outside_left')
    calibrator.adjustCalibration(images=calibrator.images, K=ideal_calibration_left['K'],
                                 D=ideal_calibration_left['D'])

    print('Mono camera calibration done')

def Chess_StereoCalibration():
    def stereo_calibrate(outside=True):
        if outside:
            images = '/home/eugeniu/Desktop/my_data/CameraCalibration/data/car_cam_data/chess/outside/Stereo'
            name = 'outside'
        else:
            images = '/home/eugeniu/Desktop/my_data/CameraCalibration/data/car_cam_data/chess/inside/Stereo'
            name = 'inside'

        calibrator = StereoChess_Calibrator(images)
        calibrator.name = name
        calibrator.readMonoData()
        calibrator.read_images(test=False)
        calibrator.calibrationReport()
        # calibrator.stereoCalibrate()
        # calibrator.readStereoData()

    #Calibrate outside--------
    #print('Stereo Calibrate outside')
    #stereo_calibrate(outside=True)

    #Calibrate inside---------
    #print('Stereo Calibrate inside')
    #stereo_calibrate(outside=False)


    def testCalibration(outside = True):
        if outside:
            images = '/home/eugeniu/Desktop/my_data/CameraCalibration/data/car_cam_data/chess/outside/Stereo'
            name = 'outside'
        else:
            images = '/home/eugeniu/Desktop/my_data/CameraCalibration/data/car_cam_data/chess/inside/Stereo'
            name = 'inside'

        images = '/home/eugeniu/Desktop/Stereo_test'
        calibrator = StereoChess_Calibrator(images)
        calibrator.name = name
        calibrator.readMonoData()
        calibrator.read_images(test=True)
        calibrator.readStereoData()
        calibrator.depth()

    #Test calibration outside-----
    print('Test calibration outside')
    testCalibration(outside=True)

    #Test calibration inside
    print('Test calibration inside')
    testCalibration(outside=False)

def Charuco_MonoCalibration():
    def calibrate_(leftCam=False, outside=False):
        if leftCam:
            if outside:
                images = glob.glob(
                    '/home/eugeniu/Desktop/my_data/CameraCalibration/data/car_cam_data/charuco/outside/Left/*.png')
                name = 'charuco_outside_left'
            else:
                images = glob.glob(
                    '/home/eugeniu/Desktop/my_data/CameraCalibration/data/car_cam_data/charuco/inside/Left/*.png')
                name = 'charuco_inside_left'
        else:
            if outside:
                images = glob.glob(
                    '/home/eugeniu/Desktop/my_data/CameraCalibration/data/car_cam_data/charuco/outside/Right/*.png')
                name = 'charuco_outside_right'
            else:
                images = glob.glob(
                    '/home/eugeniu/Desktop/my_data/CameraCalibration/data/car_cam_data/charuco/inside/Right/*.png')
                name = 'charuco_inside_right'

        calibrator = MonoCharuco_Calibrator(name=name)
        calibrator.doStuff(images=images, project=True, single_flag=False)
        return calibrator

    # ---------Work with right camera ----------------------------------------------
    # calibrate right camera outside
    print('=== Calibrate right camera outside ===')
    calibrator = calibrate_(leftCam=False, outside=True)

    # calibrate right camera inside
    print('=== Calibrate right camera inside ===')
    calibrator = calibrate_(leftCam=False, outside=False)

    ideal_calibration_right = load_obj(name='charuco_outside_right')
    calibrator.adjustCalibration(K=ideal_calibration_right['K'],
                                 D=ideal_calibration_right['D'])

    # ---------Work with left camera -----------------------------------------------
    # calibrate left camera outside
    print('=== Calibrate left camera outside ===')
    calibrator = calibrate_(leftCam=True, outside=True)

    # calibrate left camera inside
    print('=== Calibrate left camera inside ===')
    calibrator = calibrate_(leftCam=True, outside=False)

    ideal_calibration_left = load_obj(name='charuco_outside_left')
    calibrator.adjustCalibration(K=ideal_calibration_left['K'],
                                 D=ideal_calibration_left['D'])

    print('Mono camera calibration done')

def Charuco_StereoCalibration():
    def stereo_calibrate(outside=True):
        if outside:
            images = '/home/eugeniu/Desktop/my_data/CameraCalibration/data/car_cam_data/charuco/outside/Stereo'
            name = 'outside'
        else:
            images = '/home/eugeniu/Desktop/my_data/CameraCalibration/data/car_cam_data/charuco/inside/Stereo'
            name = 'inside'

        calibrator = StereoCharuco_Calibrator(images,name = name)
        calibrator.createCalibrationBoard()
        calibrator.readMonoData()
        calibrator.read_images(test=False)
        #calibrator.calibrationReport()
        calibrator.stereoCalibrate(save=True)
        # calibrator.readStereoData()

    #Calibrate outside--------
    print('Stereo Calibrate outside')
    #stereo_calibrate(outside=True)

    #Calibrate inside---------
    print('Stereo Calibrate inside')
    #stereo_calibrate(outside=False)

    def testCalibration(outside = True):
        if outside:
            images = '/home/eugeniu/Desktop/my_data/CameraCalibration/data/car_cam_data/charuco/outside/Stereo'
            name = 'outside'
        else:
            images = '/home/eugeniu/Desktop/my_data/CameraCalibration/data/car_cam_data/charuco/inside/Stereo'
            name = 'inside'

        images = '/home/eugeniu/Desktop/Stereo_test'
        calibrator = StereoCharuco_Calibrator(images, name = name)
        calibrator.createCalibrationBoard()
        calibrator.readMonoData()
        calibrator.read_images(test=True)
        calibrator.readStereoData()
        calibrator.depth()

    #Test calibration outside-----
    print('Test calibration outside')
    testCalibration(outside=True)

    #Test calibration inside
    print('Test calibration inside')
    #testCalibration(outside=False)

def visualize3Dpoints():
    import open3d
    def visualize_model(file):
        for i, file_path in enumerate(file):
            print("{} Load a ply point cloud, print it, and render it".format(file_path))
            pcd = open3d.io.read_point_cloud(file_path)
            open3d.visualization.draw_geometries([pcd])

    points = glob.glob('/home/eugeniu/Desktop/my_data/*.ply')
    visualize_model(points)

def combinedChess_and_Charuco():
    calibrator = MonoCombinedCalibration()

    name = 'outside'
    calibrator = CombinedCalibration(name=name)
    calibrator.readMonoData()
    # read stereo images & stereo calibrate
    # imgChess = '/home/eugeniu/Desktop/my_data/CameraCalibration/data/car_cam_data/chess/{}/Stereo'.format(name)
    # imgCharuco = '/home/eugeniu/Desktop/my_data/CameraCalibration/data/car_cam_data/charuco/{}/Stereo'.format(name)
    # calibrator.calibrateStereo(imgChess, imgCharuco, see=True)

    calibrator.readStereoData()

    images = '/home/eugeniu/Desktop/Stereo_test'
    calibrator.depth(testImages=images)

if __name__ == '__main__':
    print('Main')

    #Mono calibration with chessboard----------------------------------
    #Chess_MonoCalibration()

    #Stereo calibration with chessboard--------------------------------
    #Chess_StereoCalibration()

    # Mono calibration with charuco----------------------------------
    #Charuco_MonoCalibration()

    # Stereo calibration with charuco--------------------------------
    #Charuco_StereoCalibration()

    #View pointcloud-------------------------------------------------
    #visualize3Dpoints()

    #combined chess & charuco images
    #combinedChess_and_Charuco()

