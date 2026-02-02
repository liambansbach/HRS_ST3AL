#!/usr/bin/env python3
"""
TUM - ICS AiNex CameraSubscriberCompressed Demo for ROS 2 Jazzy
----------------------------------------
Subscribes to JPEG-compressed images and raw images on /camera_image/compressed and /camera_image,
shows frames with OpenCV, and displays CameraInfo.

Requires:
  sudo apt install python3-numpy python3-opencv

Msgs:
    sensor_msgs/CompressedImage
    sensor_msgs/CameraInfo
"""

import os
from typing import Tuple
import numpy as np
import yaml
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, CameraInfo
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import Point, Vector3
from pathlib import Path
import glob


class CameraCalibration(Node):
    def __init__(self):
        super().__init__('calibrate')
        self.cwd = Path.cwd()
        self.track_window = None

        self.camera_k = None
        self.camera_d = None
        self.camera_width = None
        self.camera_heigth = None

        # Kalibrierungs-Parameter
        self.mtx = None
        self.dist = None
        self.newcameramtx = None
        self.mapx = None
        self.mapy = None
        self.roi = None
        self.roi_x = None
        self.roi_y = None
        self.roi_width = None
        self.roi_heigth = None

        self.frame_undistorted = None

        self.path_to_images = str(Path.cwd()) + "/imgs/calibration_images_ainex-04"
        self.cb_group = ReentrantCallbackGroup()

        # QoS: Reliable to ensure camera_info is received
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Subscribe compressed images
        self.sub_compressed = self.create_subscription(
            CompressedImage,
            'camera_image/compressed',
            self.image_callback_compressed,
            sensor_qos,
            callback_group=self.cb_group,
        )

        # Subscribe camera info
        self.sub_camerainfo = self.create_subscription(
            CameraInfo,
            'camera_info',
            self.camera_info_callback,
            sensor_qos,
            callback_group=self.cb_group
        )

        # State variables
        self.camera_info_received = False
        self.frame = None  # BGR Frame

    def camera_info_callback(self, msg: CameraInfo):
        if not self.camera_info_received:
            self.get_logger().info(
                f'Camera Info received: {msg.width}x{msg.height}\n'
                f'K: {msg.k}\n'
                f'D: {msg.d}'
            )
            print(f'Camera Info received: {msg.width}x{msg.height}')
            print(f'Intrinsic matrix K: {msg.k}')
            print(f'Distortion coeffs D: {msg.d}')
            self.camera_info_received = True

            self.camera_k = msg.k
            self.camera_d = msg.d
            self.camera_width = msg.width
            self.camera_heigth = msg.height

    def image_callback_compressed(self, msg: CompressedImage):
        try:
            # Decode the compressed image
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                self.get_logger().warn('JPEG decode returned None')
                return

            self.frame = frame

        except Exception as exc:
            self.get_logger().error(f'Decode error in compressed image: {exc}')

    def process_key(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False  # Quit
        if key == ord('c'):
            self.show_compressed = True
            self.get_logger().info('Switched to compressed image')
        return True

    def display_loop(self):
        while rclpy.ok():
            if self.frame is not None:
                self.frame_undistorted = self.undistort_image(self.frame)
                if self.frame_undistorted is not None:
                    cv2.imshow('undistorted image', self.frame_undistorted)

                cv2.imshow('Camera Subscrber', self.frame)

            if not self.process_key():
                break

            rclpy.spin_once(self, timeout_sec=0.01)

        cv2.destroyAllWindows()

    @staticmethod
    def compute_keff_deff_from_calib_dict(calib: dict):
        """
        Computes K_eff and D_eff for the *final* image that your pipeline produces:
        undistort -> crop ROI -> resize back to (image_width, image_height).

        D_eff is set to zeros because the final image is already undistorted.
        """
        # original image size (before undistort, and after your final resize)
        W = int(calib["image_width"])
        H = int(calib["image_height"])

        Knew = np.array(calib["new_camera_matrix"], dtype=float)

        roi = calib["roi"]
        x = float(roi["x"])
        y = float(roi["y"])
        w_roi = float(roi["width"])
        h_roi = float(roi["height"])

        sx = W / w_roi
        sy = H / h_roi

        fx = Knew[0, 0]
        fy = Knew[1, 1]
        cx = Knew[0, 2]
        cy = Knew[1, 2]

        # after crop
        cx_c = cx - x
        cy_c = cy - y

        # after resize back to WxH
        K_eff = np.array([
            [fx * sx, 0.0,      cx_c * sx],
            [0.0,     fy * sy,  cy_c * sy],
            [0.0,     0.0,      1.0]
        ], dtype=float)

        # final image is undistorted already
        D_eff = np.zeros((5,), dtype=float)

        return K_eff, D_eff

    def calibrate(self):
        self.get_logger().info(f"Starting calibration using images in: {self.path_to_images}")

        # termination criteria
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001
        )

        # Size of Checkerboard
        chessboard_cols = 6  # x-Richtung
        chessboard_rows = 8  # y-Richtung

        # points in world coordinate system (z=0)
        objp = np.zeros((chessboard_rows * chessboard_cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_cols, 0:chessboard_rows].T.reshape(-1, 2)

        square_size = 24.0  # mm
        objp *= square_size

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        image_paths = glob.glob(os.path.join(self.path_to_images, "*.png"))
        if not image_paths:
            self.get_logger().error("No calibration images found!")
            return

        gray = None

        for file_path in image_paths:
            image = cv2.imread(file_path)
            if image is None:
                self.get_logger().warn(f"Could not read image: {file_path}")
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (chessboard_cols, chessboard_rows), None)

            if ret is True:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

                cv2.drawChessboardCorners(image, (chessboard_cols, chessboard_rows), corners2, ret)

        if len(objpoints) < 1:
            self.get_logger().error("Not enough valid calibration images.")
            return

        # calibrate using all images.
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        h, w = gray.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), cv2.CV_32FC1)

        # crop ROI
        x, y, w_roi, h_roi = roi

        # Save parameters for publisher
        os.makedirs("camera_parameters", exist_ok=True)

        np.save("camera_parameters/old_camera_matrix.npy", mtx)
        np.save("camera_parameters/new_camera_matrix.npy", newcameramtx)
        np.save("camera_parameters/distortion_coeff.npy", dist)
        np.save("camera_parameters/mapping_x.npy", mapx)
        np.save("camera_parameters/mapping_y.npy", mapy)
        np.save("camera_parameters/x.npy", x)
        np.save("camera_parameters/y.npy", y)
        np.save("camera_parameters/w_roi.npy", w_roi)
        np.save("camera_parameters/h_roi.npy", h_roi)
        np.save("camera_parameters/roi.npy", np.array(roi))

        # save calibration parameters in object
        self.mtx = mtx
        self.dist = dist
        self.newcameramtx = newcameramtx
        self.mapx = mapx
        self.mapy = mapy
        self.roi = roi
        self.roi_x = x
        self.roi_y = y
        self.roi_width = w_roi
        self.roi_heigth = h_roi

        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error

        mean_error /= len(objpoints)
        self.get_logger().info(f"Calibration done. Mean reprojection error: {mean_error}")

        # Save as YAML file:
        calib_dict = {
            "image_width": int(w),
            "image_height": int(h),
            "camera_matrix": mtx.tolist(),
            "distortion_coefficients": dist.tolist(),
            "new_camera_matrix": newcameramtx.tolist(),
            "roi": {
                "x": int(roi[0]),
                "y": int(roi[1]),
                "width": int(roi[2]),
                "height": int(roi[3]),
            },
            "mean_reprojection_error": float(mean_error),
        }

        # --- NEW: compute and save K_eff / D_eff into the YAML ---
        K_eff, D_eff = self.compute_keff_deff_from_calib_dict(calib_dict)
        calib_dict["K_eff"] = K_eff.tolist()
        calib_dict["D_eff"] = D_eff.tolist()
        # --------------------------------------------------------

        with open("camera_parameters/calibration.yaml", "w") as f:
            yaml.safe_dump(calib_dict, f, sort_keys=False)

        self.get_logger().info("Calibration parameters saved to camera_parameters/")

    def undistort_image(self, frame):
        try:
            if self.mapx is not None and self.mapy is not None:
                # remap mit gespeicherten Maps
                undistorted = cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR)

                # crop with ROI
                x, y, w_roi, h_roi = self.roi
                undistorted = undistorted[y:y + h_roi, x:x + w_roi]

                # Resize the cropped image to the original frame size
                frame_heigth, frame_width, _ = frame.shape
                undistorted = cv2.resize(undistorted, (frame_width, frame_heigth), interpolation=cv2.INTER_CUBIC)

                self.frame_undistorted = undistorted
                return self.frame_undistorted

        except Exception as exc:
            self.get_logger().error(f'Something with the undistortion went wrong!: {exc}')
            return None

        return None


def main(args=None):
    rclpy.init(args=args)
    node = CameraCalibration()
    node.get_logger().info('CameraCalibration node started')
    node.calibrate()  # calibrate Cameras

    try:
        node.display_loop()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
