import cv2
import numpy as np
import glob
import os

# Example parameters
checkerboard_size = (7, 7)  # Number of inner corners (rows, cols)
square_size = 0.0345  # Size of a square in meters 


def calculate_focal_length_from_checkerboard(checkerboard_size, square_size):

    # Prepare object points based on the checkerboard size and square size
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[1], 0:checkerboard_size[0]].T.reshape(-1, 2) * square_size

    # Arrays to store object points and image points from all images
    objpoints = []  # 3D point in real world space
    imgpoints = []  # 2D points in image plane

    # Define the path to the calibration images folder
    image_folder = os.path.join(os.getcwd(), 'calibration_images')
    images = glob.glob(os.path.join(image_folder, '*.png'))  # Read PNG images

    for image_path in images:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Unable to read image {image_path}. Skipping.")
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    # Check if we found any corners
    if len(imgpoints) == 0:
        print("Error: No valid checkerboard corners found in the images.")
        return None

    # Perform camera calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    if not ret:
        print("Error: Camera calibration failed.")
        return None

    # Extract focal lengths from the camera matrix
    fx = camera_matrix[0, 0]  # Focal length in pixels (x direction)
    fy = camera_matrix[1, 1]  # Focal length in pixels (y direction)

    return fx, fy


# Calculate focal length in pixels
focal_length_pixels = calculate_focal_length_from_checkerboard(checkerboard_size, square_size)

if focal_length_pixels is not None:
    print(f"Focal Length in Pixels: fx = {focal_length_pixels[0]}, fy = {focal_length_pixels[1]}")
else:
    print("Failed to calculate focal length.")