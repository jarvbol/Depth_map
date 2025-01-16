import os
import cv2
import numpy as np
import glob
from stereovision.calibration import StereoCalibrator
from stereovision.calibration import StereoCalibration

# Global variables preset
total_photos = 3  # Adjust if you have more or fewer images
photo_Width = 640
photo_Height = 480

# Chessboard parameters
rows = 7  # Number of inner corners in rows
columns = 7  # Number of inner corners in columns
square_size = 34.5  # Size of a square in millimeters

# Image size
image_size = (photo_Width, photo_Height)

# Initialize the stereo calibrator
calibrator = StereoCalibrator(rows, columns, square_size, image_size)

# Load images from the specified directory
left_images = sorted(glob.glob("calibration_images/left_*.png"))
right_images = sorted(glob.glob("calibration_images/right_*.png"))

# Ensure the number of left and right images matches
if len(left_images) != len(right_images):
    raise ValueError("The number of left and right images must match.")

print('Start cycle')

# Iterate through the pairs of images
for photo_counter in range(len(left_images)):
    print('Import pair No ' + str(photo_counter + 1))
    
    leftName = left_images[photo_counter]
    rightName = right_images[photo_counter]  
    
    imgLeft = cv2.imread(leftName, 1)
    imgRight = cv2.imread(rightName, 1)
    
    if imgLeft is not None and imgRight is not None:
        try:
            calibrator.add_corners((imgLeft, imgRight), True)
        except Exception as e:
            print(f"Error processing pair {photo_counter + 1}: {e}")
    else:
        print(f"Warning: Could not read images {leftName} or {rightName}")

print('End cycle')

print('Starting calibration... It can take several minutes!')
calibration = calibrator.calibrate_cameras()
calibration.export('ress')
print('Calibration complete!')

# Rectify and show the last pair after calibration
calibration = StereoCalibration(input_folder='ress')
rectified_pair = calibration.rectify((imgLeft, imgRight))

cv2.imshow('Left CALIBRATED', rectified_pair[0])
cv2.imshow('Right CALIBRATED', rectified_pair[1])
cv2.imwrite("rectified_left.jpg", rectified_pair[0])
cv2.imwrite("rectified_right.jpg", rectified_pair[1])
cv2.waitKey(0)
cv2.destroyAllWindows()  # Close all OpenCV windows after key press