import cv2
from matplotlib import pyplot as plt
import numpy as np
from stereovision.calibration import StereoCalibration

# Global variables preset
left_image_path = 'Left_photo_2.png'  # Path to the left camera image
right_image_path = 'Right_photo_2.png'  # Path to the right camera image
photo_Width = 640
photo_Height = 480

# Load left and right images
print('Reading left and right images...')
imgLeft = cv2.imread(left_image_path, 0)  # Load as grayscale
imgRight = cv2.imread(right_image_path, 0)  # Load as grayscale

if imgLeft is None or imgRight is None:
    raise ValueError("Could not read one or both images. Please check the file paths.")

# Implementing calibration data
print('Load calibration data...')
calibration = StereoCalibration(input_folder='ress')
rectified_pair = calibration.rectify((imgLeft, imgRight))

# Depth map function
print('Building depth map...')
def stereo_depth_map(rectified_pair, ndisp, sws):
    # Create StereoBM object
    stereo = cv2.StereoBM_create(numDisparities=ndisp, blockSize=sws)
    return stereo.compute(rectified_pair[0], rectified_pair[1])

disparity = stereo_depth_map(rectified_pair, 80, 7)
print('Done! Let\'s look at the depth map')

# Normalization and plotting
def plot(title, img, i):
    plt.subplot(2, 2, i)
    plt.title(title)
    plt.imshow(img, 'gray')
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)

plot('Left Calibrated', rectified_pair[0], 1)
plot('Right Calibrated', rectified_pair[1], 2)
plot('Depth Map', disparity / 255., 3)
plt.show()