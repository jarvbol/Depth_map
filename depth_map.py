import cv2
import numpy as np
import json
from stereovision.calibration import StereoCalibration


# Initialize the cameras
left_camera = cv2.VideoCapture(2)  # Change the index if necessary
right_camera = cv2.VideoCapture(0)  # Change the index if necessary

# Load parameters from the JSON file
def load_map_settings(filename):
    global SWS, PFS, PFC, MDS, NOD, TTH, UR, SR, SPWS
    print('Loading parameters from file...')
    with open(filename, 'r') as f:
        data = json.load(f)
        SWS = data['SADWindowSize']
        PFS = data['preFilterSize']
        PFC = data['preFilterCap']
        MDS = data['minDisparity']
        NOD = data['numberOfDisparities']
        TTH = data['textureThreshold']
        UR = data['uniquenessRatio']
        SR = data['speckleRange']
        SPWS = data['speckleWindowSize']
    print('Parameters loaded from file:', filename)

# Load depth map parameters from the file
load_map_settings("3dmap_set.txt")  # Ensure this file exists in the working directory

# Preset parameters
photo_Width = 640
photo_Height = 480

# Create windows for displaying images
cv2.namedWindow("Left Image")
cv2.namedWindow("Right Image")
cv2.namedWindow("Disparity")

# Implementing calibration data
print('Read calibration data and rectifying stereo pair...')
calibration = StereoCalibration(input_folder='ress')  # Adjust the input folder as needed


if not left_camera.isOpened() or not right_camera.isOpened():
    print("Error: Could not open one or both cameras.")
    exit()

def stereo_depth_map(rectified_pair):
    c, r = rectified_pair[0].shape
    disparity = np.zeros((c, r), dtype=np.float32)
    
    sbm = cv2.StereoBM_create(numDisparities=NOD, blockSize=SWS)
    sbm.setPreFilterType(1)
    sbm.setPreFilterSize(PFS)
    sbm.setPreFilterCap(PFC)
    sbm.setMinDisparity(MDS)
    sbm.setTextureThreshold(TTH)
    sbm.setUniquenessRatio(UR)
    sbm.setSpeckleRange(SR)
    sbm.setSpeckleWindowSize(SPWS)

    disparity = sbm.compute(rectified_pair[0], rectified_pair[1])
    disparity_visual = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disparity_color = cv2.applyColorMap(np.uint8(disparity_visual), cv2.COLORMAP_JET)

    return disparity_color

# Main loop for capturing and processing images

while True:
    # Capture frames from both cameras
    ret_left, imgLeft = left_camera.read()
    ret_right, imgRight = right_camera.read()

    if not ret_left or not ret_right:
        print("Error: Could not read from one or both cameras.")
        break

    # Convert to grayscale
    imgLeft_gray = cv2.cvtColor(imgLeft, cv2.COLOR_BGR2GRAY)
    imgRight_gray = cv2.cvtColor(imgRight, cv2.COLOR_BGR2GRAY)

    # Rectify the stereo pair
    rectified_pair = calibration.rectify((imgLeft_gray, imgRight_gray))
    
    # Compute the disparity map
    disparity = stereo_depth_map(rectified_pair)

    # Display the images and disparity map
    cv2.imshow("Left Image", imgLeft)
    cv2.imshow("Right Image", imgRight)
    cv2.imshow("Disparity", disparity)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the cameras and close windows
left_camera.release()
right_camera.release()
cv2.destroyAllWindows()