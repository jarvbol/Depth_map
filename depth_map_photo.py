import cv2
import numpy as np
import json
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
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
print('Read calibration data and rectifying stereo pair...')
calibration = StereoCalibration(input_folder='ress')
rectified_pair = calibration.rectify((imgLeft, imgRight))

# Default parameters for depth map
SWS = 5
PFS = 5
PFC = 29
MDS = -25
NOD = 128
TTH = 100
UR = 10
SR = 15
SPWS = 100

def stereo_depth_map(rectified_pair):
    print('SWS=' + str(SWS) + ' PFS=' + str(PFS) + ' PFC=' + str(PFC) + ' MDS=' + \
          str(MDS) + ' NOD=' + str(NOD) + ' TTH=' + str(TTH))
    print(' UR=' + str(UR) + ' SR=' + str(SR) + ' SPWS=' + str(SPWS))
    
    # Create StereoBM object
    stereo = cv2.StereoBM_create(numDisparities=NOD, blockSize=SWS)
    disparity = stereo.compute(rectified_pair[0], rectified_pair[1])
    
    # Normalize the disparity map for visualization
    disparity_visual = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    return disparity_visual.astype(np.uint8)

disparity = stereo_depth_map(rectified_pair)

# Set up and draw interface
axcolor = 'lightgoldenrodyellow'
fig = plt.subplots(1, 2)
plt.subplots_adjust(left=0.15, bottom=0.5)
plt.subplot(1, 2, 1)
plt.imshow(rectified_pair[0], 'gray')

# Draw buttons for saving and loading settings
saveax = plt.axes([0.3, 0.38, 0.15, 0.04])
buttons = Button(saveax, 'Save settings', color=axcolor, hovercolor='0.975')

def save_map_settings(event):
    buttons.label.set_text("Saving...")
    print('Saving to file...')
    result = {
        'SADWindowSize': SWS,
        'preFilterSize': PFS,
        'preFilterCap': PFC,
        'minDisparity': MDS,
        'numberOfDisparities': NOD,
        'textureThreshold': TTH,
        'uniquenessRatio': UR,
        'speckleRange': SR,
        'speckleWindowSize': SPWS
    }
    fName = '3dmap_set.txt'
    with open(fName, 'w') as f:
        json.dump(result, f, indent=4)
    buttons.label.set_text("Save to file")
    print('Settings saved to file ' + fName)

buttons.on_clicked(save_map_settings)

loadax = plt.axes([0.5, 0.38, 0.15, 0.04])
buttonl = Button(loadax, 'Load settings', color=axcolor, hovercolor='0.975')

def load_map_settings(event):
    global SWS, PFS, PFC, MDS, NOD, TTH, UR, SR, SPWS
    fName = '3dmap_set.txt'
    print('Loading parameters from file...')
    buttonl.label.set_text("Loading...")
    with open(fName, 'r') as f:
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
    buttonl.label.set_text("Load settings")
    print('Parameters loaded from file ' + fName)
    print('Redrawing depth map with loaded parameters...')
    update(0)
    print('Done!')

buttonl.on_clicked(load_map_settings)

plt.subplot(1, 2, 2)
dmObject = plt.imshow(disparity, aspect='equal')

# Draw interface for adjusting parameters
print('Start interface creation (it takes up to 30 seconds)...')

SWSaxe = plt.axes([0.15, 0.01, 0.7, 0.025])
PFSaxe = plt.axes([0.15, 0.05, 0.7, 0.025])
PFCaxe = plt.axes([0.15, 0.09, 0.7, 0.025])
MDSaxe = plt.axes([0.15, 0.13, 0.7, 0.025])
NODaxe = plt.axes([0.15, 0.17, 0.7, 0.025])
TTHaxe = plt.axes([0.15, 0.21, 0.7, 0.025])
URaxe = plt.axes([0.15, 0.25, 0.7, 0.025])
SRaxe = plt.axes([0.15, 0.29, 0.7, 0.025])
SPWSaxe = plt.axes([0.15, 0.33, 0.7, 0.025])

sSWS = Slider(SWSaxe, 'SWS', 5.0, 255.0, valinit=5)
sPFS = Slider(PFSaxe, 'PFS', 5.0, 255.0, valinit=5)
sPFC = Slider(PFCaxe, 'PreFiltCap', 5.0, 63.0, valinit=29)
sMDS = Slider(MDSaxe, 'MinDISP', -100.0, 100.0, valinit=-25)
sNOD = Slider(NODaxe, 'NumOfDisp', 16.0, 256.0, valinit=128)
sTTH = Slider(TTHaxe, 'TxtrThrshld', 0.0, 1000.0, valinit=100)
sUR = Slider(URaxe, 'UnicRatio', 1.0, 20.0, valinit=10)
sSR = Slider(SRaxe, 'SpcklRng', 0.0, 40.0, valinit=15)
sSPWS = Slider(SPWSaxe, 'SpklWinSze', 0.0, 300.0, valinit=100)

# Update depth map parameters and redraw
def update(val):
    global SWS, PFS, PFC, MDS, NOD, TTH, UR, SR, SPWS
    SWS = int(sSWS.val/2)*2+1 # Convert to ODD
    PFS = int(sPFS.val/2)*2+1
    PFC = int(sPFC.val/2)*2+1    
    MDS = int(sMDS.val)    
    NOD = int(sNOD.val/16)*16  
    TTH = int(sTTH.val)
    UR = int(sUR.val)
    SR = int(sSR.val)
    SPWS = int(sSPWS.val)
    
    print('Rebuilding depth map')
    disparity = stereo_depth_map(rectified_pair)
    dmObject.set_data(disparity)
    plt.draw()

# Connect update actions to control elements
sSWS.on_changed(update)
sPFS.on_changed(update)
sPFC.on_changed(update)
sMDS.on_changed(update)
sNOD.on_changed(update)
sTTH.on_changed(update)
sUR.on_changed(update)
sSR.on_changed(update)

print('Show interface to user')
plt.show()