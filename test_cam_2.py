# %%

import os, cv2, time
import numpy as np
from cv2_enumerate_cameras import enumerate_cameras
 
os.environ["XDG_SESSION_TYPE"] = "xcb" # Fix for OpenCV GUI issues on Linux with Wayland (if using a USB camera, otherwise comment out if using a built-in
 
# camera, or provide a video file path)
""" Function to check connected USB cameras and their indexes """
def camera_index():
    for camera_info in enumerate_cameras(cv2.CAP_AVFOUNDATION):
        print(camera_info)
    devices = enumerate_cameras(cv2.CAP_AVFOUNDATION)
   
    for device in devices:
        print(f'{device.index}: {device.name}')
        # Search for specific device name, change depending on camera or USB device
        if device.name == "HD Pro Webcam C920" or device.name == "Logi Webcam C920e":
            print(f'Found {device.name} on index {device.index}')
            return device.index
        else:
            print('No suitable camera found.')
 
# Find camera index
CAMERA_INDEX = camera_index()
# 1. Open the webcam
cam = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)
 
# Attempt to disable built-in parameters
cam.set(cv2.CAP_PROP_AUTOFOCUS, 1)      # Auto-focus
#cam.set(cv2.CAP_PROP_FOCUS, 100)
# cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # Auto-exposure
# cam.set(cv2.CAP_PROP_CONVERT_RGB, 0)      # RGB conversion
cam.set(cv2.CAP_PROP_BRIGHTNESS, 128)     # Brightness
cam.set(cv2.CAP_PROP_CONTRAST, 128)       # Contrast
 
# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
# Optional: Set resolution to a lower value first to check performance
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
 
# Optional: Set up video writer to save raw frames
# fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec
# out = cv2.VideoWriter('/home/orion/Pictures/Diffraction patterns/direct_sensor.avi', fourcc, 20.0, (frame_width, frame_height))
 
# 2. Force the camera to output raw/uncompressed data if possible
# cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))

# Capture video for a certain duration (e.g., 60 seconds)
duration = 1800 # Capture video for a certain duration (e.g., 60 seconds)- 2 hours =7200s
start_time = time.time()
 
if not cam.isOpened():
    print("Error: Could not open camera.")
    exit()
 
# Process video frames
save_interval = 3  # seconds
last_save = start_time
save_dir = "/Users/grace/msc_project/training_frames/HF_pH5_Iso"

while True:
    current_time = time.time()

    # Stop after 2 minutes
    if current_time - start_time >= duration:
        break

    ret, frame = cam.read()
    if not ret:
        break

    cv2.imshow('Camera', frame)

    # Save one image every 10 seconds
    if current_time - last_save >= save_interval:
        elapsed = int(current_time - start_time)
        filename = os.path.join(save_dir, f"frame_{elapsed:03d}s.png")
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")

        last_save = current_time

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Check if the duration has been reached
    if current_time - start_time > duration:
        cam.release()
        cv2.destroyAllWindows()
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

 
    # Check if the duration has been reached
    if time.time() - start_time > duration:
        # Release all space and windows
        cam.release()
        # out.release()
        cv2.destroyAllWindows()
        break
 
    # Quit logic
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# %%