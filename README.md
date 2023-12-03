# # Animate Anyone App: A simple app to animate any character from a single image
import streamlit as st
import cv2
import numpy as np
from models import Generator, Discriminator, PoseEncoder
from utils import load_image, load_pose, save_video, draw_keypoints

# Load the model weights
gen = Generator().to(device)
dis = Discriminator().to(device)
pose_enc = PoseEncoder().to(device)
gen.load_state_dict(torch.load('gen.pth'))
dis.load_state_dict(torch.load('dis.pth'))
pose_enc.load_state_dict(torch.load('pose_enc.pth'))

# Create the app interface
st.title('Animate Anyone App')
st.write('This app allows you to animate any character from a single image using a sequence of poses.')
st.write('To use the app, please follow these steps:')
st.write('1. Upload an image of the character you want to animate.')
st.write('2. Select the number of poses you want to use.')
st.write('3. Click on the image to mark the keypoints of the character.')
st.write('4. Click on the "Animate" button to generate the video.')

# Upload the image
image_file = st.file_uploader('Upload an image of the character', type=['jpg', 'png'])
if image_file is not None:
  image = load_image(image_file)
  st.image(image, caption='Uploaded image', use_column_width=True)

# Select the number of poses
pose_num = st.slider('Select the number of poses', min_value=2, max_value=10, value=5)

# Mark the keypoints on the image
st.write('Click on the image to mark the keypoints of the character.')
st.write('You need to mark 18 keypoints in the following order:')
st.write('Head, Neck, Right Shoulder, Right Elbow, Right Wrist, Left Shoulder, Left Elbow, Left Wrist, Right Hip, Right Knee, Right Ankle, Left Hip, Left Knee, Left Ankle, Right Eye, Left Eye, Right Ear, Left Ear')
keypoints = st.session_state.get('keypoints', [])
if st.button('Clear'):
  keypoints.clear()
if len(keypoints) < 18 * pose_num:
  st.write(f'You have marked {len(keypoints)} keypoints out of {18 * pose_num}.')
  if st.image(image).clicked:
    keypoints.append(st.image(image).click_coordinates)
else:
  st.write(f'You have marked {len(keypoints)} keypoints. You can clear them and start over if you want.')

# Animate the character
if st.button('Animate'):
  st.write('Please wait while the app is generating the video.')
  # Encode the image and the pose sequence
  image_enc = gen.encode(image)
  pose_seq = np.array(keypoints).reshape(pose_num, 18, 2)
  pose_seq_enc = pose_enc(pose_seq)
  # Generate the video frames
  video_frames = []
  for pose_enc in pose_seq_enc:
    video_frame = gen.decode(image_enc, pose_enc)
    video_frames.append(video_frame)
  # Save the video
  save_video('video.mp4', video_frames)
  # Display the video
  st.video('video.mp4')
