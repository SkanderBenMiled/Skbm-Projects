import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from datetime import datetime
import time

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize session state
if 'webcam_running' not in st.session_state:
    st.session_state.webcam_running = False
if 'snapshot_taken' not in st.session_state:
    st.session_state.snapshot_taken = False

# Function to detect faces in a single frame
def detect_faces_in_frame(frame, scale_factor, min_neighbors, rect_ r_bgr):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), rect_color_bgr, 2)
    
    return frame, len(faces)

# Function to capture and process webcam feed
def run_webcam_detection(scale_factor, min_neighbors, rect_color_bgr):
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Could not open webcam. Please check if your camera is connected and not being used by another application.")
        return
    
    # Create placeholders for the video feed and info
    video_placeholder = st.empty()
    info_placeholder = st.empty()
    
    # Control buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Take Snapshot", key="snapshot_btn"):
            st.session_state.snapshot_taken = True
    
    with col2:
        if st.button("Stop Webcam", key="stop_btn"):
            st.session_state.webcam_running = False
    
    with col3:
        st.write(f"Webcam Status: {'Running' if st.session_state.webcam_running else 'Stopped'}")
    
    frame_count = 0
    max_frames = 300  # Limit to prevent infinite running
    
    while st.session_state.webcam_running and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image from webcam.")
            break
        
        # Process frame
        processed_frame, face_count = detect_faces_in_frame(frame, scale_factor, min_neighbors, rect_color_bgr)
        
        # Display frame
        video_placeholder.image(processed_frame, channels="BGR", use_container_width=True)
        info_placeholder.write(f"Faces detected: {face_count} | Frame: {frame_count}")
        
        # Handle snapshot
        if st.session_state.snapshot_taken:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_filename = f"snapshot_{timestamp}.jpg"
            cv2.imwrite(snapshot_filename, processed_frame)
            st.success(f"Snapshot saved as {snapshot_filename}")
            
            # Display snapshot
            st.image(snapshot_filename, caption="Snapshot Taken", use_container_width=True)
            
            # Reset snapshot flag
            st.session_state.snapshot_taken = False
        
        frame_count += 1
        time.sleep(0.1)  # Small delay to prevent overwhelming the browser
        
        # Force a rerun to check button states
        if frame_count % 50 == 0:  # Check every 50 frames
            st.rerun()
    
    cap.release()
    if frame_count >= max_frames:
        st.warning("Webcam stopped after reaching maximum frame limit.")

# Streamlit app layout
st.title("Facial Recognition App")
st.write("This app detects faces in real-time using your webcam.")

# Sidebar settings
st.sidebar.header("Settings")
scale_factor = st.sidebar.slider("Scale Factor", 1.1, 2.0, 1.1, 0.1)
min_neighbors = st.sidebar.slider("Min Neighbors", 1, 10, 5, 1)
rect_color = st.sidebar.color_picker("Rectangle Color", "#00FF00")

# Convert hex color to BGR format
rect_color_bgr = tuple(int(rect_color[i:i + 2], 16) for i in (5, 3, 1))

# Main control buttons
st.write("### Camera Controls")

col1, col2 = st.columns(2)

with col1:
    if st.button("Start Face Detection", key="start_btn"):
        st.session_state.webcam_running = True
        st.rerun()

with col2:
    if st.button("Reset App", key="reset_btn"):
        st.session_state.webcam_running = False
        st.session_state.snapshot_taken = False
        st.rerun()

# Run webcam if started
if st.session_state.webcam_running:
    st.write("### Live Camera Feed")
    run_webcam_detection(scale_factor, min_neighbors, rect_color_bgr)
else:
    st.write("Click 'Start Face Detection' to begin.")
    st.info("**Note:** Make sure your webcam is connected and not being used by another application.")

# Additional information
st.write("### Instructions:")
st.write("1. Adjust the settings in the sidebar for better face detection")
st.write("2. Click 'Start Face Detection' to begin the camera feed")
st.write("3. Use 'Take Snapshot' to capture the current frame")
st.write("4. Use 'Stop Webcam' to end the session")
st.write("5. Use 'Reset App' to restart everything")