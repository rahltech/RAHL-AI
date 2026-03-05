"""Utility functions for RAHL"""

import cv2
import numpy as np
from PIL import Image
import os
from typing import List, Union
import torch

def save_video(frames: np.ndarray, output_path: str, fps: int = 8, codec: str = 'mp4v'):
    """
    Save numpy array frames to video file
    
    Args:
        frames: numpy array of shape (num_frames, height, width, 3)
        output_path: path to save video
        fps: frames per second
        codec: video codec (mp4v, avc1, etc)
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Get dimensions
    height, width = frames[0].shape[:2]
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        if frame.shape[-1] == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame
        out.write(frame_bgr.astype(np.uint8))
    
    out.release()
    print(f"✅ Video saved to {output_path}")

def load_video_frames(video_path: str, max_frames: int = None) -> np.ndarray:
    """
    Load video frames as numpy array
    
    Args:
        video_path: path to video file
        max_frames: maximum number of frames to load
    
    Returns:
        numpy array of frames
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
        if max_frames and len(frames) >= max_frames:
            break
    
    cap.release()
    return np.array(frames)

def resize_frames(frames: np.ndarray, target_size: tuple) -> np.ndarray:
    """Resize all frames to target size (width, height)"""
    resized = []
    for frame in frames:
        resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LANCZOS4)
        resized.append(resized_frame)
    return np.array(resized)

def frames_to_gif(frames: np.ndarray, output_path: str, fps: int = 8):
    """Save frames as GIF"""
    from PIL import Image
    
    pil_frames = [Image.fromarray(frame) for frame in frames]
    duration = int(1000 / fps)  # Convert to milliseconds
    
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0
    )
    print(f"✅ GIF saved to {output_path}")

def extract_first_frame(video_path: str, output_path: str):
    """Extract first frame from video as image"""
    frames = load_video_frames(video_path, max_frames=1)
    if len(frames) > 0:
        Image.fromarray(frames[0]).save(output_path)
        print(f"✅ First frame saved to {output_path}")
