"""Basic usage example for RAHL"""

from rahl import RAHLPipeline

# Initialize pipeline
pipeline = RAHLPipeline(device="cuda")  # or "cpu"

# Generate video
video = pipeline.generate(
    prompt="a beautiful sunset over the ocean, waves crashing on rocks, high quality, 4k",
    negative_prompt="blurry, low quality, distorted",
    num_frames=24,
    fps=12,
    height=512,
    width=512,
    seed=42
)

# Save video
pipeline.save_video(video, "sunset_video.mp4")

print("Video generation complete!")
