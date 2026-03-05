"""Advanced generation examples for RAHL"""

from rahl import RAHLPipeline
from rahl.utils import frames_to_gif, resize_frames
import time

def main():
    print("🚀 RAHL Advanced Generation Examples")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = RAHLPipeline()
    
    # Example 1: Cinematic scene with camera movement
    print("\n🎬 Example 1: Cinematic scene")
    start = time.time()
    
    video1 = pipeline.generate(
        prompt="cinematic shot of a lone wolf standing on a mountain peak at sunrise, dramatic lighting, 8k",
        negative_prompt="blurry, low quality, cartoon, anime",
        num_frames=48,
        fps=24,
        height=768,
        width=768,
        guidance_scale=8.0,
        seed=42
    )
    
    pipeline.save_video(video1, "cinematic_wolf.mp4")
    print(f"✅ Generated in {time.time() - start:.2f}s")
    
    # Example 2: Fantasy scene
    print("\n🧙 Example 2: Fantasy scene")
    start = time.time()
    
    video2 = pipeline.generate(
        prompt="magical forest with glowing mushrooms, fairies flying, ethereal atmosphere, fantasy art",
        negative_prompt="dark, scary, modern, realistic",
        num_frames=32,
        fps=16,
        seed=123
    )
    
    # Save as both MP4 and GIF
    pipeline.save_video(video2, "fantasy_forest.mp4")
    frames_to_gif(video2['frames'], "fantasy_forest.gif", fps=8)
    print(f"✅ Generated in {time.time() - start:.2f}s")
    
    # Example 3: Action sequence
    print("\n💥 Example 3: Action sequence")
    start = time.time()
    
    video3 = pipeline.generate(
        prompt="explosion in space, debris flying, intense colors, sci-fi, epic",
        negative_prompt="slow motion, calm, boring",
        num_frames=24,
        fps=12,
        guidance_scale=8.5,
        seed=999
    )
    
    pipeline.save_video(video3, "space_explosion.mp4")
    print(f"✅ Generated in {time.time() - start:.2f}s")
    
    # Example 4: Style transfer
    print("\n🎨 Example 4: Artistic style")
    start = time.time()
    
    video4 = pipeline.generate(
        prompt="a beautiful garden in the style of Van Gogh, impressionist painting, vibrant colors",
        negative_prompt="photorealistic, 3d render",
        num_frames=16,
        fps=8,
        seed=777
    )
    
    pipeline.save_video(video4, "vangogh_garden.mp4")
    print(f"✅ Generated in {time.time() - start:.2f}s")
    
    print("\n" + "=" * 50)
    print("✅ All advanced examples completed!")

if __name__ == "__main__":
    main()
