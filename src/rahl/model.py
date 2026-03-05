import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np

class TemporalModule(nn.Module):
    """Temporal processing module for video consistency"""
    def __init__(self, dim, num_frames=16):
        super().__init__()
        self.num_frames = num_frames
        self.temporal_conv = nn.Conv3d(dim, dim, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.temporal_attn = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        
    def forward(self, x):
        # x shape: (batch, frames, channels, height, width)
        b, f, c, h, w = x.shape
        
        # Temporal convolution
        x_conv = x.permute(0, 2, 1, 3, 4)  # (b, c, f, h, w)
        x_conv = self.temporal_conv(x_conv)
        x_conv = x_conv.permute(0, 2, 1, 3, 4)  # (b, f, c, h, w)
        
        # Temporal attention
        x_flat = x_conv.reshape(b, f, c * h * w)
        x_attn, _ = self.temporal_attn(x_flat, x_flat, x_flat)
        x_attn = x_attn.reshape(b, f, c, h, w)
        
        return x_attn + x

class RAHLModel(nn.Module):
    """Main RAHL model for text-to-video generation"""
    
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", num_frames=16):
        super().__init__()
        
        # Load pretrained components
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        
        # Freeze pretrained parts
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        
        # Add temporal modules
        self.temporal_modules = nn.ModuleList([
            TemporalModule(320, num_frames),  # First UNet block
            TemporalModule(640, num_frames),  # Second UNet block
            TemporalModule(1280, num_frames), # Third UNet block
            TemporalModule(1280, num_frames)  # Fourth UNet block
        ])
        
        self.num_frames = num_frames
        
    def forward(self, prompt, negative_prompt="", num_inference_steps=50, guidance_scale=7.5):
        """Generate video from text prompt"""
        
        # Encode text
        text_input = self.tokenizer(prompt, padding="max_length", max_length=77, 
                                    truncation=True, return_tensors="pt")
        text_embeddings = self.text_encoder(text_input.input_ids)[0]
        
        # Encode negative prompt
        uncond_input = self.tokenizer(negative_prompt, padding="max_length", 
                                     max_length=77, return_tensors="pt")
        uncond_embeddings = self.text_encoder(uncond_input.input_ids)[0]
        
        # Concatenate for classifier-free guidance
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # Generate initial noise
        batch_size = 1
        height = self.unet.config.sample_size * self.vae_scale_factor
        width = height
        
        latents = torch.randn((batch_size, self.num_frames, 4, height//8, width//8))
        
        # Sampling loop
        for t in range(num_inference_steps):
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            
            # Apply temporal modules
            for i, temp_mod in enumerate(self.temporal_modules):
                latent_model_input = temp_mod(latent_model_input)
            
            # Predict noise
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)
            
            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Update latents
            latents = self.scheduler_step(noise_pred, t, latents)
        
        # Decode latents to video frames
        video = self.decode_latents(latents)
        
        return video
    
    def decode_latents(self, latents):
        """Decode latents to video frames"""
        video = []
        for i in range(latents.shape[1]):
            frame = self.vae.decode(latents[:, i] / 0.18215).sample
            frame = (frame / 2 + 0.5).clamp(0, 1)
            frame = frame.cpu().permute(0, 2, 3, 1).numpy()
            video.append(frame)
        
        return np.stack(video).squeeze()
    
    def scheduler_step(self, noise_pred, t, latents):
        """Single scheduler step (simplified DDIM)"""
        alpha_t = self.scheduler.alphas_cumprod[t]
        alpha_prev = self.scheduler.alphas_cumprod[t-1] if t > 0 else self.scheduler.final_alpha_cumprod
        
        pred_x0 = (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        
        dir_xt = (1 - alpha_prev).sqrt() * noise_pred
        
        x_prev = alpha_prev.sqrt() * pred_x0 + dir_xt
        
        return x_prev
