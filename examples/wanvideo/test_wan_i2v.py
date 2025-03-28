import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
import pdb
from PIL import Image
model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
model_manager.load_models(
    [
        [
            "/workspace/cpfs-data/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors",
            "/workspace/cpfs-data/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors",
            "/workspace/cpfs-data/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors",
            "/workspace/cpfs-data/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors",
            "/workspace/cpfs-data/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors",
            "/workspace/cpfs-data/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors",
            "/workspace/cpfs-data/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors",
        ],
        "/workspace/cpfs-data/Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        "/workspace/cpfs-data/Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth",
        "/workspace/cpfs-data/Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth",
    ]
)
# model_manager.load_lora("/workspace/nas-data/Wan2.1/examples/wanvideo/models/lightning_logs/version_16/checkpoints/epoch=29-step=150.ckpt", lora_alpha=1.0)
model_manager.load_lora("/workspace/nas-data/Wan2.1/examples/wanvideo/models/lightning_logs/version_21/checkpoints/epoch=104-step=315.ckpt", lora_alpha=1.0)
pipe = WanVideoPipeline.from_model_manager(model_manager, device="cuda")
pipe.enable_vram_management(num_persistent_param_in_dit=None)
prompt = "The girl rests her hands on her face and leans forward."
image = Image.open("yuan0.png")
last_frame = Image.open("yuan1.png")
width, height = image.size
ratio = int(min(1280 / height, 720 /width))
video = pipe(
    prompt=prompt,
    negative_prompt="",
    input_image=image,
    num_inference_steps=50,
    seed=0, tiled=True, last_frame=last_frame,
    height = height*ratio, width = width*ratio
)
save_video(video, "video.mp4", fps=25, quality=9)