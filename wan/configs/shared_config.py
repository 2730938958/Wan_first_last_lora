# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
from easydict import EasyDict

#------------------------ Wan shared config ------------------------#
wan_shared_cfg = EasyDict()

# t5
wan_shared_cfg.t5_model = 'umt5_xxl'
wan_shared_cfg.t5_dtype = torch.bfloat16
wan_shared_cfg.text_len = 512

# transformer
wan_shared_cfg.param_dtype = torch.bfloat16

# inference
wan_shared_cfg.num_train_timesteps = 1000
wan_shared_cfg.sample_fps = 16
# wan_shared_cfg.sample_neg_prompt = '色调艳丽，过曝，画面放大，画面偏移，镜头偏移，转场，背景变色，细节模糊不清，整体发灰，字幕，风格，作品，画作，画面，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'
wan_shared_cfg.sample_neg_prompt = 'Vivid colors, overexposed, zoomed-in frame, frame shift, camera tilt, zoom in,frame goes blank, transitions, background color change, blurry details, overall grayish tone, subtitles, style, artwork, painting, frame, worst quality, low quality, JPEG compression artifacts, ugly, mutilated, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, misshapen limbs, fused fingers, static frame, cluttered background, three legs, crowded background, walking backward'
