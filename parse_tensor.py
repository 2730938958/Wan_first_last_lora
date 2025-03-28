import os
import torch

# 设置目录路径
directory = "/workspace/cpfs-data/all/train"  # 替换为你的目录路径

# 获取目录下所有以 .tensor.pth 结尾的文件
files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.tensors.pth')]

# 遍历文件，加载、修改并覆盖 Tensor
for file in files:
    try:
        # 加载 Tensor
        tensor = torch.load(file)
        if tensor['latents'].shape[3] != 134:
            new_latent = tensor['latents'][:, :, :, 1:]
            tensor['latents'] = new_latent
        # 修改 Tensor
        # 示例：将所有元素加 1（你可以根据需要修改这部分逻辑）
        if tensor['image_emb']['y'].shape[4] != 134:
            new_image = tensor['image_emb']['y'][:, :, :, :, 1:]
            tensor['image_emb']['y'] = new_image

        # 覆盖当前文件
        torch.save(tensor, file)


    except Exception as e:
        print(f"Failed to process {file}: {e}")