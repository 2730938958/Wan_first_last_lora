'''
 # @ Author: Lu Liu
 # @ Create Time: 2024-12-05 14:28:43
 # @ Modified by: Lu Liu
 # @ Modified time: 2024-12-05 14:36:20
 # @ Description: 从huggingface下载
 #  - 整个repo
 #  - 单个文件
 #  - 指定版本

 # huya海聪平台外网加速器 https://ai.huya.com/docs/QA/common.html#%E9%80%9A%E7%94%A8%E5%A4%96%E7%BD%91%E4%B8%8B%E8%BD%BD%E5%8A%A0%E9%80%9F%E5%99%A8
 # 使用时命令前加上 `hai run`， 如 hai run git clone xx

 pip3 install hai --no-cache-dir -U -i https://pypi.huya.info/simple/

 pip install huggingface_hub[hf_transfer]
 '''

import os
import argparse
import re
from enum import IntEnum
from huggingface_hub import snapshot_download, hf_hub_download

# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1' # https://huggingface.co/docs/huggingface_hub/guides/download#faster-downloads


def parse_url(url):
    pattern = r'https://huggingface\.co/(?P<repo_id>[^/]+/[^/]+)(?:/blob/[^/]+/(?P<filename>.+))?'
    # pattern = r'https?://huggingface\.co/(?P<repo_id>[^/]+/[^/]+)(?:/blob/[^/]+/(?P<filename>[^/]+))?'
    match = re.match(pattern, url)
    if match:
        return {
            'repo_id': match.group('repo_id'),
            'filename': match.group('filename') if match.group('filename') else None
        }
    return {
        'repo_id': None,
        'filename': None
    }

def download_repo(repo_id, local_dir, allow_patterns=None, revision=None):
    snapshot_download(repo_id=repo_id,
                      local_dir=local_dir,
                      allow_patterns=allow_patterns,
                      revision=revision)

def download_file(repo_id, filename, local_dir, revision=None):
    hf_hub_download(repo_id=repo_id,
                    filename=filename,
                    local_dir=local_dir,
                    revision=revision)

class URLType(IntEnum):
    REPO = 0
    FILE = 1

def main():
    model_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Download from Hugging Face")
    parser.add_argument('url', type=str, help='Url of repo or file to be download.')
    parser.add_argument('--revision', type=str, help='Specify version')
    parser.add_argument('--local_dir', type=str, default=model_dir, help='Local directory to save files')

    args = parser.parse_args()

    repo_id, filename = parse_url(args.url).values()
    assert repo_id is not None, 'Only url from huggingface'

    utype = URLType.REPO if filename is None else URLType.FILE
    local_dir = os.path.join(args.local_dir, repo_id) if args.local_dir == model_dir else args.local_dir
    local_path = local_dir if filename is None else os.path.join(local_dir, filename)

    if utype == URLType.REPO:
        print(
            f'downloading REPO:',
            f'\t{"repo_id":<15}{repo_id}',
            f'\t{"filename":<15}{filename}',
            f'\t{"local_path":<15}{local_path}',
            sep='\n'
            )
        download_repo(repo_id, local_dir, args.revision)

    elif utype == URLType.FILE:
        print(
            f'downloading FILE:',
            f'\t{"repo_id":<15}{repo_id}',
            f'\t{"filename":<15}{filename}',
            f'\t{"local_path":<15}{local_path}',
            sep='\n'
            )
        download_file(repo_id, filename, local_dir, args.revision)


def test_parse_url():    
    # 示例用法
    url1 = 'https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev'
    print(parse_url(url1))  # ('black-forest-labs/FLUX.1-Canny-dev', None)

    url2 = 'https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev/blob/main/flux1-canny-dev.safetensors'
    print(parse_url(url2))  # ('black-forest-labs/FLUX.1-Canny-dev', 'flux1-canny-dev.safetensors')

    url3 = 'https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev'
    print(parse_url(url3))  # ('black-forest-labs/FLUX.1-Depth-dev', None)

    url4 = 'https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Canny/blob/main/diffusion_pytorch_model.safetensors'
    print(parse_url(url4))  # ('InstantX/FLUX.1-dev-Controlnet-Canny', 'diffusion_pytorch_model.safetensors')


if __name__ == "__main__":
    """虎牙海聪上运行示例，加`hai run2`可以翻墙
    hai run2 python3 hfdown.py https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev/blob/main/flux1-canny-dev.safetensors
    """
    main()
    # test_parse_url()