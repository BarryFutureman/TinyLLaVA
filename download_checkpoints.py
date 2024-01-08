from huggingface_hub import hf_hub_download, snapshot_download
import zipfile
from tqdm import tqdm
import os


if __name__ == '__main__':
    snapshot_download(repo_id="BarryFutureman/TinyLLaVA-1.1B-pretrained-projector",
                      local_dir="checkpoints/TinyLLaVA-1.1B-pretrained-projector", resume_download=True,
                      local_dir_use_symlinks=False, repo_type="model")
