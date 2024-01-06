from huggingface_hub import hf_hub_download
import zipfile
from tqdm import tqdm

repo_IDs = ["liuhaotian/LLaVA-CC3M-Pretrain-595K"]  # ["liuhaotian/LLaVA-Pretrain"]  # "liuhaotian/LLaVA-CC3M-Pretrain-595K"
file_names = ["images.zip"]  # ["blip_laion_cc_sbu_558k.json", "images.zip"]             # [""]

local_dir = "data/LLaVA-Pretrain"


def extract(zip_path, extract_path, percentage=0.02):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get the total number of file entries in the zip archive
        total_files = len(zip_ref.infolist())

        # Calculate the number of files to extract based on the percentage
        files_to_extract = int(total_files * (percentage / 100))

        # Use tqdm for the progress bar
        with tqdm(total=files_to_extract, desc="Extracting", unit="files") as pbar:
            for file_info in zip_ref.infolist()[:files_to_extract]:
                zip_ref.extract(file_info, extract_path)
                pbar.update(1)


if __name__ == '__main__':
    for repo in repo_IDs:
        for file_n in file_names:
            """hf_hub_download(repo_id=repo, filename=file_n, local_dir=local_dir, resume_download=True,
                            local_dir_use_symlinks=False, repo_type="dataset")"""

            if file_n.endswith(".zip"):
                extract(zip_path=f"{local_dir}/{file_n}",
                        extract_path=f"{local_dir}/{file_n.replace('.zip', '')}")
