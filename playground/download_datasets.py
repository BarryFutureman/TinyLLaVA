from huggingface_hub import hf_hub_download
import zipfile
from tqdm import tqdm
import os

# Pretrain Part 1
"""repo_IDs = ["liuhaotian/LLaVA-Pretrain", "BarryFutureman/LLaVA-Pretrain-Chunks"]
file_names_lst = [["blip_laion_cc_sbu_558k.json"],
                  ['chunk_1.zip', 'chunk_2.zip', 'chunk_3.zip', 'chunk_4.zip', 'chunk_5.zip', 'chunk_6.zip',
                   'chunk_7.zip', 'chunk_8.zip', 'chunk_9.zip', 'chunk_10.zip', 'chunk_11.zip', 'chunk_12.zip',
                   'chunk_13.zip', 'chunk_14.zip', 'chunk_15.zip', 'chunk_16.zip', 'chunk_17.zip', 'chunk_18.zip'],
                  ]
"""

# minigpt4 align
"""repo_IDs = ["BarryFutureman/cc_sbu_align_llava"]
file_names_lst = [
    ["filter_cap_llava.json", "images.zip"],
                  ]"""

# Instruct Tune Part 2
repo_IDs = ["liuhaotian/LLaVA-Instruct-150K",
            "BarryFutureman/COCO2017_for_llava_chunks",
            "BarryFutureman/GQA_for_llava_chunks",
            "",
            "BarryFutureman/TextVQA_for_llava_chunks",
            "BarryFutureman/VisualGenomePart1_for_llava_chunks",
            "BarryFutureman/VisualGenomePart2_for_llava_chunks"]
file_names_lst = [["llava_v1_5_mix665k.json"],
                  ['chunk_1.zip', 'chunk_2.zip', 'chunk_3.zip', 'chunk_4.zip', 'chunk_5.zip', 'chunk_6.zip',
                   'chunk_7.zip', 'chunk_8.zip', 'chunk_9.zip', 'chunk_10.zip', 'chunk_11.zip', 'chunk_12.zip',
                   'chunk_13.zip', 'chunk_14.zip', 'chunk_15.zip', 'chunk_16.zip', 'chunk_17.zip', 'chunk_18.zip'],
                  ['chunk_1.zip', 'chunk_2.zip', 'chunk_3.zip', 'chunk_4.zip', 'chunk_5.zip', 'chunk_6.zip',
                   'chunk_7.zip', 'chunk_8.zip', 'chunk_9.zip', 'chunk_10.zip', 'chunk_11.zip', 'chunk_12.zip'],
                  ['chunk_1.zip', 'chunk_2.zip', 'chunk_3.zip', 'chunk_4.zip', 'chunk_5.zip', 'chunk_6.zip',
                   'chunk_7.zip', 'chunk_8.zip', 'chunk_9.zip', 'chunk_10.zip', 'chunk_11.zip', 'chunk_12.zip',
                   'chunk_13.zip', 'chunk_14.zip', 'chunk_15.zip', 'chunk_16.zip', 'chunk_17.zip', 'chunk_18.zip'],
                  ['chunk_1.zip', 'chunk_2.zip', 'chunk_3.zip', 'chunk_4.zip', 'chunk_5.zip', 'chunk_6.zip'],
                  ['chunk_1.zip', 'chunk_2.zip', 'chunk_3.zip', 'chunk_4.zip', 'chunk_5.zip', 'chunk_6.zip',
                   'chunk_7.zip', 'chunk_8.zip', 'chunk_9.zip'],
                  ['chunk_1.zip', 'chunk_2.zip']
                  ]

local_dir = "data"


def extract(zip_path, extract_path, percentage=0.001):
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
    for index, repo in enumerate(repo_IDs):
        file_names = file_names_lst[index]
        for file_n in file_names:
            print(file_n)
            extract_path = f"data"
            if "coco" in file_n.lower():
                extract_path = f"data/coco"
            elif "gqa" in file_n.lower():
                extract_path = f"data/gqa"
            elif "ocr" in file_n.lower():
                extract_path = f"data/ocr_vqa"
            elif "text" in file_n.lower():
                extract_path = f"data/train_images"
            elif "visual" in file_n.lower():
                extract_path = f"data/vg"

            hf_hub_download(repo_id=repo, filename=file_n, local_dir=local_dir, resume_download=True,
                            local_dir_use_symlinks=False, repo_type="dataset")

            if file_n.endswith(".zip"):
                zip_file_path = f"{local_dir}/{file_n}"
                extract(zip_path=zip_file_path,
                        extract_path=extract_path)

                os.remove(zip_file_path)

            break
