import os
import gdown
import zipfile


def download_and_unpack_zip(file_id, output_folder, zip_name):
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: {output_folder}")

    url = f"https://drive.google.com/uc?id={file_id}"
    
    zip_path = os.path.join(output_folder, zip_name)
    print(f"Starting download for {zip_name}...")
    gdown.download(url, zip_path, quiet=False)
    print(f"Downloaded {zip_name} to {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_folder)
    print(f"Unpacked {zip_name} to {output_folder}")
    
    os.remove(zip_path)
    print(f"Deleted the ZIP file: {zip_name}")

    
if __name__ == "__main__":

    data_folder = "data/datasets"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        print(f"Created directory: {data_folder}")

    file_id_original = "1cwpjycJegdhZon0xdRYNc3odazSrcVXm"
    output_folder_original = os.path.join(data_folder, "action_images_dataset_original_v1.0")
    download_and_unpack_zip(file_id_original, output_folder_original, "action_images_original_v1.0.zip")
    
    file_id_relabeled = "1WRvpY0tltnippQFcXvF9zMfpnHY7tkQS"
    output_folder_relabeled = os.path.join(data_folder, "action_images_dataset_v1.0")
    download_and_unpack_zip(file_id_relabeled, output_folder_relabeled, "action_images_v1.0.zip")
