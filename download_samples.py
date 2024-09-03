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

    data_folder = "data"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        print(f"Created directory: {data_folder}")

    file_id_samples = "1bLB6fe36DUYmjcHUC4-01WD3ZUzwtxnm"
    output_folder_samples = os.path.join(data_folder, "sample_files")
    download_and_unpack_zip(file_id_samples, output_folder_samples, "sample_files.zip")
