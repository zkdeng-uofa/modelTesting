#!/usr/bin/env python3

import pandas as pd
import os
import sys
import aiohttp
import asyncio
import tarfile
import time
import mimetypes
import argparse
from pathlib import Path
from tqdm.asyncio import tqdm

def parse_args():
    """
    Parse user inputs from arguments using argparse.
    """
    parser = argparse.ArgumentParser(description="Download images asynchronously, track bandwidth, and tar the output folder.")
    
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input CSV or Parquet file.")
    parser.add_argument("--output_tar", type=str, required=True, help="Path to the output tar file (e.g., 'images.tar.gz').")
    parser.add_argument("--url_name", type=str, default="photo_url", help="Column name containing the image URLs.")
    parser.add_argument("--class_name", type=str, default="taxon_name", help="Column name containing the class names.")

    return parser.parse_args()

async def download_image_with_extensions(session, semaphore, row, output_folder, url_col, class_col, total_bytes):
    """Download an image asynchronously with retries for different file extensions, tracking actual stored size."""
    
    fallback_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.pdf']
    async with semaphore:
        key, image_url = row.name, row[url_col]
        class_name = row[class_col].replace("'", "").replace(" ", "_")
        base_url, original_ext = os.path.splitext(image_url)
        
        def save_and_track(content, file_path):
            """Helper function to write content to file and track size"""
            with open(file_path, 'wb') as f:
                f.write(content)
            file_size = os.path.getsize(file_path)  # Get actual stored size
            total_bytes.append(file_size)  # Track real disk size
        
        # If no extension, determine it dynamically
        if not original_ext:
            try:
                async with session.get(image_url) as response:
                    if response.status == 200:
                        content = await response.read()
                        mime_type = response.headers.get('Content-Type')
                        ext = mimetypes.guess_extension(mime_type) or ".jpg"
                        file_name = f"{base_url.split('/')[-1]}{ext}"
                        file_path = os.path.join(output_folder, class_name, file_name)
                        os.makedirs(os.path.dirname(file_path), exist_ok=True)

                        save_and_track(content, file_path)
                        return key, file_name, class_name, None
            except Exception as err:
                return key, None, class_name, str(err)

        else:
            # Try downloading with original extension
            file_name = f"{base_url.split('/')[-2]}{original_ext}"
            file_path = os.path.join(output_folder, class_name, file_name)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            try:
                async with session.get(image_url) as response:
                    if response.status == 200:
                        content = await response.read()
                        save_and_track(content, file_path)
                        return key, file_name, class_name, None
            except Exception as err:
                pass  # Try fallback extensions next

            # Try fallback extensions
            for ext in fallback_extensions:
                if ext == original_ext:
                    continue
                new_url = f"{base_url}{ext}"
                file_name = f"{base_url.split('/')[-2]}{ext}"
                file_path = os.path.join(output_folder, class_name, file_name)

                try:
                    async with session.get(new_url) as response:
                        if response.status == 200:
                            content = await response.read()
                            save_and_track(content, file_path)
                            return key, file_name, class_name, None
                except Exception:
                    continue  # Try the next extension

        # If all extensions fail
        return key, None, class_name, "All extensions failed."

async def main():
    args = parse_args()

    input_path = args.input_path
    output_tar_path = args.output_tar
    url_col = args.url_name
    class_col = args.class_name
    concurrent_downloads = 1000  # Fixed number of concurrent downloads

    output_folder = os.path.splitext(os.path.basename(output_tar_path))[0]
    if output_tar_path.endswith(".tar.gz"):
        output_folder = os.path.splitext(output_folder)[0]

    if input_path.endswith(".parquet"):
        df = pd.read_parquet(input_path)
    elif input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    else:
        print("Unsupported file format. Please provide a CSV or Parquet file.")
        sys.exit(1)

    semaphore = asyncio.Semaphore(concurrent_downloads)
    total_bytes = []  # List to track total bytes downloaded

    start_time = time.monotonic()  # Start timer

    async with aiohttp.ClientSession() as session:
        tasks = [download_image_with_extensions(session, semaphore, row, output_folder, url_col, class_col, total_bytes) for _, row in df.iterrows()]
        
        errors = 0
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            key, file_name, class_name, error = await future
            if error:
                errors += 1

    total_time = time.monotonic() - start_time  # Total time taken
    total_downloaded = sum(total_bytes)  # Total bytes downloaded

    if total_time > 0 and total_downloaded > 0:
        avg_speed = total_downloaded / total_time  # Bytes per second
        print(f"\nDownload Speed Stats:\n  - Total Data: {total_downloaded / 1e6:.2f} MB")
        print(f"  - Time Taken: {total_time:.2f} sec")
        print(f"  - Avg Speed: {avg_speed / 1e6:.2f} MB/s")
    else:
        print("\nNo successful downloads to compute bandwidth statistics.")


    # Tar the output folder
    # with tarfile.open(output_tar_path, "w:gz") as tar:
    #     tar.add(output_folder, arcname=os.path.basename(output_folder))

    # full_path = Path(output_tar_path).resolve()
    # print(f"\nTared output folder into: {full_path}")


if __name__ == '__main__':
    asyncio.run(main())
