"""Data ingestion module for SBUS Captions and Narratives Dataset.

This module provides functions to process and extract images and captions
from parquet files in parallel.
"""

import os
import argparse
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm


def _process_row(args) -> dict[str, str | int]:
    """
    Process a single row of data to extract image and caption.

    Args:
        args (tuple): Tuple containing (index, row, images_dir).

    Returns:
        dict: Dictionary with id, image_path, and caption.
    """
    idx, row, images_dir = args

    caption = row["descript"]
    image_bytes = row["image.bytes"]
    image_path = os.path.join(images_dir, row["image.path"])

    with open(image_path, "wb") as f:
        f.write(image_bytes)

    return {"id": idx, "image_path": image_path, "caption": caption}


def process_parquet(df: pd.DataFrame, images_dir: str) -> pd.DataFrame | None:
    """
    Process parquet dataframe in parallel to extract images and captions.

    Args:
        df (pd.DataFrame): Input dataframe containing image data and captions.
        images_dir (str): Directory path to save extracted images.

    Returns:
        pd.DataFrame | None: Processed dataframe with image paths and captions,
            or None if no data was processed.
    """
    with Pool() as pool:
        processed_data = pool.map(
            _process_row, [(idx, row, images_dir) for idx, row in df.iterrows()]
        )

    if processed_data:
        return pd.DataFrame(processed_data)
    else:
        return None


def main(args: argparse.Namespace) -> None:
    """
    Main function to process SBUS Captions and Narratives Dataset.

    Reads parquet files, extracts images and captions, and saves the processed
    data to a parquet file.

    Args:
        args: Command-line arguments containing data_dir and output_dir.

    Returns:
        None
    """
    data_dir = args.data_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    output_df_path = os.path.join(output_dir, "processed_data.parquet")

    ## initialize empty parquet file
    output_df = pd.DataFrame(columns=["id", "image_path", "caption"])
    output_df.to_parquet(output_df_path)

    parquet_files = [f for f in os.listdir(data_dir) if f.endswith(".parquet")]

    for parquet in tqdm(parquet_files, desc="Processing parquets"):
        df = pd.read_parquet(os.path.join(data_dir, parquet))
        temp_df = process_parquet(df, images_dir)

        ## append to output parquet file
        if temp_df is not None:
            output_df = pd.concat([output_df, temp_df], ignore_index=True)
            output_df.to_parquet(output_df_path)

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest SBUS Captions and Narratives Dataset"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="sbucaptions-narratives/data",
        help="Directory where the SBUS Captions and Narratives dataset is stored",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="processed_data",
        help="Directory to save processed data",
    )
    args = parser.parse_args()
    main(args)
