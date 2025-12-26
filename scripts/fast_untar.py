import tarfile
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


def extract_batch(args):
    """
    Extracts a batch of members from a tar file.
    """
    tar_path, members, extract_path = args
    try:
        # Open the tar file in read mode
        with tarfile.open(tar_path, "r") as tar:
            for member in members:
                tar.extract(member, path=extract_path)
    except Exception as e:
        print(f"Error extracting batch: {e}")


def fast_untar(tar_path, extract_path, num_workers=None):
    """
    Extracts a tar file in parallel using multiprocessing.
    """
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)

    print(f"Scanning tarfile members from {tar_path}...")

    # Collect all members first to distribute them
    members = []
    with tarfile.open(tar_path, "r") as tar:
        members = tar.getmembers()

    total_files = len(members)
    print(f"Found {total_files} files. Preparing batches...")

    if num_workers is None:
        # Default to CPU count
        num_workers = os.cpu_count() or 4

    # Determine batch size
    # We want enough batches to keep workers busy, but not too many to cause overhead
    num_batches = num_workers * 4
    batch_size = max(1, total_files // num_batches)

    batches = []
    for i in range(0, total_files, batch_size):
        batch_members = members[i : i + batch_size]
        batches.append((tar_path, batch_members, extract_path))

    print(
        f"Starting parallel extraction with {num_workers} workers and {len(batches)} batches..."
    )

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(
            tqdm(executor.map(extract_batch, batches), total=len(batches), unit="batch")
        )

    print("Extraction complete.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/fast_untar.py <tar_file> [extract_path]")
        sys.exit(1)

    tar_file = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "."

    fast_untar(tar_file, out_dir)
