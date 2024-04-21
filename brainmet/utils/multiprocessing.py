from tqdm.contrib.concurrent import process_map
from preprocessing import resample
from config import DATASET_PATH
import SimpleITK as sitk
from multiprocessing import cpu_count, Manager
import os

# No. parallel workers
N_PROC = cpu_count() - 1

# create global variable to store failed cases
manager = Manager()
failed = manager.dict()

# Load case IDs as iterable
case_list = ...


# wrapper function
def process_function(x):
    """
    wraps code that can be run in parallel for multiple subjects
    """
    # access dict of failed cases
    global failed

    # Example: Resampling
    try:
        img = sitk.ReadImage(os.path.join(DATASET_PATH, x, f"{x}_IMAGE_NAME.nii.gz"))
        resampled_img = resample(img, out_spacing=(1, 1, 1), is_mask=False)
        sitk.WriteImage(
            resampled_img, os.path.join(DATASET_PATH, x, f"{x}_IMAGE_NAME_RESAMPLED.nii.gz")
        )

    # if exception occurs, write to dict
    except Exception as e:
        failed[x] = e


if __name__ == "__main__":
    r = process_map(process_function, case_list, max_workers=N_PROC)

    # Save log
    with open(os.path.join(DATASET_PATH, "Multiprocessing_log.txt"), "w") as f:
        print(failed, file=f)
