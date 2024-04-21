import pandas as pd
import os
import pathlib
import ants
from typing import Union, List, Tuple
import multiprocessing
import SimpleITK as sitk
from nipype.interfaces.dcm2nii import Dcm2niix
import numpy as np
from HD_BET.run import run_hd_bet
from nipype.interfaces import fsl
from intensity_normalization.normalize.zscore import ZScoreNormalize
from nipype.interfaces.ants import Registration


N_PROC = multiprocessing.cpu_count() - 1


def dcm2niix_folder(
    path_to_folder: Union[str, pathlib.Path],
) -> List[str]:
    """
    Python implementation of command line use of dcm2niix processing each subfolder individually.
    Reason for implementation: dcm2niix recursive search can be extremely long with large datasets.

    Keyword arguments:
    path_to_folder: Union[str, pathlib.Path] - Path to main directory containing subdirectories for each patient

    Returns:
    List with case names (subfolders)
    """

    root = path_to_folder
    dirlist = [
        item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item))
    ]
    print(len(dirlist), "subfolders found")

    for dir in dirlist:
        converter = Dcm2niix()
        converter.inputs.source_dir = os.path.join(root, dir)
        converter.inputs.compress = "y"
        converter.inputs.merge_imgs = True
        converter.inputs.out_filename = "%i_%t_%n_%d_%f_%p_%q_%s_%z_%m"
        converter.inputs.output_dir = os.path.join(root, dir)
        converter.run()

    return dirlist


def get_caselist(
    path_to_file: Union[str, pathlib.Path],
) -> List[str]:
    """
    Returns list of column ID from cases.xlsx in provided directory
    """
    cases = pd.read_excel(
        os.path.join(path_to_file, "Cases.xlsx"),
        header=0,
        index_col=None,
        dtype={"ID": str},
    )

    return cases.ID.to_list()


def extract_brain(
    path_to_input_image: Union[str, pathlib.Path],
    path_to_output_image: Union[str, pathlib.Path],
):
    """
    applies FSL.Reorient2Std() to input brain scan
    and returns brain extracted image

    Keyword Arguments:
    path_to_input_image: Union[str, pathlib.Path] = file path to input image (brain scan)
    path_to_output_image: Union[str, pathlib.Path] = location to store brain extracted image
    """

    reorient = fsl.Reorient2Std()
    reorient.inputs.in_file = path_to_input_image
    reorient.inputs.out_file = path_to_output_image
    reorient.run()

    run_hd_bet(mri_fnames=path_to_output_image, output_fnames=path_to_output_image)


def fill_holes(
    binary_image: sitk.Image,
    radius: int = 3,
) -> sitk.Image:
    """
    Fills holes in binary segmentation

    Keyword Arguments:
    - binary_image: sitk.Image = binary brain segmentation
    - radius: int = kernel radius

    Returns:
    - closed_image: sitk.Image = binary brain segmentation with holes filled
    """

    closing_filter = sitk.BinaryMorphologicalClosingImageFilter()
    closing_filter.SetKernelRadius(radius)
    closed_image = closing_filter.Execute(binary_image)

    return closed_image


def binary_segment_brain(
    image: sitk.Image,
) -> sitk.Image:
    """
    Returns binary segmentation of brain from brain-extracted scan via otsu thresholding

    Keyword Arguments:
    - image: sitk.Image = brain-extracted scan

    Returns:
    - sitk.Image = binary segmentation of brain scan with filled holes
    """

    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(0)
    otsu_filter.SetOutsideValue(1)
    binary_mask = otsu_filter.Execute(image)

    return fill_holes(binary_mask)


def get_bounding_box(
    image: sitk.Image,
) -> Tuple[int]:
    """
    Returns bounding box of brain-extracted scan

    Keyword Arguments:
    - image: sitk.Image = brain-extracted scan

    Returns
    - bounding_box: Tuple(int) = bounding box (startX, startY, startZ, sizeX, sizeY, sizeZ)
    """

    mask_image = binary_segment_brain(image)

    lsif = sitk.LabelShapeStatisticsImageFilter()
    lsif.Execute(mask_image)
    bounding_box = np.array(lsif.GetBoundingBox(1))

    return bounding_box


def apply_bounding_box(
    image: sitk.Image,
    bounding_box: Tuple[int],
) -> sitk.Image:
    """
    Returns image, cropped to bounding box

    Keyword Arguments:
    - image: sitk.Image = image
    - bounding_box: Tuple(ing) = bounding box of kind (startX, startY, startZ, sizeX, sizeY, sizeZ)

    Returns
    - cropped_image: sitk.Image = cropped image
    """

    cropped_image = image[
        bounding_box[0] : bounding_box[3] + bounding_box[0],
        bounding_box[1] : bounding_box[4] + bounding_box[1],
        bounding_box[2] : bounding_box[5] + bounding_box[2],
    ]

    return cropped_image


def apply_bias_correction(
    image: sitk.Image,
) -> sitk.Image:
    """applies N4 bias field correction to image but keeps background at zero

    Keyword Arguments:
    image: sitk.Image = image to apply bias correction to

    Returns:
    image_corrected_masked: sitk.Image = N4 bias field corrected image
    """

    mask_image = binary_segment_brain(image)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    image_corrected = corrector.Execute(image, mask_image)

    mask_filter = sitk.MaskImageFilter()
    mask_filter.SetOutsideValue(0)
    image_corrected_masked = mask_filter.Execute(image_corrected, mask_image)

    return image_corrected_masked


def coregister_antspy(
    fixed_path: Union[str, pathlib.Path],
    moving_path: Union[str, pathlib.Path],
    out_path: Union[str, pathlib.Path],
    num_threads=N_PROC,
) -> ants.core.ants_image.ANTsImage:
    """
    Coregister moving image to fixed image. Return warped image and save to disk.

    Keyword Arguments:
    fixed_path: path to fixed image
    moving_path: path to moving image
    out_path: path to save warped image to
    num_threads: number of threads
    """

    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(num_threads)

    res = ants.registration(
        fixed=ants.image_read(fixed_path),
        moving=ants.image_read(moving_path),
        type_of_transform="antsRegistrationSyNQuick[s]",  # or "SyNRA"
        initial_transform=None,
        outprefix="",
        mask=None,
        moving_mask=None,
        mask_all_stages=False,
        grad_step=0.2,
        flow_sigma=3,
        total_sigma=0,
        aff_metric="mattes",
        aff_sampling=32,
        aff_random_sampling_rate=0.2,
        syn_metric="mattes",
        syn_sampling=32,
        reg_iterations=(40, 20, 0),
        aff_iterations=(2100, 1200, 1200, 10),
        aff_shrink_factors=(6, 4, 2, 1),
        aff_smoothing_sigmas=(3, 2, 1, 0),
        write_composite_transform=False,
        random_seed=None,
        verbose=False,
        multivariate_extras=None,
        restrict_transformation=None,
        smoothing_in_mm=False,
    )

    warped_moving = res["warpedmovout"]

    ants.image_write(warped_moving, out_path)

    return warped_moving


def coregister_nipype(
    fixed_path: Union[str, pathlib.Path],
    moving_path: Union[str, pathlib.Path],
    out_path: Union[str, pathlib.Path],
    num_threads=N_PROC,
)-> None:

    """
    Coregister moving image to fixed image and save to disk.

    Keyword Arguments:
    fixed_path: path to fixed image
    moving_path: path to moving image
    out_path: path to save warped image to
    num_threads: number of threads
    """

    reg = Registration()
    reg.inputs.fixed_image = fixed_path
    reg.inputs.moving_image = moving_path
    reg.inputs.output_transform_prefix = "output_"
    reg.inputs.transforms = ['Rigid', 'Affine', 'SyN']
    reg.inputs.num_threads = 64
    reg.inputs.float = True
    reg.inputs.transform_parameters = [(2.0,), (2.0,), (0.25, 3.0, 0.0)]
    reg.inputs.number_of_iterations = [[1000, 200], [1000, 200], [100, 50, 30]]
    reg.inputs.dimension = 3
    reg.inputs.write_composite_transform = True
    reg.inputs.collapse_output_transforms = False
    reg.inputs.initialize_transforms_per_stage = False
    reg.inputs.metric = ['Mattes'] * 3
    reg.inputs.metric_weight = [1] * 3
    reg.inputs.radius_or_number_of_bins = [32] * 3
    reg.inputs.sampling_strategy = ['Random', 'Random', None]
    reg.inputs.sampling_percentage = [0.05, 0.05, None]
    reg.inputs.convergence_threshold = [1.e-8, 1.e-8, 1.e-9]
    reg.inputs.convergence_window_size = [20] * 3
    reg.inputs.smoothing_sigmas = [[1,0], [1,0], [2,1,0]]
    reg.inputs.sigma_units = ['vox'] * 3
    reg.inputs.shrink_factors = [[2,1], [2,1], [3,2,1]]
    reg.inputs.use_estimate_learning_rate_once = [True] * 3
    reg.inputs.use_histogram_matching = [True] * 3
    reg.inputs.output_warped_image = out_path
    reg.run()


def resample(
    itk_image: sitk.Image,
    out_spacing: Tuple[float, ...],
    is_mask: bool,
) -> sitk.Image:
    """
    Resamples sitk image to expected output spacing

    Keyword Arguments:
    itk_image: sitk.Image
    out_spacing: Tuple
    is_mask: bool = True if input image is label mask -> NN-interpolation

    Returns
    output_image: sitk.Image = image resampled to out_spacing
    """

    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(round(osz * osp / nsp))
        for osz, osp, nsp in zip(original_size, original_spacing, out_spacing)
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(0)

    if is_mask:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)

    else:
        resample.SetInterpolator(
            sitk.sitkBSpline
        )  # sitk.sitkLinear sitk.sitkNearestNeighbor

    output_image = resample.Execute(itk_image)

    return output_image


def zscore_normalize(image: sitk.Image) -> sitk.Image:
    """
    Applies z score normalization to brain scan using a brain mask

    Keyword Arguments:
    image: sitk.Image = input brain scan

    Returns:
    normalized_brain_image: sitk.Image = normalized brain scan
    """

    brain_mask = binary_segment_brain(image)

    normalizer = ZScoreNormalize()
    normalized_brain_array = normalizer(
        sitk.GetArrayFromImage(image),
        sitk.GetArrayFromImage(brain_mask),
    )

    normalized_brain_image = sitk.GetImageFromArray(normalized_brain_array)
    normalized_brain_image.CopyInformation(image)

    return normalized_brain_image


def binarize_itkimage(
    img: sitk.Image,
    l_thresh: int,
    u_thresh: int,
) -> sitk.Image:
    """
    Binarizes and returns sitk.Image

    Keyword Arguments:
    img: sitk.Image
    """
    
    binary_filter = sitk.BinaryThresholdImageFilter()
    binary_filter.SetLowerThreshold(l_thresh)
    binary_filter.SetUpperThreshold(u_thresh)
    binary_filter.SetInsideValue(1)
    binary_filter.SetOutsideValue(0)

    return binary_filter.Execute(img)