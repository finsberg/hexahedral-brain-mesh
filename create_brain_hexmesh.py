from pathlib import Path
import pooch
import skimage.morphology as skim
import skimage.filters as skif
import pandas as pd
import numpy as np
import nibabel as nib


SEGMENTATION = pooch.create(
    path=pooch.os_cache("mni"),
    base_url="https://doc.ibs.tu-berlin.de/cedalion/datasets/v25.1.0/",
    env="MNI_DATA_DIR",
    version="v1.0.0",
    registry={
        "ICBM152_2020.zip": "sha256:8dda23aa1f4592d50ba8528bb4ef7124c6593872bddeb9cbd510e7b1891568f3",  # noqa: E501
    },
)
TEMPLATE = pooch.create(
    path=pooch.os_cache("mni"),
    base_url="https://www.bic.mni.mcgill.ca/~vfonov/icbm/2020/",
    env="MNI_DATA_DIR",
    version="v1.0.0",
    registry={
        "icbm152_ext55_model_sym_2020_nifti.zip": "sha256:28244a77baf21f6e8d5e3907cf8ed90a86aaacc6150809cb76f03957b4fce505",  # noqa: E501
    },
)


def fix_mask(
    mask, ball_radius: float = 1.0, hole_size: int = 1000, sigma_gaussian: float = 1.0
):
    print(f"Generating surface")

    mask = skim.binary_dilation(mask, skim.ball(ball_radius))  # type: ignore[call-arg]
    mask = skim.remove_small_objects(mask, hole_size)  # type: ignore[call-arg]
    mask = skim.remove_small_holes(mask, hole_size)  # type: ignore[call-arg]
    # mask = skif.gaussian(mask, sigma=sigma_gaussian).astype(bool)
    return mask


def main():
    template_fnames = TEMPLATE.fetch(
        "icbm152_ext55_model_sym_2020_nifti.zip", processor=pooch.Unzip()
    )
    segmentation_fnames = SEGMENTATION.fetch(
        "ICBM152_2020.zip", processor=pooch.Unzip()
    )

    template_folder = Path(template_fnames[0]).parent

    data = (
        nib.loadsave.load(
            template_folder / "mni_icbm152_CerebrA_tal_nlin_sym_55_ext.nii"
        )
        .get_fdata()
        .astype(np.uint8)
    )
    seg_folder = Path(segmentation_fnames[0]).parent
    gm = fix_mask(
        nib.loadsave.load(seg_folder / "mask_gray.nii").get_fdata().astype(bool)
    )
    wm = fix_mask(
        nib.loadsave.load(seg_folder / "mask_white.nii").get_fdata().astype(bool)
    )
    # csf = fix_mask(
    #     nib.loadsave.load(seg_folder / "mask_csf.nii").get_fdata().astype(bool)
    # )

    brain = np.zeros(data.shape, dtype=np.uint8)

    # brain[gm | wm | csf] = 1  # brain mask
    brain[gm | wm] = 1  # brain mask
    brain[gm] = 2  # gray matter
    brain[wm] = 3  # white matter
    # brain[csf] = 4  # csf

    # Can read these from the labels as well
    # but hardcoding for now to avoid dependency on this csv file.
    # labels = pd.read_csv("CerebrA_LabelDetails.csv")
    third_ventricle_rh = 29
    third_ventricle_lh = 80
    fourhth_ventricle_rh = 37
    fourhth_ventricle_lh = 88
    lateral_ventricle_rh = 41
    lateral_ventricle_lh = 92
    inf_lat_vent_rh = 5
    inf_lat_vent_lh = 56
    ventricles = fix_mask(
        np.isin(
            data,
            [
                third_ventricle_rh,
                third_ventricle_lh,
                fourhth_ventricle_rh,
                fourhth_ventricle_lh,
                lateral_ventricle_rh,
                lateral_ventricle_lh,
                inf_lat_vent_rh,
                inf_lat_vent_lh,
            ],
        )
    )

    brain[ventricles] = 4  # ventricles

    # breakpoint()
    FILE_NAME = "brain_hexahedral.npy"
    np.save(FILE_NAME, brain)
    print(f"Saved {FILE_NAME} with shape {data.shape}.")


if __name__ == "__main__":
    main()
