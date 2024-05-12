"""Module for the RandomDeformationDataset class."""

import glob
import json
import os
import time

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as nnf
from torch.utils.data import Dataset
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import random
from typing import Tuple, Optional, Literal, Union
from jaxtyping import Float32, UInt64, Float
from torch import Tensor
from einops import rearrange
import skimage
import torch.nn.functional as F

# Get the directory one level up
#parent_dir = Path(__file__).resolve().parent.parent

# Add it to sys.path
#sys.path.append(str(parent_dir))

from spine_diffusion.datasets.augmentations import RandomErosion
from spine_diffusion.datasets.spatial_transformer import (
    SpatialTransformer,
    gauss_gen3D,
)
from spine_diffusion.utils.weighted_bce_loss import (
    calculate_weight_matrix,
)
from spine_diffusion.utils.array_reshaping import (
    pad_mask_to_size,
    resample_reorient_to,
)
from spine_diffusion.utils.general_utils import nib_to_numpy
from spine_diffusion.utils.handcrafting_segmentations import (
    handcraft_lq_segmentation,
)
from spine_diffusion.utils.label_manipulation import (
    convert_labels_to_conseq,
    remove_neverused_labels_and_crop,
    remove_unused_labels,
)

LQ_IDX = [1, 7, 13, 19, 25, 31, 37, 43, 49, 55, 61, 67, 73, 79, 85, 91, 97, 103, 109, 115, 121, 127, 133, 139, 145, 151, 157, 163, 169, 175, 181, 187, 193, 199, 205, 211, 217, 223]

class RandomDeformationDataset(Dataset):
    """Dataset class for the random deformation dataset using the full spine approach."""

    def __init__(
        self,
        hqPath: str,
        lqPath: str,
        classSettingsFile: str,
        mode: Literal["train","val","test"],
        resolution: int,
        class_mode: Literal["nerves","occ","all"],
        rand_coords: bool,
        num_points: int,
        patient: str = None,
        hqSeg: bool = False,
        lqSeg: bool = False,
        handcraft_lq: bool = True,
        spacing: Tuple[float] = (1, 1, 1),
        orientation: Tuple[str] = ("P", "L", "I"),
        imageShape: Tuple[int] = (256, 256, 256),
        alpha: float = 0.66,
        beta: int = 5,
        zeta: int = 0,
        eta: int = 1,
        omega: int = 4,
        randomSeed: int = 42,
        debug: bool = False,
        overwrite: bool = False,
        weightEq: bool = False,
        num_samples: int = 300,
        random_crop: bool = False,
        random_deformation: bool = True,
        update: bool = False,
    ) -> None:
        """Constructor for RandomDeformationDataset.

        Args:
            hqPath: Path to high-quality files (parent directory of 01_training etc.)
            lqPath: Same but for low-quality files
            patient: Name of the patient directory, usually only one directory 
                in this folder (e.g. model-patient, sub-verse506)
            classSettingsFile: path to json file containing the class settings
            hqSeg: Whether to use high-quality segmentations. Defaults to False.
            lqSeg: Same but for low-quality. Defaults to False.
            handcraft_lq: Whether to handcraft the segmentations or load 
                real files. Defaults to False.
            spacing: Common resolution. Defaults to (1, 1, 1).
            orientation: Common orientation. Defaults to ("P", "L", "I").
            imageShape: Output image shape, only to be changed with random_crop. 
                Defaults to (256, 256, 256).
            alpha: Handcrafting hyperparameter. Defaults to 0.66.
            beta: Handcrafting hyperparameter. Defaults to 5.
            zeta: Weighted loss hyperparameter. Defaults to 1.
            eta: Weighted loss hyperparameter. Defaults to 15.
            omega: Weighted loss hyperparameter. Defaults to 4.
            randomSeed: Random Seed. Defaults to 42.
            mode: "train", "val" or "test". Defaults to "train".
            debug: Print more debug information. Defaults to False.
            overwrite: Overwrite previously preprocessed files. Defaults to False.
        """
        self.resolution = resolution
        self.multiplier = 256 // resolution
        self.rand_coords = rand_coords
        self.num_points = num_points
        self.occ = True
        self.class_mode = class_mode
        # changes by Lionel

        self.patient = patient
        self.hqPath = hqPath
        self.lqPath = lqPath
        classSettingsFile = os.path.join(hqPath, classSettingsFile)
        self.classSettingsFile = classSettingsFile
        self.hqSeg = hqSeg
        self.lqSeg = lqSeg
        self.handcraftLq = handcraft_lq
        self.spacing = spacing
        self.orientation = orientation
        self.imageShape = imageShape
        self.alpha = alpha
        self.beta = beta
        self.zeta = zeta
        self.eta = eta
        self.omega = omega
        self.randomSeed = randomSeed
        self.mode = mode
        self.debug = debug
        self.overwrite = overwrite
        self.weightEq = weightEq
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.num_samples = num_samples
        if random_crop:
            raise NotImplementedError("Cropping not supported in this version.")
        self.random_crop = random_crop
        self.random_deformation = random_deformation
        self.mode = mode

        self.mode_folder = (
            "01_training"
            if mode == "train"
            else "02_validation" if mode == "val" else "03_test"
        )

        if self.patient is None:
            print(
                "No specific patient is chosen. Taking all patients"
                " in this dataset folder."
            )
            self.patient = "**"

        self.read_class_settings(classSettingsFile)

        if hqSeg:
            self.hqFolder = "segmentations_full"
        else:
            self.hqFolder = "derivatives_full"

        if mode == "train":
            self.hqFiles = sorted(
                glob.glob(
                    os.path.join(
                        self.hqPath,
                        self.mode_folder,
                        self.hqFolder,
                        self.patient,
                        "*.nii.gz",
                    )
                )
            )
        else:
            try:
                self.hqFiles = sorted(
                    glob.glob(
                        os.path.join(
                            self.hqPath,
                            self.mode_folder,
                            self.hqFolder,
                            self.patient,
                            "*.nii.gz",
                        )
                    )
                )
            except:
                print('This patient was not found in the folder:', self.patient)
                self.hqFiles = sorted(
                    glob.glob(
                        os.path.join(
                            self.hqPath,
                            self.mode_folder,
                            self.hqFolder,
                            self.patient,
                            "*/*.nii.gz",
                        )
                    )
                )

        print('hqFiles:', len(self.hqFiles))

        if lqSeg:
            self.lqFolder = "segmentations_full"
        else:
            self.lqFolder = "derivatives_full"

        self.lqFiles: list[str] = []

        if self.handcraftLq:
            randomState = np.random.RandomState(self.randomSeed)
            self.transform = RandomErosion(randomState, self.alpha, self.beta)

        ### Instead of loading the model in get_item, load it here 
        print('Preprocessing files...')
        hqFiles = []
        for hqFile in self.hqFiles:
            if (
                    os.path.isfile(
                        hqFile.replace(
                            self.hqFolder, f"{self.hqFolder}_preprocessed_reconnet"
                        )
                    )
                    and not self.overwrite
            ):
                if self.debug:
                    print(
                        "Loading preprocessed file"
                        f" {hqFile.replace(self.hqFolder, f'{self.hqFolder}_preprocessed_reconnet')}"
                    )
                hqFile = hqFile.replace(
                    self.hqFolder, f"{self.hqFolder}_preprocessed_reconnet"
                )
                hqImg = nib.load(hqFile)
            else:
                if self.debug:
                    print(f"Preprocessing file {hqFile}")

                hqImg = nib.load(hqFile)
                hqImg = resample_reorient_to(hqImg, self.orientation, self.spacing)
                hqImg = remove_neverused_labels_and_crop(hqImg, self.hqAllClasses)
                
                hqImg = nib.Nifti1Image(
                    pad_mask_to_size(nib_to_numpy(hqImg), self.imageShape),
                    affine=hqImg.affine,
                )

                hqFile = hqFile.replace(
                    self.hqFolder, f"{self.hqFolder}_preprocessed_reconnet"
                )
                os.makedirs(os.path.dirname(hqFile), exist_ok=True)
                nib.save(hqImg, hqFile)
                
            hqFiles.append(hqFile)

        self.hqFiles = hqFiles
        
        print('Preprocessing done. Files:', len(self.hqFiles), 'mode:', self.mode)

        # Initialize the spatial-transformer and Gaussian blur used for smoothing
        self.spatial_transformer = SpatialTransformer(
            size=self.imageShape, mode="nearest")
        self.blur_tensor = gauss_gen3D(n=5, s=5, sigma=2)

        print('hqLabels:', self.hqLabels)

        self.numclasses = len(self.hqLabels.keys())

        if self.mode != 'train':
            print('Using fixed undersampling for LQ images.')
            self.idx = LQ_IDX
        else:
            self.idx = None

    def read_class_settings(self, classSettingsFile) -> None:
        """Extract the needed information from the class settings files.

        This function sets the following attributes:
        - hqLabels: Dictionary containing the labels for the high-quality dataset.
        - lqLabels: Dictionary containing the labels for the low-quality dataset.
        - hqClasses: Dictionary containing the classes for the high-quality dataset, which label is a vertebra, disc or spinal canal. This is required for determining the number of erosion iterations.
        - lqClasses: same as hqClasses but for low-quality.
        - hqAllClasses: Dictionary containing the classes for the high-quality dataset. This is used to determine which classes can safely be deleted from the segmentation before preprocessing. Everything that is not in this dict will not be in the saved preprocessed file.
        - lqAllClasses: same as hqAllClasses but for low-quality.
        - objectTypes: unique types of objects that appear in hqClasses (should be the same as lqClasses)

        Args:
            classSettingsFile (str): path to json file containing the class settings.
        """
        self.classSettings = json.load(open(classSettingsFile, "r"))

        self.hqLabels = {
            int(k): v for k, v in self.classSettings["hqLabels"].items()
        }
        self.lqLabels = {
            int(k): v for k, v in self.classSettings["lqLabels"].items()
        }

        self.hqClasses = {
            int(k): v for k, v in self.classSettings["hqClasses"].items()
        }
        self.lqClasses = {
            int(k): v for k, v in self.classSettings["lqClasses"].items()
        }
        self.hqAllClasses = {
            int(k): v for k, v in self.classSettings["hqAllClasses"].items()
        }
        self.lqAllClasses = {
            int(k): v for k, v in self.classSettings["lqAllClasses"].items()
        }
        self.objectTypes = set(self.hqClasses.values())

    def deform_random_hq(
        self, img: torch.Tensor, weight: torch.Tensor, it_mean=3, it_sigma=1, inter_range=30
    ) -> torch.Tensor:
        """Randomly deform high-quality image.

        Args:
            img (torch.Tensor): image to deform.
            weight (torch.Tensor): pre-calculated weight to transform
            it_mean (int, optional): Mean of gaussian from which number of smoothing iterations is sampled. Defaults to 100.
            it_sigma (int, optional): Sigma of gaussian from which number of smoothing iterations is sampled. Defaults to 40.
            device (str, optional): cuda or cpu. Defaults to "cpu".

        Returns:
            torch.Tensor: deformed image.
        """
        assert len(img.shape) == 5, (
            "Input image needs to have shape [1,1,x,y,z], where x,y,z is not"
            " indicating orientation"
        )

        # # build transformer layer
        # spatial_transformer = SpatialTransformer(
        #     size=(img.shape[2], img.shape[3], img.shape[4]), mode="nearest"
        # )

        # initialize displacement layer
        #disp_tensor = torch.randn(img.shape[2], img.shape[3], img.shape[4])[None, None, :].to(device)
        # blur_tensor = gauss_gen3D(n=5, s=5, sigma=2).to(device)

        # sample parameter: The magnitude is proportional to the iterations. Since the more iterations the smaller values.
        random_res = np.random.randint(1, inter_range, size=1)[0]
        resolution = img.shape[2] // random_res
        disp_init = torch.randn(resolution, resolution, resolution)[None, None, :]
        disp_tensor = torch.nn.functional.interpolate(disp_init, size=[img.shape[2], img.shape[3], img.shape[4], ],
                                                      mode="trilinear")

        # create the deformation field
        it_num = np.clip(int(np.abs(np.random.normal(it_mean, it_sigma, 1))), 0, 10)
        mag_num = np.clip(int(np.abs(np.random.normal(it_num, int(it_num * 0.4), 1))), 0.1, 10)
        for i in range(it_num):
            disp_tensor = nnf.conv3d(disp_tensor.float(), self.blur_tensor.float(), padding=2)

        # warp the image with the transformer
        moved_img = self.spatial_transformer(img.float(), disp_tensor.float() * mag_num)
        moved_weight = self.spatial_transformer(weight.float(), disp_tensor.float() * mag_num)

        return moved_img, moved_weight

    def __len__(self) -> int:
        """Length of the dataset.

        Returns:
            int: length
        """
        # if not self.mode == "test" and len(self.hqFiles) == 1:
        return len(self.hqFiles)*self.num_samples

    def __getitem__(
        self, index
    ):
        """__getitem__ method of the dataset.

        Metadata:
        - name: patient identifier
        - affine: affine matrix of the high-quality image
        - object: "all", ends up in the output filename
        - label_transform_hq: label transform for the high-quality image, multi-class prediction needs consecutive labels but the input labels are not, this is the mapping
        - label_transform_lq: same as label_transform_hq but for the low-quality image

        Args:
            index (int): which patient to load

        Raises:
            NotImplementedError: if not handcraftLq because for real low-quality images this was not implemented yet.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]: high-quality image, low-quality image, weight matrix, metadata
        """

        ### Change deformation to also change the weights

        hqFile = self.hqFiles[index % len(self.hqFiles)]

        hqImg = nib.load(hqFile)
        hqImg = remove_unused_labels(hqImg, self.hqLabels)
        hqNp = nib_to_numpy(hqImg).astype(np.uint8)

        weight = torch.from_numpy(calculate_weight_matrix(
            hqNp, self.zeta, self.eta, self.omega)).detach()
        weight[hqNp > 0] = 1.0
        weight = weight.unsqueeze(0).unsqueeze(0)
        hqTen = torch.from_numpy(hqNp)
        hqTen = hqTen.unsqueeze(0).unsqueeze(0)

        start_deformation = time.time()
        if self.random_deformation:
            hqTen, _ = self.deform_random_hq(hqTen, weight)

        hqNp = hqTen.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
        end_deformation = time.time()

        if self.debug:
            print(
                "Deformation took: {:.3f}s".format(
                    end_deformation - start_deformation
                )
            )

        # in testing mode either defined you want to use handcrafted or assume you want to use real MRI
        if not self.handcraftLq:
            raise NotImplementedError(
                "Random deformation only implemented for handcrafting!"
            )
        else:
            # Downsample hqImg, erosion and randomly chose 17-21 slices
            (lq, idx), lqNp = handcraft_lq_segmentation(
                hqNp,
                self.hqClasses,
                self.classSettings,
                self.mode,
                self.transform,
                self.idx,
            )

        hqTen = torch.from_numpy(hqNp.astype(np.uint8))

        lqTen = torch.from_numpy(lqNp)
        lqTen = lqTen.type(hqTen.dtype)
        
        #weightTen = torch.from_numpy(weight)
        # weightTen = weightTen.squeeze(0).squeeze(0)
        
        hqTen, label_transform_hq = convert_labels_to_conseq(
            hqTen, self.hqLabels
        )
        lqTen, label_transform_lq = convert_labels_to_conseq(
            lqTen, self.lqLabels
        )
        
        metadata = {
            "name": os.path.basename(os.path.dirname(hqFile)),
            "object": "all",
            "affine": hqImg.affine,
            "label_transform_hq": label_transform_hq,
            "label_transform_lq": label_transform_lq,
        }

        assert (
            hqTen.shape == lqTen.shape == self.imageShape
        ), f"shapes dont match {hqTen.shape}, {lqTen.shape}, {self.imageShape}"
        
        individual_voxel_ids = [torch.arange(num_elements) for num_elements in hqTen.shape]
        individual_voxel_ids_meshed = torch.meshgrid(individual_voxel_ids, indexing='ij')        
        voxel_ids = torch.stack(individual_voxel_ids_meshed, -1)
        
        spacing = np.array(self.spacing)
        offset = spacing / 2
        coords_hq = voxel_ids * spacing + offset

        if ((self.mode == "train") or (self.mode == "tune")) and self.random_crop:
            if self.mode == 'tune':
                hqTen = hqTen[:, idx, :]
                hqTen[hqTen == 11] = 0
                coords_hq = coords_hq[:, idx, :]
                hqTen, coords_hq, crop_idx = random_crop_3d(hqTen, coords_hq, crop_size=(64, len(idx), 64))
                metadata["idx"] = np.array(idx)
            else:
                hqTen, coords_hq, crop_idx = random_crop_3d(hqTen, coords_hq, crop_size=(64, 64, 64))
                metadata["idx"] = np.array(())
            metadata["crop_idx"] = crop_idx
        else:
            metadata["crop_idx"] = ()
            metadata["idx"] = np.array(())

        # changes by Lionel
        res = {}

        # downsample
        hqTen = skimage.measure.block_reduce(hqTen, (self.multiplier, self.multiplier, self.multiplier), np.max)

        res["occ"] = torch.from_numpy(hqTen).to(torch.long).unsqueeze(0)
        if self.class_mode == "occ":
            res["occ"] = (res["occ"] > 0).to(torch.long)
        elif self.class_mode == "nerves":
            nerves = (res["occ"] == 10).to(torch.long) * 2
            others = ((res["occ"] > 0) & (res["occ"] < 10)).to(torch.long)
            empty = torch.zeros_like(nerves)
            res["occ"] = empty + others + nerves

        if self.rand_coords:
            coords = self._sample_normalized_coords(self.num_points)
            vals = self._get_coord_values(res["occ"].to(torch.float32), coords).to(torch.long).squeeze(0)
            res["coords"] = coords
            res["targets_occ"] = vals
        else:
            coords_hq = coords_hq / 255 - (0.5/255)
            res["coords"] = rearrange(coords_hq.to(torch.float32), "r r r n -> (r r r) n")
            res["targets_occ"] = rearrange(res["occ"], "c r r r -> c (r r r)").squeeze(0)

        res["loss_fn"] = "crossentropylogits"

        return res["occ"]
    
    def _get_coord_values(
            self,
            volume: Float32[Tensor, "ch depth height width"], 
            coords: Float32[Tensor, "num_coords 3"]
        ) -> Union[
            Float32[Tensor, "ch num_coords"],
            UInt64[Tensor, "ch num_coords"]
        ]:
        volume_unsq = volume.unsqueeze(0) # add batch dim
        coords_unsq = coords[None, None, None, :, :]
        if self.occ:
            vals = F.grid_sample(volume_unsq, coords_unsq, mode="nearest", align_corners=True)
            vals = vals.to(torch.long)[0,:,0,0]
        else:
            vals = F.grid_sample(volume_unsq, coords_unsq, mode="bilinear", align_corners=True)
            vals = vals[0,:,0,0]
        assert vals.size() == (volume.shape[0], coords.shape[0]), f"{vals.size()}{volume.size()}{coords.size()}"
        return vals
    
    def _sample_normalized_coords(self, num_coords: int) -> Float32[Tensor, "num_coords 3"]:
        coords = torch.rand((num_coords, 3))
        return coords * 2 - 1


def random_crop_3d(tensor, tensor2, crop_size=(64, 64, 64)):
    """
    Randomly crops a 3D tensor to the specified size.

    Args:
    - tensor (torch.Tensor): The input 3D tensor to be cropped.
    - crop_size (tuple): The desired crop size in (depth, height, width) dimensions.

    Returns:
    - cropped_tensor (torch.Tensor): The randomly cropped 3D tensor.
    """
    depth, height, width = crop_size
    max_x = tensor.shape[2] - width
    max_y = tensor.shape[1] - height
    max_z = tensor.shape[0] - depth

    if max_x < 0 or max_y < 0 or max_z < 0:
        raise ValueError("Crop size exceeds the dimensions of the input tensor.")

    # Randomly select the starting coordinates of the crop
    random_x = random.randint(0, max_x)
    random_y = random.randint(0, max_y)
    random_z = random.randint(0, max_z)

    # Crop the tensor
    cropped_tensor1 = tensor[random_z:random_z + depth, random_y:random_y + height, random_x:random_x + width]
    cropped_tensor2 = tensor2[random_z:random_z + depth, random_y:random_y + height, random_x:random_x + width]

    crop_idx = (random_x, random_y, random_z, depth, height, width)

    return cropped_tensor1, cropped_tensor2, crop_idx

def save_nib(img, filename, affine):
    nib.save(nib.Nifti1Image(img.numpy(), affine), filename)
    

def plot_examples(imgs, sample, n_images=10):
    fig, axs = plt.subplots(3, n_images, figsize=(15, 10))
    step = 20

    for i in range(n_images):
        axs[0, i].imshow(imgs[i*step, :, :], cmap='gray')
        axs[1, i].imshow(imgs[:, i*step, :], cmap='gray')
        axs[2, i].imshow(imgs[:, :, i*step], cmap='gray')

    plt.savefig(f'/home/klanna/github/Spine3D/DeepSDF/tests/example_{sample}.png')




if __name__ == "__main__":
    path = '/usr/bmicnas02/data-biwi-01/lumbarspine/datasets_lara/'
    hqPath = os.path.join(
        path, "Atlas_Houdini"
    )
    lqPath = os.path.join(
        path, "Atlas_Houdini"
    )
    patient = "model-patient"  # or sub-verse505
    classSettingsFile = os.path.join(
        path,
        "Atlas_Houdini/settings_dict_ideal_patient_corr_v1.json",
    )

    outpath = '/home/klanna/github/Spine3D/DeepSDF/tests/'
    dataset = RandomDeformationDataset(
        hqPath,
        lqPath,
        patient,
        classSettingsFile,
        hqSeg=False,
        lqSeg=False,
        handcraft_lq=True,
        mode="train",
        num_samples=3,
        spacing=(1, 1, 1),
    )

    print('Dataset size:', len(dataset))
    
    # hqTen0, *_ = dataset[0]

    for i in range(len(dataset)):
        hqTen, lqTen, coords, metadata = dataset[i]
        print(hqTen.shape)
        print(coords.reshape(-1, 3))
        # plot_examples(hqTen, i)
        # uncomment to save the images
        # save_nib(hqTen, f'{outpath}hqTen_{i}.nii.gz', metadata["affine"])
        # save_nib(lqTen, f'{outpath}lqTen_{i}.nii.gz', metadata["affine"])
