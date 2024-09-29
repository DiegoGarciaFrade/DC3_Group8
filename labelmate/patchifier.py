import os
import cv2
import shutil
import logging
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

class SimplePatchifier():
    def __init__(
            self, 
            experiment_name, 
            samples, 
            num_classes, 
            working_folder, 
            patch_height=256,
            patch_width=256,
            step_size=256
            ):
        self.experiment_name = experiment_name
        self.samples = samples
        self.num_classes = num_classes
        self.working_folder = working_folder

        self.patch_height = patch_height
        self.patch_width = patch_width
        self.step_size = step_size

        self.dataset = \
            pd.DataFrame(
                columns=[
                    'quadratid', 'sample_id', 'patch_id', 'patch_origin_h', 'patch_origin_w', 
                    'point_labels_count', 
                    ], 
                )
        self.point_labels = \
            pd.DataFrame(
                columns=['quadratid', 'sample_id', 'patch_id', 'y', 'x', 'class_name', 'class_label'], 
                )
            
        self.sub_folders = \
            dict(
                images=Path.joinpath(working_folder, 'images'), 
                labels=Path.joinpath(working_folder, 'labels'), 
                masks=Path.joinpath(working_folder, 'masks'), 
                predictions=Path.joinpath(working_folder, 'predictions'), 
            )

    @staticmethod
    def create_folder(folder, purpose):
        """Creates a folder if it does not already exist.
        """
        if os.path.exists(folder):
            logger.debug(f"Folder already exists for {purpose} at: {folder}")
        else:
            os.makedirs(folder)
            logger.debug(f"Created folder for {purpose} at: {folder}")

    @staticmethod
    def delete_folder(folder, purpose):
        """Deletes a folder and its contents recursively.
        """
        if os.path.exists(folder):
            shutil.rmtree(folder)
            logger.debug(f"Deleted folder for {purpose} at: {folder}")
        else:
            logger.debug(f"No folder exists for {purpose} at: {folder}")

    def create_sub_folders(self):
        for folder_purpose, sub_folder in self.sub_folders.items():
            self.create_folder(sub_folder, folder_purpose)
    
    def delete_sub_folders(self):
        for folder_purpose, sub_folder in self.sub_folders.items():
            self.delete_folder(sub_folder, folder_purpose)

    def read_image(self, image_path):
        """Read an image from specified path in 3-channel RGB format
        """
        image = cv2.imread(str(image_path.resolve()))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def read_mask(self, mask_path):
        """Read a single channel mask from specified path
        """
        mask = cv2.imread(str(mask_path.resolve()), cv2.IMREAD_UNCHANGED)
        return mask

    def get_patches_origins(self, image_shape):
        """Computes the origins of all the different patches that are possible for given parameters
        """
        image_height = image_shape[0]
        image_width = image_shape[1]

        # patch origins start from top left hand corner of image in (H, W, C) format
        # height index maps to row of image array; width index maps to column of image array
        # number of patches along height depends on image height and step size
        # number of patches along width depends on image width and step size
        # last patch along height or width may have non-standard dimensions if not divisible by step size
        patches_h_indices = [i * self.step_size for i in range(int(np.ceil(image_height / self.step_size)))]
        patches_w_indices = [i * self.step_size for i in range(int(np.ceil(image_width / self.step_size)))]

        # create a meshgrid to get all origins for all the different patches that are possible 
        patches_origins_meshgrid = np.meshgrid(patches_h_indices, patches_w_indices, indexing='ij')

        # format the meshgrid to a list of dictionary for ease of use
        patches_origins = [
            dict(h=coords[0], w=coords[1])
            for coords in zip(patches_origins_meshgrid[0].reshape(-1), patches_origins_meshgrid[1].reshape(-1))
            ]

        return patches_origins

    def patchify_sample(self, sample):
        """Extracts and saves image and mask patches for given sample and subsets point labels
        in a manner consistent with patch dimensions.
        """
        # load image and mask for given sample
        image = self.read_image(sample['image_path'])
        mask = self.read_mask(sample['mask_path'])

        # identify the origins for all the patches to be created from the sample
        patches_origins = self.get_patches_origins(image.shape)

        # extract patches from sample and save them
        patch_id = 0
        for patch_origin in patches_origins:
            # naming convention for patchified quadrats
            quadratid_new = f"{sample['sample_id']}-P{patch_id}"

            # extract image patch from given originand save in images sub folder
            image_patch = \
                image[
                    patch_origin['h'] : patch_origin['h'] + self.patch_height,
                    patch_origin['w'] : patch_origin['w'] + self.patch_width,
                    :
                    ]
            image_patch_write_path = Path.joinpath(self.sub_folders['images'], f"{quadratid_new}.png")
            image_patch_write_status = \
                cv2.imwrite(
                    str(image_patch_write_path.resolve()),
                    cv2.cvtColor(image_patch, cv2.COLOR_RGB2BGR),
                    )

            # extract mask patch from given origin and save in masks sub folder
            mask_patch = \
                mask[
                    patch_origin['h'] : patch_origin['h'] + self.patch_height,
                    patch_origin['w'] : patch_origin['w'] + self.patch_width,
                    ]
            mask_patch_write_path = Path.joinpath(self.sub_folders['masks'], f"{quadratid_new}.png")
            mask_patch_write_status = cv2.imwrite(str(mask_patch_write_path.resolve()), mask_patch)

            # extract point labels for patch
            point_labels_patch = \
                sample['point_labels']\
                    .query(f"quadratid == '{sample['sample_id']}'")\
                    .query(f"y >= {patch_origin['h']}")\
                    .query(f"y < {patch_origin['h'] + self.patch_height}")\
                    .query(f"x >= {patch_origin['w']}")\
                    .query(f"x < {patch_origin['w'] + self.patch_width}")\
                    [['quadratid', 'y', 'x', 'class_name', 'class_label']]\
                    .copy()

            # revise point label coordinates to be consistent with newly extracted patch
            point_labels_patch['y'] = point_labels_patch['y'] - patch_origin['h']
            point_labels_patch['x'] = point_labels_patch['x'] - patch_origin['w']

            # set sample id, patch id and new quadrat id for the newlu extracted patch
            point_labels_patch['sample_id'] = sample['sample_id']
            point_labels_patch['patch_id'] = patch_id
            point_labels_patch['quadratid'] = quadratid_new

            # create a patch sample dictionary
            patch_sample = \
                dict(
                    quadratid=quadratid_new, 
                    sample_id=sample['sample_id'], 
                    patch_id=patch_id, 
                    patch_origin_h=patch_origin['h'], 
                    patch_origin_w=patch_origin['w'], 
                    point_labels_count=point_labels_patch.shape[0], 
                    )

            # append to class level variables
            self.point_labels = pd.concat([self.point_labels, point_labels_patch], ignore_index=True)
            self.dataset = pd.concat([self.dataset, pd.DataFrame([patch_sample])], ignore_index=True)

            patch_id += 1

    def patchify_samples(self):
        """Runs patchifier for each sample one by one
        """
        # patchify each sample one by one
        for index in tqdm(range(len(self.samples)), total=len(self.samples)):
            self.patchify_sample(sample=self.samples[index])

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        experiment_name = self.experiment_name
        sample_id = self.dataset.quadratid[idx]
        image_path = Path.joinpath(self.sub_folders['images'], f"{sample_id}.png")
        mask_path = Path.joinpath(self.sub_folders['masks'], f"{sample_id}.png")
        prediction_path = Path.joinpath(self.sub_folders['predictions'], f"{sample_id}.png")
        sample_point_labels = self.point_labels.query(f"quadratid == '{sample_id}'").copy()

        sample = \
            dict(
                experiment_name=experiment_name, 
                sample_id=sample_id, 
                image_path=image_path, 
                mask_path=mask_path, 
                prediction_path=prediction_path, 
                point_labels=sample_point_labels,
            )
        return sample