import os
import cv2
import shutil
import logging
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm.auto import tqdm
from scipy.sparse import coo_array
from labelmate import encode_mask_with_contours, CLASS_NAME_MAPPING

logger = logging.getLogger(__name__)

class LabelPropDataLoader():
    """Base class for setting up label propagation pipeline for the given samples.
    """
    def __init__(
        self, 
        experiment_name, 
        dataset, 
        point_labels, 
        num_classes, 
        working_folder, 
        transforms=None, 
        execution_tag=None, 
        random_seed=42
        ):
        """Initializes label propagation setup.
        """
        self.experiment_name = experiment_name
        self.dataset = dataset
        if point_labels is not None:
            if point_labels.shape[0] > 0:
                self.point_labels = point_labels
            else:
                self.point_labels = pd.DataFrame(
                    columns=['quadratid', 'x', 'y', 'class_name', 'class_label']
                    )
        else:
            self.point_labels = pd.DataFrame(
                columns=['quadratid', 'x', 'y', 'class_name', 'class_label']
                )
        self.num_classes = num_classes
        self.working_folder = working_folder
        self.transforms=transforms
        self.execution_tag=execution_tag
        self.random_seed = random_seed

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
    
    @staticmethod
    def get_random_point_labels(mask_path, num_point_labels, class_name_mapping=CLASS_NAME_MAPPING, random_seed=42):
        # read dense mask that is in label encoded format from masks sub folder
        # extract only the first channel as all 3 channels hold identical information
        # flatten single channel mask
        mask = cv2.imread(mask_path)
        mask_single_channel = mask[:,:,0]
        mask_flattened = mask_single_channel.reshape(-1)

        # create meshgrid of all possible coordinates based on image height and width
        mask_h, mask_w = mask_single_channel.shape
        coords_meshgrid = np.meshgrid(np.arange(mask_h), np.arange(mask_w), indexing='ij')
        coords_flattened = [
            dict(y=coords[0], x=coords[1]) 
            for coords in zip(coords_meshgrid[0].reshape(-1), coords_meshgrid[1].reshape(-1))
            ]
        
        # sample required number of point label indices based on flattened mask shape
        np.random.seed(random_seed)
        point_label_indices = np.random.randint(low=0, high=mask_flattened.shape[0], size=num_point_labels)

        # select data from coordinates meshgrid and flattened mask using sampled indices
        # combine them together to create a list of dictionaries
        sampled_point_labels = [
            {**z[0], **z[1]} 
            for z in \
                zip(
                    np.array(coords_flattened)[point_label_indices], 
                    [
                        dict(class_name=class_name_mapping[pixel], class_label=pixel) 
                        for pixel in mask_flattened[point_label_indices]
                        ]
                    )
            ]

        return sampled_point_labels
    
    @staticmethod
    def get_evenly_spaced_point_labels(mask_path, num_point_labels, class_name_mapping=CLASS_NAME_MAPPING):
        # read dense mask that is in label encoded format from masks sub folder
        # extract only the first channel as all 3 channels hold identical information
        # flatten single channel mask
        mask = cv2.imread(mask_path)
        mask_single_channel = mask[:,:,0]

        # create a uniformly spaced set of points based on image height and width
        # get mask height and width
        mask_h, mask_w = mask_single_channel.shape

        # find number of equally spaced points along height and width
        # it can be found by solving two equations
        # x = w_h_ratio * y         ------- equation 1
        # x * y = num_point_labels  ------- equation 2
        # solution for above equations are as follows:
        # x = sqrt(N * w_h_ratio)
        # y = sqrt(N / w_h_ratio)

        w_h_ratio = mask_w / mask_h
        num_points_along_width = (num_point_labels * w_h_ratio)**(0.5)
        num_points_along_height = (num_point_labels / w_h_ratio)**(0.5)
        points_along_width = \
            np.linspace(
                start=0, 
                stop=mask_w-1, 
                num=int(num_points_along_width), 
                endpoint=True, 
                dtype=np.uint16, 
                )
        points_along_height = \
            np.linspace(
                start=0, 
                stop=mask_h-1, 
                num=int(num_points_along_height), 
                endpoint=True, 
                dtype=np.uint16, 
                )
        grid_points_h, grid_points_w = np.meshgrid(points_along_height, points_along_width, indexing='ij')

        # get point labels from mask for selected point coordinates
        sampled_point_labels = \
            [dict(y=z[0], x=z[1], class_name=class_name_mapping[z[2]], class_label=z[2])
            for z in zip(
                grid_points_h.reshape(-1), 
                grid_points_w.reshape(-1), 
                mask_single_channel[grid_points_h.reshape(-1), grid_points_w.reshape(-1)], 
                )
                ]

        return sampled_point_labels

    def load_image(self, sample_id, image_path):
        # read image from specified path
        image = cv2.imread(image_path)

        # apply transforms to image if available
        if self.transforms is not None:
            transformed = self.transforms(image=image)
            image = transformed['image']

        image_write_path = Path.joinpath(self.sub_folders['images'], f"{sample_id}.png") 
        image_write_status = cv2.imwrite(str(image_write_path.resolve()), image)
        return image_write_path, image_write_status, image.shape

    def load_mask(self, sample_id, mask_path, dataset_name):
        # TODO: handle mask encoding for all datasets
        # read ground truth mask from specified path
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # change from RGB color coding to label encoding format
        # check if the contents of mask array are different from set of allowed labels
        if len(set(mask.reshape(-1)) - set([class_id for class_id in range(self.num_classes)])) > 0:
            mask_encoded = encode_mask_with_contours(mask)
        else:
            mask_encoded = mask

        # apply transforms to mask if available
        if self.transforms is not None:
            transformed = self.transforms(image=np.zeros(mask.shape), mask=mask_encoded)
            mask_encoded = transformed['mask']

        # save mask without any color channel i.e. (H,W) instead of (H,W,C)
        mask_write_path = Path.joinpath(self.sub_folders['masks'], f"{sample_id}.png") 
        mask_write_status = cv2.imwrite(str(mask_write_path.resolve()), mask_encoded[:,:,0])
        return mask_write_path, mask_write_status

    def load_labels(self, sample_id, image_shape):
        # read point labels data, create sparse array and then
        # convert to dense array with the same shape as original image
        sample_point_labels = self.point_labels.query(f"quadratid == '{sample_id}'")[['x', 'y', 'class_label']].copy()
        row = sample_point_labels.y.values
        col = sample_point_labels.x.values
        class_labels = sample_point_labels.class_label + 1 # add 1 so that unlabelled is 0
        labels_dense = coo_array((class_labels, (row, col)), shape=image_shape[0:2]).toarray()
        # point label aware superpixels expects point labels to be of shape H, W without color channels
        labels_write_path = Path.joinpath(self.sub_folders['labels'], f"{sample_id}.png")
        labels_write_status = cv2.imwrite(str(labels_write_path.resolve()), labels_dense)
        return labels_write_path, labels_write_status

    def prepare_input_data(self):
        # TODO: handle exceptions
        for index, sample in tqdm(self.dataset.iterrows(), total=self.dataset.shape[0]):
            logger.debug(f"Processing sample: {sample.sample_id}")
            # load image into working folder
            image_path, image_write_status, image_shape = self.load_image(sample.sample_id, sample.image_path)
            
            # load mask into working folder
            mask_path, mask_write_status = self.load_mask(sample.sample_id, sample.mask_path, sample.dataset_name)
            
            # load point labels in image format (H,W,C) into working folder
            if sample.point_labels_source.upper() == 'SPARSE':
                labels_path, labels_write_status = self.load_labels(sample.sample_id, image_shape)
            else:
                # point labels are not provided in point labels data frame
                # it could be because there was no random point annotation done for those images
                # or, a different number of point labels are needed than that is actually available
                # so, it is upto this dataloader to sample point labels from dense mask
                
                # delete rows from point labels data frame for given sample id
                self.point_labels.drop(
                    self.point_labels[self.point_labels.quadratid == sample.sample_id].index, 
                    inplace=True, 
                    )

                # sample required number of point labels from dense mask
                if sample.point_labels_source.upper() == 'GRID':
                    sampled_point_labels = \
                        self.get_evenly_spaced_point_labels(
                            mask_path=os.path.join(self.sub_folders['masks'], f"{sample.sample_id}.png"), 
                            num_point_labels=sample.point_labels_count, 
                            class_name_mapping=CLASS_NAME_MAPPING, 
                            )
                else:
                    sampled_point_labels = \
                        self.get_random_point_labels(
                            mask_path=os.path.join(self.sub_folders['masks'], f"{sample.sample_id}.png"), 
                            num_point_labels=sample.point_labels_count, 
                            class_name_mapping=CLASS_NAME_MAPPING, 
                            random_seed=\
                                sample.manifest_index \
                                if 'manifest_index' in sample \
                                else self.random_seed, 
                            )

                # convert the selected point labels to a data frame
                sampled_point_labels_df = pd.DataFrame(sampled_point_labels)
                sampled_point_labels_df['quadratid'] = sample.sample_id

                # append the selected point labels to main point labels data frame
                self.point_labels = \
                    pd.concat(
                        [self.point_labels, sampled_point_labels_df], 
                        ignore_index=True
                        )

                # change data types of key columns to requried type so that sparse array function works
                self.point_labels = \
                    self.point_labels.astype(
                        {
                            'quadratid': str, 
                            'y': np.uint16, 
                            'x': np.uint16, 
                            'class_label': np.uint16, 
                        }
                    )
                
                # load point labels in image format (H,W,C) into working folder
                labels_path, labels_write_status = self.load_labels(sample.sample_id, image_shape)
    
    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        experiment_name = self.experiment_name
        # TODO: add dataset name
        # dataset_name=self.dataset.dataset_name[idx]
        sample_id = self.dataset.sample_id[idx]

        image_path = Path.joinpath(self.sub_folders['images'], f"{sample_id}.png")
        if not os.path.exists(image_path):
            image_path = Path(self.dataset.image_path[idx])
        
        mask_path = Path.joinpath(self.sub_folders['masks'], f"{sample_id}.png")
        if not os.path.exists(mask_path):
            mask_path = Path(self.dataset.mask_path[idx])   
        
        if self.execution_tag is not None:
            prediction_path = \
                Path.joinpath(
                    self.sub_folders['predictions'], 
                    f"{sample_id}-{self.execution_tag}.png"
                    )
        else:
            prediction_path = Path.joinpath(self.sub_folders['predictions'], f"{sample_id}.png")

        if self.point_labels is not None:
            sample_point_labels = self.point_labels.query(f"quadratid == '{sample_id}'").copy()
        else:
            sample_point_labels = None

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