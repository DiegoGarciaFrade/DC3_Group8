import os
import cv2
import torch
import shutil
import logging
import numpy as np

from pathlib import Path
from tqdm.auto import tqdm
from plaspix import plaspix
from labelmate.evaluator import LabelPropEvaluator

logger = logging.getLogger(__name__)

class PLASPIXLabelProp():
    """Class to execute Point Label Aware Superpixels approach for a given dataset, 
    generate evaluation metrics based on ground truth and save experiment data.
    """
    def __init__(self, dataloader, execution_tag, hyper_params):
        # initialize input parameters
        self.dataloader = dataloader
        self.execution_tag = execution_tag
        self.hyper_params = hyper_params

        # derive few parameters from dataloader
        self.experiment_name = self.dataloader.experiment_name
        self.num_classes = self.dataloader.num_classes
        self.sub_folders = self.dataloader.sub_folders

        # initialize variables needed for various steps
        self.input_params = []
        self.processing_failures = []
        self.evaluator = \
            LabelPropEvaluator(
                experiment_name=self.experiment_name, 
                execution_tag=self.execution_tag, 
                num_classes=self.num_classes, 
                hyper_params=self.hyper_params, 
                )
    
    def validate_input_data(self):
        """Checks the validity of input data setup for label propagation
        """
        # check if input data (image, mask, point labels) has been setup correctly
        # check number of samples in images folder
        if len(os.listdir(self.sub_folders['images'])) != len(self.dataloader):
            raise Exception("Validation Error: Mismatch in number of images in dataset and input folder")
        
        # check synchronization between image folder and point labels folder
        if (
            len(set(os.listdir(self.sub_folders['images'])) - \
                set(os.listdir(self.sub_folders['masks']))) != 0 or \
            len(set(os.listdir(self.sub_folders['masks'])) - \
                set(os.listdir(self.sub_folders['images']))) != 0
            ):
            raise Exception("Validation Error: Mismatch in samples between images and masks input folders")
    
    def setup_input_params(self):
        """Builds input parameter list based on dataloader setup and hyper parameters
        """
        # setup input parameters list based on given hyper parameters
        self.input_params = [
            "-r", f"{self.sub_folders['images']}", # input images
            "-g", f"{self.sub_folders['labels']}", # input point labels in image format
            "-l", f"{self.sub_folders['predictions']}", # path for saving propagated dense masks
            "-c", f"{self.num_classes + 1}", # number of classes + 1 for unlabelled pixels
            "-u", f"{self.num_classes}", # label reserved for unlabelled pixels 
        ]

        # alpha (lambda) parameter that controls weightage for conflict loss
        if 'alpha' in self.hyper_params.keys():
            self.input_params.extend(["-a", f"{self.hyper_params['alpha']}"])
        
        # xy sigma parameter that controls weightage for x-y distance based measure
        if 'xysigma' in self.hyper_params.keys():
            self.input_params.extend(["-x", f"{self.hyper_params['xysigma']}"])

        # cnn sigma parameter that controls weightage for cnn features based mesaure
        if 'cnnsigma' in self.hyper_params.keys():
            self.input_params.extend(["-f", f"{self.hyper_params['cnnsigma']}"])

        # num_spixels parameter that controls the number of superpixels intialized
        if 'num_spixels' in self.hyper_params.keys():
            self.input_params.extend(["-s", f"{self.hyper_params['num_spixels']}"])

        # type of input for superpixels approach: point labels vs dense mask
        if self.hyper_params.get('input_type', 'SPARSE').upper() == 'SPARSE':
            self.input_params.append("--points")
        else:
            # do nothing as PLASPIX code by default runs for DENSE input
            pass
        
        # ensemble option for superpixels approach
        if self.hyper_params.get('ensemble', 'NO').upper() == 'YES':
            self.input_params.append("--ensemble")
        else:
            # do nothing as PLASPIX code by default runs without ensemble
            pass
        
        logger.debug(f"Input Parameters for PLASPIX: {self.input_params}")
    
    def propagate_labels(self):
        """Run Point Label Aware Superpixels module for given dataset and hyper parameters.
        """
        # run point label aware superpixels module for given dataset and hyper parameters
        logger.info(f"Initiating PLASPIX module execution ...")
        plaspix.propagate_labels(self.input_params)
        logger.info(f"Finished PLASPIX module execution")

        # check if all images have been processed successfully
        if len(os.listdir(self.sub_folders['predictions'])) != len(self.dataloader):
            self.processing_failures = \
                set(os.listdir(self.sub_folders['images'])) - \
                set(os.listdir(self.sub_folders['predictions']))
            logger.debug(f"Label propagation failures: {self.processing_failures}")
            raise Exception(f"Processing Error: Label propagation could not be performed for {len(self.processing_failures)} samples")
        else:
            self.processing_failures = []
            logger.info("Label propagation was successfull for all samples")
    
    def evaluate_output(self):
        """Compute evaluation metrics for the dataset
        """
        if len(self.processing_failures) == 0:
            # evaluate all samples
            _ = self.evaluator.evaluate_samples(self.dataloader)

            # compute dataset level metrics
            eval_results_summary = self.evaluator.generate_summary()
            logger.info(f"Evaluation Summary: {eval_results_summary}")
        else:
            error_message = (
                f"Validation Error: "
                f"Label propagation was not successful for {len(self.processing_failures)} samples. "
                f"Please fix the issues related to the failures before performing evaluation."
            )
            raise Exception(error_message)

    def save_experiment(self, save_path, sub_folders=['predictions']):
        """Saves the data related to current experiment for future reference
        """
        for sub_folder_name in sub_folders:
            # derive the destination path for the given sub folders
            sub_folder_path = Path.joinpath(save_path, sub_folder_name)

            logger.info(f"Saving {sub_folder_name} to {sub_folder_path} ...")

            # create the sub folder within the given save path
            self.dataloader.create_folder(sub_folder_path, sub_folder_name)

            # copy files related to each sample
            for index in range(len(self.dataloader)):
                # derive path for source file
                source_file_path = \
                    Path.joinpath(
                        self.sub_folders[sub_folder_name], 
                        f"{self.dataloader[index]['sample_id']}.png"
                        )

                # derive path for destination file
                if sub_folder_name == 'predictions':
                    target_file_path = \
                        Path.joinpath(
                            sub_folder_path, 
                            f"{self.dataloader[index]['sample_id']}-{self.execution_tag}.png"
                            )
                else:
                    target_file_path = \
                        Path.joinpath(
                            sub_folder_path, 
                            f"{self.dataloader[index]['sample_id']}.png"
                            )
                
                # copy from source to destination
                shutil.copyfile(source_file_path, target_file_path)
        
        # save evaluation metrics to given save path
        if self.evaluator.eval_results_samples.shape[0] > 0:
            self.evaluator.eval_results_samples.to_csv(
                Path.joinpath(save_path, f"{self.experiment_name}-Results-Samples-{self.execution_tag}.csv"), 
                index=False, 
                )
        if self.evaluator.eval_results_summary.shape[0] > 0:
            self.evaluator.eval_results_summary.to_csv(
                Path.joinpath(save_path, f"{self.experiment_name}-Results-Summary-{self.execution_tag}.csv"), 
                index=False, 
                )

        logger.info(f"Data related to the experiment were saved to {save_path}")

    def run_pipeline(self):
        """Runs all the steps required to perform label propagation and then runs evaluation
        """
        self.validate_input_data()
        self.setup_input_params()
        self.propagate_labels()
        self.evaluate_output()


class SAMPointPromptsLabelProp():
    """Class to execute Multiple Point-based Prompting of SAM and blending of masks 
    approach for a given dataset, generate evaluation metrics based on ground truth and 
    save experiment data.
    """
    def __init__(self, dataloader, execution_tag, hyper_params):
        # initialize input parameters
        self.dataloader = dataloader
        self.execution_tag = execution_tag
        self.hyper_params = hyper_params

        # derive few parameters from dataloader
        self.experiment_name = self.dataloader.experiment_name
        self.num_classes = self.dataloader.num_classes
        self.sub_folders = self.dataloader.sub_folders

        # initialize variables needed for SAM
        self.device = None
        self.sam_model = None
        self.sam_mask_predictor = None

        # initialize variables needed for evaluation
        self.evaluator = \
            LabelPropEvaluator(
                experiment_name=self.experiment_name, 
                execution_tag=self.execution_tag, 
                num_classes=self.num_classes, 
                hyper_params=self.hyper_params, 
                )
    
    def validate_input_data(self):
        """Checks the validity of input data setup for label propagation
        """
        # check if input data (image, mask, point labels) has been setup correctly
        # check number of samples in images folder
        if len(os.listdir(self.sub_folders['images'])) != len(self.dataloader):
            raise Exception("Validation Error: Mismatch in number of images in dataset and input folder")
        
        # check synchronization between image folder and point labels folder
        if (
            len(set(os.listdir(self.sub_folders['images'])) - \
                set(os.listdir(self.sub_folders['masks']))) != 0 or \
            len(set(os.listdir(self.sub_folders['masks'])) - \
                set(os.listdir(self.sub_folders['images']))) != 0
            ):
            raise Exception("Validation Error: Mismatch in samples between images and masks input folders")

    def load_sam_model(self):
        # setup device for loading SAM or SAM-HQ
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Determine which SAM model to load based on the weights file
        weights_file = self.hyper_params['image_encoder_weights_path']
    
        if 'sam_hq' in weights_file:
            # If the weights file is for SAM-HQ
            from segment_anything_hq import SamPredictor, sam_model_registry
        else:
            # Default to SAM
            from segment_anything import SamPredictor, sam_model_registry

        # load SAM model into selected device
        self.sam_model = \
            sam_model_registry[self.hyper_params['image_encoder']](
                checkpoint=self.hyper_params['image_encoder_weights_path'], 
                ).to(device=self.device)
        
        # initialize SAM predictor
        self.sam_mask_predictor = SamPredictor(self.sam_model)

    @staticmethod
    def build_point_prompts(point_labels):
        # initialize an empty list
        prompts = []

        # build prompt only for those class labels present in the data
        for positive_class_label in np.sort(point_labels.class_label.unique()):
            prompt_point_coords = []
            prompt_point_labels = []

            # build points based prompting parameters using positive and negative labels
            for index, point_label in point_labels.iterrows():
                prompt_point_coords.append([point_label['x'], point_label['y']])
                if point_label.class_label == positive_class_label:
                    prompt_point_labels.append(1)
                else:
                    prompt_point_labels.append(0)

            logger.debug(f"Positive Class Label: {positive_class_label}")
            logger.debug(f"Points Coordinates (x,y): {prompt_point_coords}")
            logger.debug(f"Point Labels: {prompt_point_labels}")

            # add positive class label and points details as a prompt
            prompt = dict(
                class_label=positive_class_label, 
                point_coords=prompt_point_coords, 
                point_labels=prompt_point_labels, 
                )
            prompts.append(prompt)
        
        return prompts
    
    @staticmethod
    def prompt_sam_with_multiple_point_prompts(mask_predictor, prompts):
        for index in range(len(prompts)):
            sam_masks = sam_scores = sam_logits = None

            sam_masks, sam_scores, sam_logits = \
                mask_predictor.predict(
                    point_coords=np.array(prompts[index]['point_coords']),
                    point_labels=np.array(prompts[index]['point_labels']),
                    multimask_output=False,
                    )

            prompts[index]['mask'] = sam_masks[0]
            prompts[index]['score'] = sam_scores[0]
            prompts[index]['logits'] = sam_logits[0]
            prompts[index]['logits_min'] = \
                np.min(sam_logits[0]) \
                if np.min(sam_logits[0]) < mask_predictor.model.mask_threshold \
                else (mask_predictor.model.mask_threshold - 0.1)

        return prompts.copy()

    @staticmethod
    def blend_sam_outputs(sam_model, sam_mask_predictor, sam_results):
        # stack the logits from each prompting exercise together
        sam_logits_stacked = np.array([sam_result['logits'] for sam_result in sam_results])

        # find the indices (prompts) that have maximum logit values
        sam_logits_arg_max = np.argmax(sam_logits_stacked, axis=0)

        # blend logits by suppressing the logits values for prompts that do not have max logit values  
        for index in range(len(sam_results)):
            logits_blended = sam_results[index]['logits'][:,:]
            suppress_indices = np.where(sam_logits_arg_max != index, True, False)
            logits_blended[suppress_indices] = sam_results[index]['logits_min']
            sam_results[index]['logits_blended'] = logits_blended[:,:]
        
        # generate masks from blended logits
        for index in range(len(sam_results)):
            post_processed_mask = \
                sam_model.postprocess_masks(
                    masks=torch.from_numpy(sam_results[index]['logits_blended']).unsqueeze(dim=0).unsqueeze(dim=0),
                    input_size=sam_mask_predictor.input_size,
                    original_size=sam_mask_predictor.original_size,
                    )

            sam_results[index]['mask_blended'] = \
                post_processed_mask.squeeze().numpy() > sam_mask_predictor.model.mask_threshold

        # stack the blended masks from each prompting exercise together
        # but, multiply by class label so that pixel value points to respective class labels
        sam_masks_stacked = \
            np.array(
                [sam_result['mask_blended'] * sam_result['class_label'] 
                for sam_result in sam_results], 
                )
        
        # generate a single blended mask by adding up pixel level values
        # which should ideally be non-overlapping at this stage
        mask_blended = np.sum(sam_masks_stacked, axis=0)

        return mask_blended        

    def process_sample(self, idx):
        # read image
        image = cv2.imread(str(self.dataloader[idx]['image_path'].resolve()))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = self.dataloader.read_image(self.dataloader[idx]['image_path'])

        # define multiple point based prompts based on available class labels
        prompts = self.build_point_prompts(point_labels=self.dataloader[idx]['point_labels'])

        # prompt SAM if any prompts are available
        if len(prompts) > 0:
            # assign image to SAM mask predictor
            self.sam_mask_predictor.set_image(image)

            # prompt SAM multiple times based on prompts dictionary
            sam_results = \
                self.prompt_sam_with_multiple_point_prompts(
                    mask_predictor=self.sam_mask_predictor, 
                    prompts=prompts.copy(), 
                    )

            # blend logits output from SAM
            predicted_mask = \
                self.blend_sam_outputs(
                    sam_model=self.sam_model, 
                    sam_mask_predictor=self.sam_mask_predictor, 
                    sam_results=sam_results, 
                    )        
        else:
            # set prediction to zeros if no prompt is available
            predicted_mask = np.zeros(image.shape[:2])

        # save predicted mask as an image
        prediction_write_status = \
            cv2.imwrite(
                str(self.dataloader[idx]['prediction_path'].resolve()), 
                predicted_mask, 
                )

    def propagate_labels(self):
        """Run Multiple Point-based Prompting of SAM approach for given dataset and hyper parameters.
        """
        # process each sample one by one
        for index in tqdm(range(len(self.dataloader)), total=len(self.dataloader)):
            self.process_sample(idx=index)

        # check if all images have been processed successfully
        if len(os.listdir(self.sub_folders['predictions'])) != len(self.dataloader):
            self.processing_failures = \
                set(os.listdir(self.sub_folders['images'])) - \
                set(os.listdir(self.sub_folders['predictions']))
            logger.debug(f"Label propagation failures: {self.processing_failures}")
            raise Exception(f"Processing Error: Label propagation could not be performed for {len(self.processing_failures)} samples")
        else:
            self.processing_failures = []
            logger.info("Label propagation was successfull for all samples")
    
    def evaluate_output(self):
        """Compute evaluation metrics for the dataset
        """
        if len(self.processing_failures) == 0:
            # evaluate all samples
            _ = self.evaluator.evaluate_samples(self.dataloader)

            # compute dataset level metrics
            eval_results_summary = self.evaluator.generate_summary()
            logger.info(f"Evaluation Summary: {eval_results_summary}")
        else:
            error_message = (
                f"Validation Error: "
                f"Label propagation was not successful for {len(self.processing_failures)} samples. "
                f"Please fix the issues related to the failures before performing evaluation."
            )
            raise Exception(error_message)

    def save_experiment(self, save_path, sub_folders=['predictions']):
        """Saves the data related to current experiment for future reference
        """
        for sub_folder_name in sub_folders:
            # derive the destination path for the given sub folders
            sub_folder_path = Path.joinpath(save_path, sub_folder_name)

            logger.info(f"Saving {sub_folder_name} to {sub_folder_path} ...")

            # create the sub folder within the given save path
            self.dataloader.create_folder(sub_folder_path, sub_folder_name)

            # copy files related to each sample
            for index in range(len(self.dataloader)):
                # derive path for source file
                source_file_path = \
                    Path.joinpath(
                        self.sub_folders[sub_folder_name], 
                        f"{self.dataloader[index]['sample_id']}.png"
                        )

                # derive path for destination file
                if sub_folder_name == 'predictions':
                    target_file_path = \
                        Path.joinpath(
                            sub_folder_path, 
                            f"{self.dataloader[index]['sample_id']}-{self.execution_tag}.png"
                            )
                else:
                    target_file_path = \
                        Path.joinpath(
                            sub_folder_path, 
                            f"{self.dataloader[index]['sample_id']}.png"
                            )
                
                # copy from source to destination
                shutil.copyfile(source_file_path, target_file_path)
        
        # save evaluation metrics to given save path
        if self.evaluator.eval_results_samples.shape[0] > 0:
            self.evaluator.eval_results_samples.to_csv(
                Path.joinpath(save_path, f"{self.experiment_name}-Results-Samples-{self.execution_tag}.csv"), 
                index=False, 
                )
        if self.evaluator.eval_results_summary.shape[0] > 0:
            self.evaluator.eval_results_summary.to_csv(
                Path.joinpath(save_path, f"{self.experiment_name}-Results-Summary-{self.execution_tag}.csv"), 
                index=False, 
                )

        logger.info(f"Data related to the experiment were saved to {save_path}")

    def run_pipeline(self):
        """Runs all the steps required to perform label propagation and then runs evaluation
        """
        self.validate_input_data()
        self.load_sam_model()
        self.propagate_labels()
        self.evaluate_output()
