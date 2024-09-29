import cv2
import torch
import logging
import numpy as np
import pandas as pd
import torchmetrics
import seaborn as sns
import matplotlib.pyplot as plt

from labelmate import encode_mask_with_contours, CLASS_NAME_MAPPING

logger = logging.getLogger(__name__)

class LabelPropEvaluator():
    """Class for computing evaluation metrics for label propagation
    """
    def __init__(self, experiment_name, execution_tag='C0', num_classes=3, hyper_params={}):
        # set experiment name and a name for identifying different runs with different parameters
        self.experiment_name = experiment_name
        self.execution_tag = execution_tag
        self.hyper_params = hyper_params

        # set the number of classes
        self.num_classes = num_classes

        # define the list of evaluation metrics that will be computed
        self.eval_functions = \
            dict(
                pa=torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes, ignore_index=self.num_classes), 
                mpa=torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes, ignore_index=self.num_classes, average='macro'), 
                pa_class=torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes, ignore_index=self.num_classes, average='none'), 
                miou=torchmetrics.JaccardIndex(task="multiclass", num_classes=self.num_classes, ignore_index=self.num_classes, average='macro'), 
                iou_class=torchmetrics.JaccardIndex(task="multiclass", num_classes=self.num_classes, ignore_index=self.num_classes, average='none'), 
                mdice=torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes, average='macro'), 
                dice_class=torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes, average='none'), 
                confusion_matrix=torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_classes, ignore_index=self.num_classes, normalize='true'), 
            )
        
        # define data structure to store sample level results and summary of evaluation metrics
        self.sample_level_columns = ['experiment_name', 'execution_tag', 'sample_id', *self.hyper_params.keys()]
        self.summary_level_columns = ['experiment_name', 'execution_tag', *self.hyper_params.keys()]
        for key in self.eval_functions.keys():
            if '_class' in key:
                class_wise_columns = [f"{key}_{class_id}" for class_id in range(self.num_classes)]
                self.sample_level_columns.extend(class_wise_columns)
                self.summary_level_columns.extend(class_wise_columns)
            else:
                self.sample_level_columns.append(key)
                self.summary_level_columns.append(key)

        self.eval_results_samples = pd.DataFrame(columns=self.sample_level_columns)
        self.eval_results_summary = pd.DataFrame(columns=self.summary_level_columns)
        self.confusion_matrix_summary = None
    
    def evaluate_sample(self, sample_id, mask_path, prediction_path):
        """Compute evaluation metrics for a single sample
        """
        # read ground truth mask from specified path
        mask = cv2.imread(str(mask_path.resolve()))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        # change from RGB color coding to label encoding format if necessary
        # check if the contents of mask array are different from set of allowed labels
        if len(set(mask.reshape(-1)) - set([class_id for class_id in range(self.num_classes)])) > 0:
            mask_encoded = encode_mask_with_contours(mask=mask)
        else:
            mask_encoded = mask
        
        # read prediction from specified path
        prediction = cv2.imread(str(prediction_path.resolve()))
        prediction = cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB)

        # convert numpy arrays to torch tensors 
        # use only the first channel as labels are duplicated across RGB channels
        mask_torch = torch.from_numpy(mask_encoded[:,:,0])
        prediction_torch = torch.from_numpy(prediction[:,:,0])

        # initialize a dictionary to store the results
        eval_results = \
            dict(
                experiment_name=self.experiment_name, 
                execution_tag=self.execution_tag, 
                sample_id=sample_id, 
                **self.hyper_params, 
            )

        # compute evaluation metrics for given sample
        # workaround for PLASPIX returning extra labels in some cases
        if np.max(prediction) < 3:
            for metric_name in self.eval_functions.keys():
                eval_result = self.eval_functions[metric_name](prediction_torch, mask_torch)
                if '_class' in metric_name:
                    for class_id in range(self.num_classes):
                        eval_results[f"{metric_name}_{class_id}"] = eval_result[class_id].item()
                elif metric_name == 'confusion_matrix':
                    eval_results[metric_name] = [eval_result.numpy()]
                else:
                    eval_results[metric_name] = eval_result.item()
        
        # store evaluation results for this sample to sample level data frame
        self.eval_results_samples = pd.concat([self.eval_results_samples, pd.DataFrame([eval_results])])

        return eval_results

    def evaluate_samples(self, samples):
        """Compute evaluation metrics for given set of samples
        """
        eval_results_list = []
        # evaluate each sample one by one
        for index in range(len(samples)):
            eval_results = \
                self.evaluate_sample(
                    sample_id=samples[index].get('sample_id', None), 
                    mask_path=samples[index].get('mask_path', None), 
                    prediction_path=samples[index].get('prediction_path', None), 
                )
            eval_results_list.append(eval_results)
        
        return eval_results_list

    def generate_summary(self):
        """Compute dataset level summary for all samples that were evaluated prior to this call
        """
        # initialize a dictionary to store the results
        eval_summary = \
            dict(
                experiment_name=self.experiment_name, 
                execution_tag=self.execution_tag, 
                **self.hyper_params, 
            )
        
        # summarize each evaluation metric and store in a dictionary
        for metric_name in self.eval_functions.keys():
            eval_result = self.eval_functions[metric_name].compute()
            if '_class' in metric_name:
                for class_id in range(self.num_classes):
                    eval_summary[f"{metric_name}_{class_id}"] = eval_result[class_id].item()
            elif metric_name == 'confusion_matrix':
                eval_summary[metric_name] = [eval_result.numpy()]
                self.confusion_matrix_summary = \
                    pd.DataFrame(
                        eval_result.numpy(), 
                        index=CLASS_NAME_MAPPING.values(), 
                        columns=CLASS_NAME_MAPPING.values(), 
                        )
            else:
                eval_summary[metric_name] = eval_result.item()
        
        # save evaluation summary to summary data frame
        self.eval_results_summary = pd.concat([self.eval_results_summary, pd.DataFrame([eval_summary])])

        return eval_summary
    
    def plot_confusion_matrix(self):
        """Plots the overall confusion matrix for all samples in the form of a heat map
        """
        if self.confusion_matrix_summary is not None:
            fig, ax = plt.subplots(figsize=(4,4))
            sns.heatmap(
                self.confusion_matrix_summary,
                annot=True,
                fmt='.2%',
                square=True,
                cmap='Blues',
                ax=ax, 
                )
            plt.xlabel("Predictions")
            plt.ylabel("Ground Truth")
            plt.xticks(rotation=45)
            plt.yticks(rotation=45)
            plt.show()