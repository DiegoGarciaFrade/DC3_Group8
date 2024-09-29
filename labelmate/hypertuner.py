import os
import logging
import pandas as pd

from pathlib import Path
from itertools import product
from labelmate.propagator import PLASPIXLabelProp

logger = logging.getLogger(__name__)

class PLASPIXLabelPropGridSearch():
    """Class to manage grid search to find best hyper parameters.
    """
    def __init__(self, dataloader, hyper_params_ranges, save_path, restart_index=1, save_interval=50):
        # initialize input parameters
        self.dataloader = dataloader
        self.hyper_params_ranges = hyper_params_ranges
        self.save_path = save_path

        # derive few parameters from dataloader
        self.experiment_name = self.dataloader.experiment_name
        self.num_classes = self.dataloader.num_classes
        self.sub_folders = self.dataloader.sub_folders

        # set restart index and save interval
        self.restart_index = restart_index
        self.save_interval = save_interval

        # intialize search results variables        
        self.grid_search_results_samples = pd.DataFrame({})
        self.grid_search_results_summary = pd.DataFrame({})
        self.results_samples_file_path = \
            Path.joinpath(
                save_path, 
                f"{self.experiment_name}-Grid-Search-Samples.csv", 
            )
        self.results_summary_file_path = \
            Path.joinpath(
                save_path, 
                f"{self.experiment_name}-Grid-Search-Summary.csv", 
            )

        # re-initialize the results variables from csv files if this is a restart
        if self.restart_index > 1:
            if os.path.exists(self.results_samples_file_path):
                self.grid_search_results_samples = pd.read_csv(self.results_samples_file_path)
            if os.path.exists(self.results_summary_file_path):
                self.grid_search_results_summary = pd.read_csv(self.results_summary_file_path)

    def hyper_params_combinations(self):
        """Generator object for hyper parameters grid based on specified ranges or values.
        """
        for values_combo in product(*self.hyper_params_ranges.values()):
            yield dict(zip(self.hyper_params_ranges.keys(), values_combo))
    
    def process_single_combination(self, combo_id, hyper_params):
        """Perform all operations to run Point Label Aware Superpixel for a single 
        combination of hyper parameters, get evaluation metrics and save the data.
        """
        # delete predictions sub folder to remove predictions for previous combinations
        self.dataloader.delete_folder(self.sub_folders['predictions'], 'predictions')

        # create predictions sub folder
        self.dataloader.create_folder(self.sub_folders['predictions'], 'predictions')

        # instantiate PLASPIX label propagator object
        label_propagator = \
            PLASPIXLabelProp(
                dataloader=self.dataloader, 
                execution_tag=f"C{combo_id}", 
                hyper_params=hyper_params, 
                )
        
        # run label propagation
        label_propagator.run_pipeline()

        # save experiment data
        label_propagator.save_experiment(
            save_path=self.save_path, 
            sub_folders= self.sub_folders.keys() if combo_id == 1 else ['predictions'],  
            )
        
        # append results to grid search results variables
        self.grid_search_results_samples = \
            pd.concat(
                [self.grid_search_results_samples, label_propagator.evaluator.eval_results_samples], 
                ignore_index=True, 
                )
        self.grid_search_results_summary = \
            pd.concat(
                [self.grid_search_results_summary, label_propagator.evaluator.eval_results_summary], 
                ignore_index=True, 
                )

    def process_all_combinations(self):
        """Run label propagation for each combination of hyper parameters, capture evaluation metrics
        and store relevant experiment data.
        """
        # set variables to track number of combinations so that restart can be done at any index
        combo_id = 1

        # loop through hyper parameter combinations generator object
        for hyper_params in self.hyper_params_combinations():
            # check against restart index to decide if a combo needs to be processed or skipped
            if combo_id >= self.restart_index:
                self.process_single_combination(
                    combo_id=combo_id, 
                    hyper_params=hyper_params, 
                    )
                
                # periodically save the results to save path
                if combo_id % self.save_interval == 0:
                    self.grid_search_results_samples.to_csv(self.results_samples_file_path, index=False)
                    self.grid_search_results_summary.to_csv(self.results_summary_file_path, index=False)
            else:
                # skip processing until combo id matches restart index
                pass
            
            # increment combo id
            combo_id += 1
        
        # save results one last time now that all combinations are processed
        self.grid_search_results_samples.to_csv(self.results_samples_file_path, index=False)
        self.grid_search_results_summary.to_csv(self.results_summary_file_path, index=False)