import argparse
import json
import torch
from multiprocessing import Pool

import losses

from cfg.load_config import load_config
from data_iteration.data_iterators import RunIteratorSettings, RunIterator, ExperimentSettings
from surnet.utils.logger import log
from surnet.utils.model_utils import build_model


class ExperimentMain:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-cfg-env', help='Which config to load', type=str)
        parser.add_argument('-n-cpus', help='Number of CPUs', type=int)
        parser.add_argument('-dev', help='Development environment', action='store_true')
        parser.add_argument('-params-path', help='Path to the parameters json', type=str)
        parser.add_argument('-job', help='Job number', type=str)
        self.args = parser.parse_args()

        self.cfg = load_config(env=self.args.cfg_env)

        import os
        print(os.path.realpath('.'))
        with open(self.args.params_path) as params_file:  # TODO
            self.params = json.load(params_file)

        # Disable torch level parallelism
        torch.set_num_threads(1)

        iterator_settings = RunIteratorSettings(
            zbynek_mode=False
            , remove_duplicits_from_population=True
            , remove_duplicits_from_archive=True
            , remove_already_known_points_from_tss_selection_process=False
            , remove_already_known_points_from_evaluation_process=True
            , tss=2  # avaiable = {0,2}
            , use_distance_weight=True
        )

        model_settings = ExperimentSettings(
            name=f'exp-spectr-m-dkl{self.args.job}'
            , hyperparameters=self.params
            , losses=[losses.LossL1(), losses.LossL2(), losses.LossRDE_auto(), losses.LossKendall()]
            , additional_datasets=[]
            , root_directory=self.cfg['results']
            , allow_skipping=True
            , allow_skipping_completed_results=True
            , save_results=True
        )

        self.run_iterator = RunIterator(iterator_settings, model_settings, filter_dict=self.params['run'],
                                        data_folder=self.cfg['run_datafiles_folder'])

    def execute(self):
        for run in self.run_iterator:
            run.initialize_experiment()

        if self.args.n_cpus <= 1:
            for run in self.run_iterator:
                self.process_exp(run)
        else:
            with Pool(self.args.n_cpus) as pool:
                pool.map(self.process_exp, self.run_iterator)

    def process_exp(self, run):
        print('FILE:', run)

        for state in run:
            print(state)
            surrogate = build_model(self.params, log=log)
            if len(state.x_fit) == 0 or len(state.x_eval) == 0:
                state.provide_empty_result()
            else:
                fitted = surrogate.fit(state.x_fit, state.y_fit)
                if not fitted:
                    state.provide_empty_result()
                else:
                    y_pred = surrogate.predict(state.x_eval)
                    losses = state.provide_results(y_pred)


if __name__ == "__main__":
    ExperimentMain().execute()
