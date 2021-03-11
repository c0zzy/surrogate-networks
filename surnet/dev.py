import argparse
import json
import os
from collections.abc import Iterable
from multiprocessing import Pool

import numpy as np
import torch
from matplotlib import pyplot as plt

from cfg import load_config
from surnet import losses
from data_iteration.data_iterators import RunIteratorSettings, RunIterator, ExperimentSettings, State
from models.linear import LinearSurrogate
from utils.data_files_utils import decode_filename
from utils.data_utils import range_normalize, min_max_normalize
from utils.logger import no_log
from utils.model_utils import build_model


def run_filter(run):
    # return run.name == 'exp_doubleEC_28_log_nonadapt_results_19_2D_166_0.npz'

    return run.dimensions == 2 \
           and run.kernel_id == 4 \
           and run.run == 0


def line_min(ax, xlim, x, y, color):
    on_x = x[np.argmin(y)]
    offset = (0.05 * np.random.rand() - 0.025) * np.ptp(x)
    # ax.lines(x=on_x, ymin=-1.2, ymax=-1, colors=color)

    ax.set_xlim(xlim)
    ax.set_ylim(-0.2, 1)
    ax.autoscale(False)

    ax.plot((on_x + offset, on_x), (-0.2, -0.15), color=color)


class ExperimentMain:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-cfg-env', help='Which config to load', type=str)
        parser.add_argument('-n-cpus', help='Number of CPUs', type=int)
        parser.add_argument('-dev', help='Development environment', action='store_true')
        self.args = parser.parse_args()

        self.cfg = load_config(env=self.args.cfg_env)

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
            name='testExp1'
            , hyperparameters={'a': 2}
            , losses=[losses.LossL1(), losses.LossL2(), losses.LossRDE_auto(), losses.LossKendall()]
            , additional_datasets=[]
            , root_directory='results'  # where the experiments will be saved
            , allow_skipping=True
            , allow_skipping_completed_results=True
            , save_results=True
        )

        filters = [run_filter]
        #
        # def aux_filter(run):
        #     return run.function_id == 1 \
        #            and run.dimensions == 2 \
        #            and run.kernel_id == 3 \
        #            and run.run == 1
        # filters = [aux_filter]

        self.run_iterator = RunIterator(iterator_settings, model_settings, filters=filters,
                                        data_folder=self.cfg['run_datafiles_folder'])

        with open('../params.json') as params_file:  # TODO
            self.params = json.load(params_file)

        # self.lock = Lock()

    def execute(self):
        if not self.process_exp_dev:
            for run in self.run_iterator:
                run.initialize_experiment()

        # processor = self.process_exp_dev if self.args.dev else self.process_exp
        processor = self.process_exp_plots
        # processor = self.process_exp_debug

        compare = []
        if self.args.n_cpus <= 1:
            for run in self.run_iterator:
                distances = processor(run)  # TODO
                if not distances:
                    continue
                mll, l2 = distances
                compare.append((mll, l2))
                better = 'MLL' if mll < l2 else 'L2'
                np_cmp = np.array(compare)
                mll_cnt = sum(np.less(np_cmp[:, 0], np_cmp[:, 1]))
                l2_cnt = len(np_cmp) - mll_cnt

                print(f"MLL: {mll}", f"L2: {l2}", better, f"cnt: {mll_cnt}\t{l2_cnt}", f"avg {np.average(np_cmp, 0)}",
                      f"med {np.median(np_cmp, 0)}",
                      sep='\t')

        else:
            with Pool(self.args.n_cpus) as pool:
                pool.map(processor, self.run_iterator)

    def process_exp(self, run):
        for state in run:
            print(state)
            surrogate = LinearSurrogate()
            if len(state.x_fit) == 0 or len(state.x_eval) == 0:
                state.provide_empty_result()
            else:
                surrogate.fit(state.x_fit, state.y_fit)
                y_pred = surrogate.predict(state.x_eval)
                losses = state.provide_results(y_pred)

    def process_exp_dev(self, run):
        for g, s in run.sparse_iter(10):
            print(g, run.name)
            rel = self.evaluate(run, self.params, g, s)
            print(rel)

    def diffs(self, y_pred, y_true, aggregate=np.average):
        diff_abs = np.square(y_true - y_pred)
        diff_rel = diff_abs / np.abs(y_true)
        if isinstance(aggregate, Iterable):
            agg_abs = []
            agg_rel = []
            for a in aggregate:
                agg_abs.append(a(diff_abs))
                agg_rel.append(a(diff_rel))
            return agg_abs, agg_rel
        return aggregate(diff_abs), aggregate(diff_rel)

    def evaluate(self, run, params, g, s):
        surrogate = build_model(params, log=no_log)
        results = surrogate.fit(s.x_fit, s.y_fit, iters=100)
        if not results:
            return None
        # model, loss, val_err = results
        y_pred = surrogate.predict(s.x_eval)
        abs, rel = self.diffs(y_pred, s.y_eval, aggregate=(np.median, np.average, np.max))
        return rel

    def process_exp_debug(self, run):
        g = 22
        s = State(run, g)
        surrogate = build_model(self.params, log=no_log)
        print(g, run.name, f'train: {s.x_fit.shape}', f'eval: {s.x_eval.shape}')
        model, loss, val_err, ls, tr, val = surrogate.fit(s.x_fit, s.y_fit, iters=500)

    def process_exp_plots(self, run):
        decoded = decode_filename(run.name)

        for g, s in run.sparse_iter(10):
            decoded['gen'] = g
            surrogate = build_model(self.params, log=no_log)
            print(g, run.name, f'train: {s.x_fit.shape}', f'eval: {s.x_eval.shape}')
            reps = 100
            gs = np.arange(10, 10 * reps + 1, 10)
            abss = np.empty((reps, 3))
            rels = np.empty((reps, 3))
            losses = np.empty(reps)
            errsL2 = np.empty(reps)
            errsMLL = np.empty(reps)
            lss = np.empty(reps)

            results = False
            for i in range(reps):
                results = surrogate.fit(s.x_fit, s.y_fit, iters=10)
                if results:
                    model, loss, val_err, ls, tr, val = results
                    losses[i] = loss
                    errsL2[i], errsMLL[i] = val_err
                    lss[i] = ls
                    y_pred = surrogate.predict(s.x_eval)
                    abs, rel = self.diffs(y_pred, s.y_eval, aggregate=(np.median, np.average, np.max))
                    abss[i] = abs
                    rels[i] = rel

            if not results:
                print('skipped')
                continue

            amin_mll = np.argmin(errsMLL)
            amin_l2 = np.argmin(errsL2)
            amin_med, amin_avg, amin_mx = np.argmin(rels, axis=0)

            dist_mll = np.abs(amin_mll - amin_avg)
            dist_l2 = np.abs(amin_l2 - amin_avg)

            # self.plot_model_optimization(decoded, errsL2, errsMLL, g, gs, lss, rels, run, s, tr, val)
            return dist_mll, dist_l2

        return None

    @staticmethod
    def plot_model_optimization(decoded, errsL2, errsMLL, g, gs, lss, rels, run, s, tr, val):
        # normalize val errors:
        errsL2 = min_max_normalize(errsL2)
        errsMLL = - min_max_normalize(errsMLL)
        plt.figure(figsize=(12, 12))
        ax = plt.subplot(211)
        plt.subplots_adjust(right=0.75)
        ax.plot(gs, rels[:, 0], color='blue')
        ax.plot(gs, rels[:, 1], color='green')
        ax.plot(gs, rels[:, 2], color='red')
        ax.set_xlabel("Optimizer iteration")
        ax.set_ylabel("Relative")
        ax2 = ax.twinx()
        # ax2.plot(gs, abss[:, 1], alpha=.1)
        # ax2.set_ylabel("Absolute")
        # print(d)
        ax2.plot(gs, errsL2, color='magenta')
        ax2.plot(gs, errsMLL, color='purple')
        ax2.set_ylabel("Error")

        def make_patch_spines_invisible(ax):
            ax.set_frame_on(True)
            ax.patch.set_visible(False)
            for sp in ax.spines.values():
                sp.set_visible(False)

        ax3 = ax.twinx()
        ax3.spines["right"].set_position(("axes", 1.2))
        make_patch_spines_invisible(ax3)
        ax3.spines["right"].set_visible(True)
        ax3.plot(gs, lss, color='cyan')
        ax3.set_ylabel("LS")
        ax4 = ax.twinx()
        line_min(ax4, ax3.get_xlim(), gs, lss, color='cyan')
        line_min(ax4, ax.get_xlim(), gs, rels[:, 0], color='blue')
        line_min(ax4, ax.get_xlim(), gs, rels[:, 1], color='green')
        line_min(ax4, ax.get_xlim(), gs, rels[:, 2], color='red')
        line_min(ax4, ax2.get_xlim(), gs, errsMLL, color='purple')
        line_min(ax4, ax2.get_xlim(), gs, errsL2, color='magenta')
        plt.title(decoded)
        bottom = plt.subplot(212)
        bottom.set_aspect(1)
        ev = range_normalize(s.x_eval)
        bottom.scatter(ev[:, 0], ev[:, 1], color='red')
        bottom.scatter(tr[:, 0], tr[:, 1], color='blue')
        bottom.scatter(val[:, 0], val[:, 1], color='lime')
        folder = f'plots-mlp-{run.dimensions}'
        os.makedirs(f'output/{folder}', exist_ok=True)
        plt.savefig(f'output/{folder}/{run.name}_{g}.png')


if __name__ == "__main__":
    ExperimentMain().execute()
