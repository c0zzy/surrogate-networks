#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
import shutil
import numpy as np
from scipy.io import loadmat
import collections

import copy
import tqdm

import import_ipynb
from dataset_duplicits_toolkit import *


DATA_PATH = '../../exp_doubleEC_28_log_nonadapt/'
DATA_OUTPUT = '../../npz-data/'

if False:
    shutil.rmtree(DATA_OUTPUT, ignore_errors=True)

os.makedirs(DATA_OUTPUT, exist_ok=True)

# In[2]:


named_tuple_types = {}


def flatten(item, verbose=False):
    # print(f'Flattening: {item}')
    # print(f'Shape: {item.shape}')
    if item.dtype.kind in ['O', 'V'] and item.shape == (1, 1):  # prob. cell
        if verbose:
            print("Object")
        return flatten(item[0, 0], verbose)
    elif item.dtype.kind == 'V' and item.shape == tuple():  # prob structure
        if verbose:
            print("Void")
        if item.dtype.names not in named_tuple_types:
            named_tuple_types[item.dtype.names] = collections.namedtuple('Structure', item.dtype.names)
        conv = [flatten(x, verbose) for x in item]
        assert len(conv) == len(item.dtype.names)
        return named_tuple_types[item.dtype.names](*conv)
    else:
        if item.shape == (1, 1):
            return item[0, 0]
        elif item.shape == (1,):
            return item[0]
        else:
            if verbose:
                print('Other - ?')
            return item


# In[3]:


class MatFileRun:
    def __init__(self, mat_file_dataset, index):
        self.glob = mat_file_dataset
        self.index = index
        self.y_evals = flatten(mat_file_dataset.y_evals[index, 0])
        self.cmaes_out = flatten(mat_file_dataset.cmaes_out[0, index])

    @staticmethod
    def keep_array(st):
        if isinstance(st, (float, int, np.int64, np.int32, np.int16, np.bool)):
            return np.array([st])
        elif isinstance(st, np.ndarray) and len(st.shape) == 2 and st.shape[0] == 1:
            return st[0, :]
        elif isinstance(st, np.ndarray) and len(st.shape) == 2 and st.shape[1] == 1:
            return st[:, 0]
        return st

    def get_filtered_content(self):
        points = self.cmaes_out.arxvalids.T  # points
        fvalues = self.keep_array(self.cmaes_out.fvalues)  # baseline
        orig_evaled = self.keep_array(self.cmaes_out.origEvaled.astype(bool))  # fvalues is orig?
        gen_split = self.keep_array((self.cmaes_out.generationStarts - 1))  # gen
        fvalues_orig = self.keep_array(self.cmaes_out.fvaluesOrig)  # !!! proc je to jinak??

        '''
        mask = duplicit_mask_with_hot_elements(points, orig_evaled)
        
        points = points[mask, ...]
        fvalues = fvalues[mask]
        orig_evaled = orig_evaled[mask]
        fvalues_orig = fvalues_orig[mask]
        gen_split = move_indices_given_boolean_mask(mask, gen_split)
        '''

        return {'points': points
            , 'fvalues': fvalues
            , 'orig_evaled': orig_evaled
            , 'fvalues_orig': fvalues_orig
            , 'gen_split': gen_split
                }

    def save(self, path):
        np.savez(path
                 , dimensions=self.glob.bbParams.dimensions  # #of dim
                 , function_id=self.glob.bbParams.functions  # function evaled
                 , exp_id=self.glob.exp_id  # name of experiment

                 , surrogate_param_set_size_max=self.glob.surrogateParams.modelOpts.trainsetSizeMax
                 , surrogate_param_range=self.glob.surrogateParams.modelOpts.trainRange
                 , surrogate_param_type=self.glob.surrogateParams.modelOpts.trainsetType
                 , surrogate_data_means=self.cmaes_out.means
                 , surrogate_data_sigmas=self.keep_array(self.cmaes_out.sigmas)
                 , surrogate_data_bds=np.stack(self.cmaes_out.BDs[0, :], axis=0)
                 , surrogate_data_diagCs=np.stack(self.cmaes_out.diagCs[0, :], axis=0)[..., 0]
                 , surrogate_data_diagDs=np.stack(self.cmaes_out.diagDs[0, :], axis=0)[..., 0]

                 # , points = self.cmaes_out.arxvalids.T # points
                 # , fvalues = self.keep_array(self.cmaes_out.fvalues) # baseline
                 # , orig_evaled = self.keep_array(self.cmaes_out.origEvaled.astype(bool)) # fvalues is orig?
                 # , gen_split = self.keep_array((self.cmaes_out.generationStarts - 1)) # gen
                 # , coco = self.keep_array(self.cmaes_out.fvaluesOrig) # !!! proc je to jinak??

                 , iruns=self.keep_array(self.cmaes_out.iruns)  # ??
                 , evals=self.cmaes_out.evals  # evaluations of o. fitness function ?
                 , **self.get_filtered_content()
                 )


# In[4]:


class MatFileDataset:
    '''
        bbParams          -
        cmaesParams       -
        cmaes_out         - (run, 1)
        exp_id            - str
        exp_results       -
        exp_settings      -
        surrogateParams   - 
        y_evals           - (1, run)
    '''

    def __init__(self, data_path):
        data_file = loadmat(data_path
                            , verify_compressed_data_integrity=True
                            , mat_dtype=False
                            , struct_as_record=True
                            )

        self._options_top_level = ['bbParams', 'cmaesParams', 'cmaes_out', 'exp_id', 'exp_results', 'exp_settings',
                                   'surrogateParams', 'y_evals']
        self._data_path = data_path

        self.__dict__.update(
            {name: value for name, value in data_file.items() if not name.startswith('__')}
        )

        self.bbParams = flatten(self.bbParams)
        self.cmaesParams = flatten(self.cmaesParams)
        # self.cmaes_out = _
        self.exp_id = flatten(self.exp_id)
        self.exp_results = flatten(self.exp_results)
        self.exp_settings = flatten(self.exp_settings)
        self.surrogateParams = flatten(self.surrogateParams)
        # self.y_evals = _

    def consistency_check(self):
        for i in self._options_top_level:
            assert hasattr(self, i)

        assert isinstance(self.exp_id, str)

        # assert len(self.cmaes_out) == len(self.bbParams.instances) == len(self.y_evals)

    @staticmethod
    def safe_cell_removal(array):
        assert len(array) == 1
        return array[0, 0]

    @staticmethod
    def convert_dtype_array_to_dictionary(array):
        assert array.shape == tuple()
        return {name: value for name, value in zip(array.dtype.names, array)}

    def __iter__(self):
        iterator = [MatFileRun(self, i) for i in range(len(self.y_evals))]
        return iter(iterator)


# In[5]:


if False:
    # pouze pro testovani

    # mf = MatFileDataset(DATA_PATH + "exp_doubleEC_28_log_nonadapt_results_1_2D_3.mat")
    mf = MatFileDataset(DATA_PATH + 'exp_doubleEC_28_log_nonadapt_results_5_2D_37.mat')
    for run in mf:
        pass

    print(run.get_filtered_content())

# In[6]:


if True:
    from multiprocessing import Pool

    N_CORES = os.cpu_count()
else:
    N_CORES = 1

mat_files = os.listdir(DATA_PATH)
mat_files = filter(lambda x: x.endswith('.mat'), mat_files)
mat_files = filter(lambda x: x.startswith('exp_doubleEC_28_log_nonadapt'), mat_files)
mat_files = list(mat_files)


def paralel_function(filename):
    strip_filename = filename[:-4]
    data_path = DATA_PATH + filename

    mfd = MatFileDataset(data_path)
    for runid, run in enumerate(mfd, start=0):
        run.save(DATA_OUTPUT + strip_filename + f"_{runid}.npz")


# In[7]:


if N_CORES <= 1:
    for filename in tqdm.tqdm(mat_files):
        paralel_function(filename)
else:
    import random

    random.shuffle(mat_files)
    with Pool(N_CORES) as p:
        v = p.imap_unordered(paralel_function, mat_files)
        list(tqdm.tqdm(v, total=len(mat_files)))

# In[ ]:
