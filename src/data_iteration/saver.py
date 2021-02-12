#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import sys
sys.path.append('../data_iteration')

import functools
import itertools
import copy
import datetime
import collections
import logging
import re
from abc import ABC, abstractmethod

import numpy as np

import csv
import json
import tempfile
import h5py


class _HDF5_Model_base(ABC):
    def __init__(self, *, model: str, root_directory='results'):
        self.model = model
        self.root_directory = root_directory
        
    @property
    def _dispatcher_path(self):
        return os.path.join(self.root_directory, self.model, 'dispatcher.csv')
        
    def _create_dispatcher(self):
        with open(self._dispatcher_path, mode='w', newline='') as dis_file:
            dis_writer = csv.writer(dis_file)
            dis_writer.writerow(['dir_name', 'hyperparameters'])


class HyperparameterInspector(_HDF5_Model_base):
    def __getitem__(self, item):
        assert isinstance(item, int)
        with open(self._dispatcher_path, mode='r', newline='') as dis_file:
            dis_reader = csv.reader(dis_file)
            for (directory, hp) in itertools.islice(dis_reader, 1 + item, None):
                return json.loads(hp)
    
    def __repr__(self):
        result = ''
        with open(self._dispatcher_path, mode='r', newline='') as dis_file:
            dis_reader = csv.reader(dis_file)
            for i, (directory, hp) in enumerate(itertools.islice(dis_reader, 1, None)):
                result += f'[{i}]: < {str(json.loads(hp))} >\n'
        return result


class _HDF5_Hyperparameters_base(_HDF5_Model_base):
    def __init__(self, *, model: str, hyperparameters: dict, root_directory = 'results'):
        super().__init__(model = model, root_directory=root_directory)
        self.hyperparameters = hyperparameters
        self._model_directory_cached = None
            
    def _create_model_directory(self):
        d_path = self._dispatcher_path
        serialization = json.dumps(self.hyperparameters)
        directory_prefix =  datetime.datetime.now().strftime('%Y-%m-%d_%H:%M_')
        directory = tempfile.mkdtemp( prefix=directory_prefix, dir=os.path.dirname(d_path) )
        
        with open(d_path, mode='a', newline='') as dis_file:
            dis_writer = csv.writer(dis_file)
            dis_writer.writerow([directory, serialization])
        return directory
        
    def _compare_hyperparameters(self, loaded, searched):
        return loaded == searched
    
    @property
    def _model_directory(self): # == use dispatcher
        if self._model_directory_cached is not None:
            return self._model_directory_cached
        
        with open(self._dispatcher_path, newline='') as dis_file:
            dis_reader = csv.reader(dis_file)
            for (directory, hp) in itertools.islice(dis_reader, 1, None):
                if self._compare_hyperparameters(json.loads(hp), self.hyperparameters):
                    self._model_directory_cached = directory
                    return self._model_directory_cached
            return None

class _HDF5_Concrete_base(_HDF5_Hyperparameters_base):
    def __init__(self, *
        , model: str , hyperparameters: dict , root_directory = 'resluts'
        , function_id: int
        , dimension: int
        , run: int
        ):
        super().__init__(model=model, hyperparameters=hyperparameters, root_directory=root_directory)
        
        self.run = run
        self.dimension = dimension
        self.function_id = function_id
        self.init()
    
    @property
    def hdf5_final_path(self):
        return os.path.join(self._model_directory, str(self.function_id),
            str(self.dimension), f'{self.run}.hdf5')
    @property
    def hdf5_tmp_path(self):
        return os.path.join(self._model_directory, str(self.function_id),
            str(self.dimension), f'{self.run}.hdf5_tmp')
    @abstractmethod
    def init(self):
        pass

class Initilaizer(_HDF5_Concrete_base):
    def __init__(self, *args, 
            remove_existing_files = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.remove_existing_files = remove_existing_files

    def init(self):
        os.makedirs(os.path.dirname(self._dispatcher_path), exist_ok=True)
        
        if not os.path.exists(self._dispatcher_path):
            self._create_dispatcher()
            
        model_directory = self._model_directory
        if model_directory is None:
            model_directory = self._create_model_directory()

        if self.remove_existing_files:
            try:
                os.remove(self.hdf5_final_path, )
            except OSError:
                pass

            try:
                os.remove(self.hdf5_tmp_path)
            except OSError:
                pass
        
        self._construct_hdf5_dataset(self.hdf5_tmp_path)

    
    def _construct_hdf5_dataset(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # use only 'w'
        with h5py.File(path, 'w-') as h5f:
            dset = h5f.create_dataset('prediction'
                        , shape=(0,0)
                        , chunks=(128,32)
                        , maxshape=(None, None)
                        , dtype=np.float32
                        , fillvalue=-np.inf)
            dset = h5f.create_dataset('target'
                        , shape=(0,0)
                        , chunks=(128,32)
                        , maxshape=(None, None)
                        , dtype=np.float32
                        , fillvalue=-np.inf)
            dset = h5f.create_dataset('training_samples'
                        , shape=(0,)
                        , chunks=(128,)
                        , maxshape=(None,)
                        , dtype=np.int32
                        , fillvalue=0)


class Saver(_HDF5_Concrete_base):
    def init(self, disable=False):
        self.completed = False
        self._skip = 0
        self.disable = disable
        
        if os.path.exists(self.hdf5_final_path):
            logging.info(f'Skipping: Found already created final file: "{self.hdf5_final_path}"')
            self.completed = True
        else: # continue
            self._skip = self._check_hdf5_dataset_state(self.hdf5_tmp_path)
            self._already_computed = self._skip
        
    def finalize(self):
        os.rename(self.hdf5_tmp_path, self.hdf5_final_path)
        self.completed = True
        
    def _check_hdf5_dataset_state(self, path):
        with h5py.File(path, 'r') as h5f:
            prediction = h5f['prediction']
            target = h5f['target']
            training_size = h5f['training_samples']
            
            if prediction.shape != target.shape or prediction.shape[0] != training_size.shape[0]:
                logging.warning(f'Shape mismatch prediction:{prediction.shape} target:{target.shape} training_samples:{training_size.shape} in file {path}')
            
            return min(min(prediction.shape[0], target.shape[0]), training_size.shape[0])
            
    def write_results_to_dataset(self, *, prediction, target, training_samples):
        assert isinstance(prediction, np.ndarray)
        assert isinstance(target, np.ndarray)
        assert len(prediction.shape) == 1
        assert target.shape == prediction.shape
        training_samples = int(training_samples)
        
        if self.disable:
            return 
        
        with h5py.File(self.hdf5_tmp_path, 'a') as h5f:
            pred_d = h5f['prediction']
            targ_d = h5f['target']
            size_d = h5f['training_samples']
            
            pred_d.resize( (self._skip+1, max(pred_d.shape[1], len(prediction))) )
            pred_d[-1, :len(prediction)] = prediction
            
            targ_d.resize( (self._skip+1, max(targ_d.shape[1], len(prediction))) )
            targ_d[-1, :len(target)] = target 
            
            size_d.resize( (self._skip+1,) )
            size_d[-1] = training_samples
            
            self._skip += 1
            
    @property
    def already_computed(self):
        return self._already_computed

class Loader(_HDF5_Concrete_base):
    def init(self):
        self.completed = False
        
        if os.path.exists(self.hdf5_final_path):
            self.completed = True
            self._data_path = self.hdf5_final_path
        else:
            logging.warning('Final result not found: using tmp')
            self._data_path = self.hdf5_tmp_path
            
    @property
    def data(self):
        with h5py.File(self._data_path, 'r') as h5f:
            prediction = h5f['prediction']
            target = h5f['target']
            size = h5f['training_samples']
            return np.array(prediction), np.array(target), np.array(size)
            
class LoaderIterator(_HDF5_Hyperparameters_base):
    def __iter__(self):
        m = copy.deepcopy(self)
        m.files = []
        for root, _, files in os.walk(self._model_directory):
            for fil in files:
                m.files.append(os.path.normpath(os.path.join(root, fil)))
        m.type_output = collections.namedtuple(
            'RunDescription', 
            ['function_id', 'dimension', 'run']
        )
        return m
    
    def __next__(self):
        try:
            filepath = self.files.pop() 
        except IndexError:
            raise StopIteration()

        func_id, dim, basefilename = filepath.split(os.sep)[-3:]
        run = basefilename.split('.')[0]

        conf = self.type_output(int(func_id), int(dim), int(run))
        l = Loader(
                model=self.model, 
                hyperparameters=self.hyperparameters, 
                function_id=conf.function_id,
                dimension=conf.dimension,
                run=conf.run,
                root_directory=self.root_directory
            )
        return l, conf


# In[16]:


if __name__ == '__main__':
    import unittest
    import shutil
    import random
    import collections


# In[17]:


if __name__ == '__main__':
    class TestInitializer(unittest.TestCase):
        testFileDirName = 's1412_test'
        
        @classmethod
        def setUpClass(cls):
            shutil.rmtree(cls.testFileDirName, ignore_errors=True)
            
        @classmethod
        def tearDownClass(cls):
            shutil.rmtree(cls.testFileDirName)
            
        def setUp(self):
            #self.testFileDirName = self.__class__.testFileDirName
            pass
            
        def test_loader(self):
            # CREATE FIRST MODEL
            Initilaizer( model='testModel1', hyperparameters={'a': 12.3, 'b': 15}
                    , function_id=3 , dimension=2 , run=0 , root_directory = self.testFileDirName
                  )
            self.assertTrue(os.path.exists(
                os.path.join(self.testFileDirName, 'testModel1', 'dispatcher.csv') ))
            m1_dir = set(os.listdir(os.path.join(self.testFileDirName, 'testModel1')))
            m1_dir.remove('dispatcher.csv')
            m1_dir = m1_dir.pop()
            
            self.assertTrue(os.path.exists(
                os.path.join(self.testFileDirName, 'testModel1', m1_dir, '3', '2', '0.hdf5_tmp') ))
            with open(os.path.join(self.testFileDirName, 'testModel1', 'dispatcher.csv'), 'r') as f:
                self.assertEqual(sum(1 for line in f), 2)
            self.assertEqual(len(os.listdir(self.testFileDirName)), 1)
            
            # CREATE SECOND MODEL
            Initilaizer( model='testModel2', hyperparameters={'a': 12.3, 'b': 15}
                    , function_id=2 , dimension=4 , run=2 , root_directory = self.testFileDirName
                  )
            self.assertTrue(os.path.exists(
                os.path.join(self.testFileDirName, 'testModel2', 'dispatcher.csv') ))
            m2_dir = set(os.listdir(os.path.join(self.testFileDirName, 'testModel2')))
            m2_dir.remove('dispatcher.csv')
            m2_dir = m2_dir.pop()
            self.assertTrue(os.path.exists(
                os.path.join(self.testFileDirName, 'testModel2', m2_dir, '2', '4', '2.hdf5_tmp') ))
            with open(os.path.join(self.testFileDirName, 'testModel2', 'dispatcher.csv'), 'r') as f:
                self.assertEqual(sum(1 for line in f), 2)
            self.assertEqual(len(os.listdir(self.testFileDirName)), 2)
                
                
            # CREATE THIRD...
            Initilaizer( model='testModel1', hyperparameters={'a': 12.3, 'b': 16} # <--- minor change of hyp.
                    , function_id=1 , dimension=2 , run=0 , root_directory = self.testFileDirName
                  )
            self.assertTrue(os.path.exists(
                os.path.join(self.testFileDirName, 'testModel1', 'dispatcher.csv') ))
            with open(os.path.join(self.testFileDirName, 'testModel1', 'dispatcher.csv'), 'r') as f:
                self.assertEqual(sum(1 for line in f), 3)
            m3_dir = set(os.listdir(os.path.join(self.testFileDirName, 'testModel1')))
            m3_dir.remove('dispatcher.csv')
            m3_dir.remove(m1_dir)
            m3_dir = m3_dir.pop()
            
            self.assertTrue(os.path.exists(
                os.path.join(self.testFileDirName, 'testModel1', m3_dir, '1', '2', '0.hdf5_tmp') ))
            self.assertEqual(len(os.listdir(self.testFileDirName)), 2)
            
            f = list(itertools.chain.from_iterable((files for subdir, dirs, files in os.walk(self.testFileDirName))))
            self.assertEqual(len([files for files in f if '.hdf5' in files]), 3)
            self.assertEqual( len([files for files in f if files == 'dispatcher.csv']), 2)
        


# In[18]:


if __name__ == '__main__':
    class TestSaverLoader(unittest.TestCase):
        testFileDirName = 's1412_test'
        
        @classmethod
        def setUpClass(cls):
            shutil.rmtree(cls.testFileDirName, ignore_errors=True)
            
        @classmethod
        def tearDownClass(cls):
            shutil.rmtree(cls.testFileDirName)
            
        def setUp(self):
            pass
            
        def test_loader(self):
            # CREATE FIRST MODEL
            Initilaizer(model='testModel1', hyperparameters={'a': 12.3, 'b': 15}
                , function_id=3 , dimension=2 , run=0 , root_directory = self.testFileDirName
                  )
            
            s = Saver(model='testModel1', hyperparameters={'a': 12.3, 'b': 15}
                , function_id=3 , dimension=2 , run=0 , root_directory = self.testFileDirName
                  )
            
            self.assertEqual(s.already_computed, 0)
            
            sizes = [10, 72, 111, 12, 312]
            
            for i,size in enumerate(sizes):
                # saver
                s.write_results_to_dataset(
                    prediction = np.arange(size),
                    target = np.arange(size) + 1,
                    training_samples = size
                )
                
                # loader
                l = Loader(model='testModel1', hyperparameters={'a': 12.3, 'b': 15}
                    , function_id=3 , dimension=2 , run=0 , root_directory = self.testFileDirName
                    )
                
                pred, tar, si = l.data
                self.assertEqual(pred.shape[0], i+1)
                self.assertEqual(tar.shape[0], i+1)
                self.assertEqual(si.shape[0], i+1)
                
                for y in range(i+1):
                    self.assertTrue(np.all(pred[y, :sizes[y]] == np.arange(sizes[y])))
                    self.assertTrue(np.all(pred[y, sizes[y]:] == -np.inf))
                    
                    self.assertTrue(np.all(tar[y, :sizes[y]] == 1 + np.arange(sizes[y])))
                    self.assertTrue(np.all(tar[y, sizes[y]:] == -np.inf))
                    
                    self.assertEqual(si[y], sizes[y])
                    
            ####  NEW SAVER
            s = Saver(model='testModel1', hyperparameters={'a': 12.3, 'b': 15}
                , function_id=3 , dimension=2 , run=0 , root_directory = self.testFileDirName )
            self.assertEqual(len(sizes), s.already_computed)
            
            s.write_results_to_dataset(
                prediction = np.arange(10),
                target = np.arange(10) + 1,
                training_samples = 10)
            
            l = Loader(model='testModel1', hyperparameters={'a': 12.3, 'b': 15}
                , function_id=3 , dimension=2 , run=0 , root_directory = self.testFileDirName)
            
            pred, tar, si = l.data
            self.assertEqual(pred.shape[0], len(sizes) + 1)
            self.assertEqual(tar.shape[0], len(sizes) + 1)
            self.assertEqual(si.shape[0], len(sizes) + 1)
            
            self.assertTrue(np.all(pred[-1, :10] == np.arange(10)))
            self.assertTrue(np.all(pred[-1, 10:] == -np.inf))

            self.assertTrue(np.all(tar[-1, :10] == 1 + np.arange(10)))
            self.assertTrue(np.all(tar[-1, 10:] == -np.inf))

            self.assertEqual(si[y], sizes[y])
            
            s.finalize()
            
            #### FINALIZE + LOADER AGAIN
            
            l = Loader(model='testModel1', hyperparameters={'a': 12.3, 'b': 15}
                , function_id=3 , dimension=2 , run=0 , root_directory = self.testFileDirName)
            
            sizes = sizes + [10]
            
            for y in range(len(sizes)):
                self.assertTrue(np.all(pred[y, :sizes[y]] == np.arange(sizes[y])))
                self.assertTrue(np.all(pred[y, sizes[y]:] == -np.inf))

                self.assertTrue(np.all(tar[y, :sizes[y]] == 1 + np.arange(sizes[y])))
                self.assertTrue(np.all(tar[y, sizes[y]:] == -np.inf))

                self.assertEqual(si[y], sizes[y])


# In[19]:


if __name__ == '__main__':
    class TestIterator(unittest.TestCase):
            testFileDirName = 's1412_test'

            @classmethod
            def setUpClass(cls):
                shutil.rmtree(cls.testFileDirName, ignore_errors=True)

            @classmethod
            def tearDownClass(cls):
                shutil.rmtree(cls.testFileDirName)

            def setUp(self):
                #self.testFileDirName = self.__class__.testFileDirName
                pass

            def create_data(self, model, hyperparameters, fid, dim, run, offset=0):
                Initilaizer(model=model, hyperparameters=hyperparameters, 
                    function_id=fid , dimension=dim , run=run , root_directory = self.testFileDirName)

                s = Saver(model=model, hyperparameters=hyperparameters, 
                        function_id=fid , dimension=dim , run=run , root_directory = self.testFileDirName )

                for size in [2, 5, 3, 17, 13]:
                    s.write_results_to_dataset(
                        prediction = np.arange(size),
                        target = np.arange(size) + fid + dim*100 + run*10000 + offset,
                        training_samples = size)

                s.finalize()


            def check_data(self, data, fid, dim, run, offset = 0):
                pred, targ, tras = data

                for i, size in enumerate([2, 5, 3, 17, 13]):
                    self.assertTrue(np.all(np.arange(size) == pred[i,:size]))
                    self.assertTrue(np.all(-np.inf == pred[i,size:]))
                    self.assertTrue(np.all(np.arange(size) + fid + dim*100 + run*10000 + offset == targ[i,:size]))
                    self.assertTrue(np.all(-np.inf == targ[i,size:]))

                    self.assertEqual(tras[i], size)


            def test_loader(self):
                c = collections.defaultdict(set)
                for (model, of1) in [('testModel1', 10), ('testModel2', 11)]:
                    for hyp in [{'hyp': 2}, {'hyp': 3}, {'hyp': 4}]:
                        for fid in [11,22,33]:
                            for dim in [10,20,30]:
                                for run in [1,2,3]:
                                    c[(model, hyp['hyp'])].add((fid,dim,run))
                                    self.create_data(model, hyp, fid, dim, run, offset=of1 + hyp['hyp'])

                for (model, of1) in [('testModel1', 10), ('testModel2', 11)]:
                    for hyp in [{'hyp': 2}, {'hyp': 3}, {'hyp': 4}]:
                        it = LoaderIterator(model=model, hyperparameters=hyp, 
                                            root_directory=self.testFileDirName )
                        for (loader, ids) in it:
                            c[(model, hyp['hyp'])].remove( (ids.function_id, ids.dimension, ids.run))
                            self.check_data(loader.data, ids.function_id, ids.dimension, ids.run, offset=of1 + hyp['hyp'])
                
                for t in c.values():
                    self.assertEqual(0, len(t))
                    
                hi = HyperparameterInspector(model='testModel1', root_directory=self.testFileDirName)
                print(hi)
                print(hi[2])


# In[20]:



if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

