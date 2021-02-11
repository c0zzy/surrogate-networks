#!/usr/bin/env python
# coding: utf-8
from collections import Iterable

import numpy as np
import os
import itertools
import collections
import re
import copy
import math
from abc import ABC, abstractmethod

import dataset_duplicits_toolkit


class RunIteratorSettings:
    def __init__(self
                 , zbynek_mode=False  # I'm interested in how much I'm loosing vs our lord Zbyňo
                 , remove_duplicits_from_population=True
                 , remove_duplicits_from_archive=True
                 , remove_already_known_points_from_tss_selection_process=False  # aka mistake...
                 , remove_already_known_points_from_evaluation_process=True
                 , tss=2  # avaiable = {0,2}
                 , use_distance_weight=True
                 ):
        a = locals()
        del a['self']
        self.__dict__.update(a)


class TSSbase(ABC):
    def __init__(self
                 , population
                 , mahalanobis_transf):
        self.population = population
        self.mahalanobis_transf = mahalanobis_transf

        assert isinstance(self.mahalanobis_transf, np.ndarray)
        assert len(self.mahalanobis_transf.shape) == 2
        assert self.mahalanobis_transf.shape[0] == self.mahalanobis_transf.shape[1]
        assert self.mahalanobis_transf.shape[0] > 0
        assert len(self.population.shape) == 2
        assert self.population.shape[1] == self.mahalanobis_transf.shape[0]

    @abstractmethod
    def __call__(self, archive_points, archive_evaluation, compute_distances=True, add_minimal_distances=False):
        assert len(archive_points.shape) == 2
        assert archive_points.shape[1] == self.population.shape[1]
        assert len(archive_evaluation.shape) == 1
        assert archive_evaluation.shape[0] == archive_points.shape[0]

        if compute_distances or add_minimal_distances:
            # (O, G, D)
            differences = self.population[np.newaxis, :, :] - archive_points[:, np.newaxis, :]
            # (O, G)
            distances = self.mahalanobis_distance(differences, self.mahalanobis_transf, do_not_square=True)
        else:
            distances = None

        if add_minimal_distances:
            return distances, {'minimal_distances': np.sqrt(np.amin(distances, axis=1))}
        else:
            return distances, {}

    @staticmethod
    def apply_mask_to_dictionary(mask, dictionary):
        for key in dictionary.keys():
            data = dictionary[key]
            if isinstance(data, np.ndarray) and len(data.shape) == 1 and len(data) == len(mask):
                dictionary[key] = data[mask]
            else:
                raise NotImplementedError(f'How to convert {key} ?')
        return dictionary

    @staticmethod
    def mahalanobis_distance(differences, mahalanobis_transf, do_not_square=False):
        N = differences.shape[:-1]
        D = differences.shape[-1]

        assert mahalanobis_transf.shape == (D, D)

        # centered_points = np.matmul(differences, mahalanobis_transf)
        centered_points = np.matmul(mahalanobis_transf, differences[..., np.newaxis])[..., 0]

        if do_not_square:
            return np.sum(np.square(centered_points), axis=-1)
        else:
            return np.sqrt(np.sum(np.square(centered_points), axis=-1))


class TSS0(TSSbase):
    # The hungry one...

    def __call__(self, archive_points, archive_evaluation, **kwargs):
        distances, other_info = super().__call__(archive_points, archive_evaluation, compute_distances=False, **kwargs)
        mask = np.ones(shape=(len(archive_points),), dtype=np.bool)
        return mask, other_info


class TSS2(TSSbase):
    def __init__(self
                 , population
                 , mahalanobis_transf
                 , maximum_distance
                 , maximum_number):
        super().__init__(population, mahalanobis_transf)

        self.maximum_distance = maximum_distance
        self.maximum_number = maximum_number
        assert isinstance(self.maximum_number, (int, float))
        assert isinstance(self.maximum_distance, (int, float))

    def __call__(self, archive_points, archive_evaluation, **kwargs):
        distances, other_info = super().__call__(archive_points, archive_evaluation, **kwargs)

        D = archive_points.shape[1]
        N = archive_points.shape[0]
        P = self.population.shape[0]

        # Firstly, do not limit number of neighbours
        mask = np.any(distances <= self.maximum_distance ** 2, axis=1)
        if np.sum(mask) <= self.maximum_number:
            return mask, self.apply_mask_to_dictionary(mask, other_info)

        # It's too much
        distances_indices = np.argsort(distances, axis=0)
        distances_sorted = np.stack(
            [distances[distances_indices[:, i], i] for i in range(P)], axis=1)
        distances_enable = distances_sorted <= self.maximum_distance ** 2

        # compute minimal size
        min_perin_size = math.floor(self.maximum_number / P)

        mask.fill(False)  # reuse mask array
        new_elems_ind = distances_indices[:min_perin_size, :][distances_enable[:min_perin_size, :]]
        mask[new_elems_ind] = True
        enabled_n = np.sum(mask)

        if enabled_n == self.maximum_number:
            return mask, self.apply_mask_to_dictionary(mask, other_info)
        else:
            # create another array
            new_mask = mask.copy()

        for perinp in itertools.count(start=min_perin_size):
            new_elems_ind = np.unique(distances_indices[perinp, :][distances_enable[perinp, :]])
            new_elems_ind = new_elems_ind[new_mask[new_elems_ind] == False]

            new_mask[new_elems_ind] = True

            enabled_o = enabled_n
            enabled_n += len(new_elems_ind)  # np.sum(new_mask)

            if enabled_n == self.maximum_number:
                return new_mask, self.apply_mask_to_dictionary(new_mask, other_info)
            elif enabled_n < self.maximum_number:
                mask, new_mask = new_mask, mask
                np.copyto(new_mask, mask)
            else:
                new_elems_eval = archive_evaluation[new_elems_ind]
                new_elems_ind_selection = np.argsort(new_elems_eval)[:self.maximum_number - enabled_o]
                selection = new_elems_ind[new_elems_ind_selection]

                mask[selection] = True
                return mask, self.apply_mask_to_dictionary(mask, other_info)


class State:
    def __init__(self, run, gen):
        self._run = run
        self._settings = run.settings
        self._gen = gen

        if gen + 2 == len(self._run.gen_split):
            pos, next_pos = self._run.gen_split[gen], None
        else:
            pos, next_pos = self._run.gen_split[gen:gen + 2]

        orig_evaled_mask = self._run.orig_evaled[:pos]
        self.x_fit = self._run.points[:pos, ...][orig_evaled_mask, ...]
        self.y_fit = self._run.fvalues_orig[:pos][orig_evaled_mask]

        population = self._run.points[pos:next_pos]

        # Duplicit filtering
        if self._settings.remove_duplicits_from_archive:
            self.x_fit, self.y_fit = self._remove_duplicits_basic(self.x_fit, self.y_fit)
            
        archive_points = self.x_fit
            
        if self._settings.remove_duplicits_from_population:    
            population, = self._remove_duplicits_basic(population)
        
        if self._settings.remove_already_known_points_from_tss_selection_process:
            population = self._remove_elements(archive_points, population)
            
        # TSS filtering
        tss = self._get_tss(population)
        self.x_fit, self.y_fit = self._select_by_tss(tss, self.x_fit, self.y_fit)
        
        # Evaluation ....
        self.x_eval = self._run.points[pos:next_pos, ...]
        self.y_eval = self._run.fvalues_orig[pos:next_pos]

        if self._settings.zbynek_mode:
            orig = self._run.orig_evaled[pos:next_pos]

            self.x_eval = self.x_eval[~orig, ...]
            self.y_eval = self.y_eval[~orig]
            self.y_eval_base = self._run.fvalues[pos:next_pos][~orig]

        # Remove evaluation dupl.
        if self._settings.remove_already_known_points_from_evaluation_process:
            if self._settings.zbynek_mode: 
                self.x_eval, self.y_eval, self.y_eval_base =                     self._remove_elements(archive_points, self.x_eval, self.y_eval, self.y_eval_base)
            else:
                self.x_eval, self.y_eval = self._remove_elements(archive_points, self.x_eval, self.y_eval)
            
        
    @staticmethod
    def _remove_duplicits_basic(points, *others):
        oepm = dataset_duplicits_toolkit.duplicit_mask(points)
        points = points[oepm, ...]
        return (points,) + tuple(x[oepm, ...] for x in others)

    @staticmethod
    def _remove_elements(to_remove, points, *others):
        assert to_remove.shape[1] == points.shape[1]
        assert len(to_remove.shape) == len(points.shape) == 2

        all_elems = np.concatenate([to_remove, points], axis=0)
        hot_mask = np.concatenate([
            np.ones(shape=(len(to_remove),), dtype=np.bool),
            np.zeros(shape=(len(points),), dtype=np.bool)
        ])
        mask = dataset_duplicits_toolkit.duplicit_mask_with_hot_elements(all_elems, hot_mask)[len(to_remove):]
        return (points[mask, ...], ) + tuple(x[mask, ...] for x in others)
        
    def _select_by_tss(self, tss, points, evals, *others):
        kwargs = {}
        if self._settings.use_distance_weight:
            kwargs['add_minimal_distances'] = True

        mask, other_info = tss(points, evals, **kwargs)

        if self._settings.use_distance_weight:
            self.distances = other_info['minimal_distances']

        return (points[mask, ...], evals[mask]) + tuple(x[mask, ...] for x in others)
        
    def _get_tss(self, population):
        assert self._run.surrogate_param_type == 'nearest'

        sigma = self._run.surrogate_data_sigmas[self._gen]
        BD = self._run.surrogate_data_bds[self._gen, ...]
        # diagD = self._run.surrogate_data_diagDs[self._gen]
        # diagC = self._run.surrogate_data_diagCs[self._gen]

        mahalanobis_transf = np.linalg.inv(BD * sigma)

        if self._settings.tss == 2:
            maximum_distance = float(self._run.surrogate_param_range)

            maximum_number = str(self._run.surrogate_param_set_size_max)
            v = re.match(r'(\d+)\*dim', maximum_number)
            if not v:
                raise NotImplementedError(f"maximum_number cannot be interpreted: {maximum_number}")
            maximum_number = int(int(v.group(1)) * self._run.dimensions)

            return TSS2(population, mahalanobis_transf, maximum_distance, maximum_number)
        elif self._settings.tss == 0:
            return TSS0(population, mahalanobis_transf)
        else:
            raise NotImplementedError("I don't know what to say...")


class StateIterator:  # / == Run
    '''
        name 
        path
        
        points - points in domain
        fvalues - points in codomain that are assumed to be true
        orig_evaled - what points are exactly evaluated and what are assumed to be true (surrogate)
        gen_split - 
        iruns - 
        evals - the over all number of evaluations of the original fitness function (integer).
        coco - ground truth for codomain
        ...
    '''

    def __init__(self, name=None, path=None, **kwargs):
        self.name = name
        self.path = path
        self.__dict__.update(kwargs)
        self._loaded = False

    def __getattr__(self, name):
        # lazy evaluation :)
        if name == "__setstate__":
            raise AttributeError(name)
        if not self._loaded:
            # print('Loading...')
            npob = np.load(self.path)
            self.__dict__.update(npob.items())
            self._loaded = True

            self.evals = int(self.evals)
        return super().__getattribute__(name)

    def __iter__(self):
        return iter(map(lambda g: State(self, g), range(1, len(self.gen_split) - 1)))

    def __getitem__(self, items):
        if isinstance(items, Iterable):
            return [State(self, i) for i in items]
        return State(self, items)

    def __repr__(self):
        return self.name

    def sparse_iter(self, steps=10):
        gens = len(self.gen_split)
        selected = np.linspace(1, gens - 2, steps, dtype=int)
        return iter((g, State(self, g)) for g in selected)


class RunIterator:  # == File
    def __init__(self
                 , settings
                 , data_folder='../../npz-data'
                 , filters=None
                 ):
        self.data_folder = data_folder
        self.settings = settings

        if filters is None:
            self.filters = []
        elif isinstance(filters, collections.abc.Iterable):
            self.filters = filters
        else:
            self.filters = [filters]

        self._inspect_data_folder()

    @staticmethod
    def _order_key(file):
        return [int(num) for num in re.findall("\d+", file)]

    def _inspect_data_folder(self):
        self.data_files = os.listdir(self.data_folder)
        self.data_files = list(self.data_files)
        self.data_files.sort(key=self._order_key)

    def __iter__(self):
        instance = copy.deepcopy(self)
        return instance

    def __next__(self):
        try:
            while True:
                name = self.data_files.pop(0)
                path = self.data_folder + '/' + name
                obj = StateIterator(name=name, path=path, settings=self.settings)

                ok = True
                for fce in self.filters:
                    if not fce(obj):
                        ok = False
                        break
                if not ok:
                    continue
                return obj
        except IndexError:
            raise StopIteration()


# TESTS

if __name__ == '__main__':
    from IPython.core.debugger import set_trace
    import unittest
    import scipy
    import scipy.linalg
    import random


    def bisect_left(func_less, lo, hi):
        while lo < hi:
            mid = (lo + hi) // 2
            # Use __lt__ to match the logic in list.sort() and in heapq
            if func_less(mid):
                lo = mid + 1
            else:
                hi = mid
        return lo


    class TestTSS2(unittest.TestCase):
        def call_tss2(self, generation, points, evaluations, maximum_distance=None, maximum_elements=None,
                      mahalanobis_transf=None):
            if mahalanobis_transf is None:
                mahalanobis_transf = np.eye(generation.shape[1])
            if maximum_elements is None:
                maximum_elements = points.shape[0]
            if maximum_distance is None:
                maximum_distance = np.inf

            tss2 = TSS2(generation, mahalanobis_transf, maximum_distance, maximum_elements)
            mask, other_stuff = tss2(points, evaluations)
            return mask

        def test_maximum_elements(self):
            maximum_elements = 30

            for elements in range(100):
                generation = np.random.rand(10, 2)
                points = np.random.rand(elements, 2)
                evals = np.random.rand(elements, )

                mask = self.call_tss2(generation, points, evals, maximum_elements=maximum_elements)

                self.assertEqual(np.sum(mask), min(elements, maximum_elements))
                self.assertEqual(len(mask.shape), 1)
                self.assertEqual(mask.shape[0], elements)

        def test_maximum_distance_without_limit(self):
            maximum_distance = 0.2
            generation = np.array([
                [0.2, 0.5],
                [0.8, 0.5]
            ])
            points = np.random.rand(100000, 2)
            evals = np.random.rand(100000)

            mask = self.call_tss2(generation, points, evals, maximum_distance=maximum_distance)

            true_mask = np.minimum(
                np.sum(np.square(points - generation[0, :]), axis=1),
                np.sum(np.square(points - generation[1, :]), axis=1)
            ) <= maximum_distance ** 2

            diff = mask ^ true_mask
            self.assertFalse(np.any(diff))

        def test_maximum_distance_with_limit_nonoverlaping(self):
            maximum_distance = 0.2
            generation = np.array([
                [0.2, 0.5],
                [0.8, 0.5]
            ])

            points = np.random.rand(10000, 2)
            evals = np.random.rand(10000)

            dist_left = np.sqrt(np.sum(np.square(points - generation[0, :]), axis=1))
            dist_right = np.sqrt(np.sum(np.square(points - generation[1, :]), axis=1))

            avail = np.sum(np.logical_or(dist_left <= maximum_distance, dist_right <= maximum_distance))

            ord_left = np.argsort(dist_left)
            ord_right = np.argsort(dist_right)
            ord_left = ord_left[:np.sum(dist_left <= maximum_distance)]
            ord_right = ord_right[:np.sum(dist_right <= maximum_distance)]

            for elems in [random.randint(1, min(ord_left.size, ord_right.size) - 1) for _ in range(10)]:
                mask = self.call_tss2(generation, points, evals,
                                      maximum_elements=elems,
                                      maximum_distance=maximum_distance)

                cor = np.zeros_like(mask)
                cor[ord_left[:elems // 2]] = True
                cor[ord_right[:elems // 2]] = True

                if elems % 2 == 1:
                    if evals[ord_left[elems // 2]] < evals[ord_right[elems // 2]]:
                        cor[ord_left[elems // 2]] = True
                    else:
                        cor[ord_right[elems // 2]] = True

                diff = mask ^ cor
                self.assertFalse(np.any(diff))

        def test_maximum_distance_with_limit_overl(self):
            maximum_distance = 0.8
            generation = np.array([
                [0.2, 0.5],
                [0.8, 0.5]
            ])

            points = np.random.rand(10000, 2)
            evals = np.random.rand(10000)

            dist_left = np.sqrt(np.sum(np.square(points - generation[0, :]), axis=1))
            dist_right = np.sqrt(np.sum(np.square(points - generation[1, :]), axis=1))

            ord_left = np.argsort(dist_left)
            ord_right = np.argsort(dist_right)
            # outliers - remove
            ord_left = ord_left[:np.sum(dist_left <= maximum_distance)]
            ord_right = ord_right[:np.sum(dist_right <= maximum_distance)]

            max_com = min(ord_left.size, ord_right.size)

            # najdi takovy element, ktery je v obou pri dane vzdalenosti
            # ??? TODO -> min vzd.

            def func(mid):
                return not bool(len(set(iter(ord_left[:mid])).intersection(set(iter(ord_right[:mid])))))

            min_com = bisect_left(func, 0, max_com)
            assert ord_left[min_com - 1] in ord_right[:min_com] or ord_right[min_com - 1] in ord_left[:min_com]

            for elems in [random.randint(min_com, max_com) for _ in range(10)]:
                mask = self.call_tss2(generation, points, evals,
                                      maximum_distance=maximum_distance,
                                      maximum_elements=elems)
                n_mask = np.sum(mask)
                mask_arg = set(iter(np.argwhere(mask).flatten()))

                # number
                self.assertEqual(n_mask, elems)

                # distance
                selected_elements_distances = np.minimum(dist_left[mask], dist_right[mask])
                self.assertLessEqual(np.amax(selected_elements_distances), maximum_distance)

                # distance order
                def func(mid):
                    return mask_arg >= set(np.concatenate((ord_left[:mid], ord_right[:mid])))

                first_nonres = bisect_left(func, 0, len(mask_arg))

                # 1. check if all are ok / fist better (fittness) is ok
                almask = np.zeros_like(mask)
                # almask[
                #    np.unique(np.concatenate((ord_left[:first_nonres-1], ord_right[:first_nonres-1])))
                # ] = True
                almask[np.concatenate((ord_left[:first_nonres - 1], ord_right[:first_nonres - 1]))] = True

                assert np.sum(almask) <= elems

                g = almask ^ mask
                ng = np.sum(g)

                self.assertLessEqual(ng, 1)

                if ng > 0:
                    if evals[ord_left[first_nonres - 1]] < evals[ord_right[first_nonres - 1]]:
                        addi = ord_left[first_nonres - 1]
                    else:
                        addi = ord_right[first_nonres - 1]

                    self.assertTrue(g[addi])
                    self.assertFalse(np.any(g[:addi]))
                    self.assertFalse(np.any(g[addi + 1:]))

if __name__ == '__main__':
    class TestState(unittest.TestCase):
        def test_mahalanobis_distance_determ(self):
            C = np.array([[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]])
            C = scipy.linalg.sqrtm(C)
            # C is <cov mat.>^-1/2

            mu = np.array([0., 1., 0.])
            points = np.array([[1., 0., 0.]])

            # 1. test
            res = TSS2.mahalanobis_distance(points - mu, C)
            self.assertEqual(res.shape, (1,))
            res = res[0]
            self.assertAlmostEqual(res, 1.0)

            # 2. test
            points = np.array([[0, 2., 0.]])

            res = TSS2.mahalanobis_distance(points - mu, C)
            self.assertEqual(res.shape, (1,))
            res = res[0]
            self.assertAlmostEqual(res, 1.0)

            # 3. test
            points = np.array([
                [1., 0., 0.],
                [0., 2., 0.],
                [2., 0., 0.]
            ])

            res = TSS2.mahalanobis_distance(points - mu, C)
            self.assertEqual(res.shape, (3,))
            for res_or, res_my in zip([1, 1, 1.7320508075688772], res):
                self.assertAlmostEqual(res_or, res_my)

        def test_mahalanobis_multiple_dimensions(self):
            C = np.array([[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]])
            C = scipy.linalg.sqrtm(C)
            # C is <cov mat.>^-1/2

            mu = np.array([0., 1., 0.])
            differences = np.array([
                [
                    [1., 0., 0.],
                    [0., 2., 0.],
                    [2., 0., 0.]
                ], [
                    [1., 0., 0.],
                    [0., 2., 0.],
                    [1., 0., 0.]
                ], [
                    [2., 0., 0.],
                    [0., 2., 0.],
                    [2., 0., 0.]
                ]
            ]) - mu[np.newaxis, np.newaxis, :]

            res_ok = np.array([
                [1, 1, 1.7320508075688772],
                [1, 1, 1],
                [1.7320508075688772, 1, 1.7320508075688772]
            ])
            res = TSS2.mahalanobis_distance(differences, C)

            self.assertTrue(res.shape == (3, 3))
            for a, b in zip(res.flatten(), res_ok.flatten()):
                self.assertAlmostEqual(a, b)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False, verbosity=2)
