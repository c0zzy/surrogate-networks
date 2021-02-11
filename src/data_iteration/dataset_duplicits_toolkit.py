#!/usr/bin/env python
'''
module exports: 
    duplicit_mask
    duplicit_mask_with_hot_elements
    move_indices_given_boolean_mask

    it's all
'''

import numpy as np

def _sort_array(array):
    ar = tuple(map(lambda x: x[:,0], reversed(np.hsplit(array, array.shape[1]))))
    return np.lexsort(ar)

def _sort_matching(matrixA, matrixB):
    # optimization II.
    assert len(matrixA.shape) == len(matrixB.shape) == 2
    assert matrixA.shape[1] == matrixB.shape[1]
    
    ind_A_sort = _sort_array(matrixA)
    sorted_matrix_A = matrixA[ind_A_sort, :]
    
    ind_B_sort = _sort_array(matrixB)
    sorted_matrix_B = matrixB[ind_B_sort, :]
    
    iA, iAmax = 0, matrixA.shape[0]
    iB, iBmax = 0, matrixB.shape[0]
    
    if iAmax == 0 or iBmax == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((iAmax,), dtype=np.bool)
    
    matches = np.empty(shape=(iAmax,), dtype=np.int32)
    
    A = list(sorted_matrix_A[iA,:])
    B = list(sorted_matrix_B[iB,:])
    
    while True:
        if A > B:
            iB += 1
            if iB == iBmax:
                matches[iA:] = -1
                break
            B = list(sorted_matrix_B[iB,:])
        elif A == B:
            matches[iA] = iB
            iA += 1
            if iA == iAmax:
                break
            A = list(sorted_matrix_A[iA,:])
        else: # sorted_matrix_A[iA,:] < sorted_matrix_B[iB,:]:
            matches[iA] = -1
            iA += 1
            if iA == iAmax:
                break
            A = list(sorted_matrix_A[iA,:])
                
    # change A order back
    #matches[ind_A_sort] = matches
    A = np.empty_like(matches)
    A[ind_A_sort] = matches
    matches = A
    
    # where are
    positive_matches = matches != -1
    matches = matches[positive_matches]
    
    matches = ind_B_sort[matches]
    #matches = np.stack([np.arange(len(matches)), matches], axis=1)
    
    return matches, positive_matches
    
        

def duplicit_mask(points, pass_additional_info=False):
    # Returns boolean mask:
    #   True if an element in his place for the first time (in the sequence)
    #   False - already encountered this element
    
    # data[ui] = up
    # up[ui] = data
    assert len(points.shape) == 2
    
    if points.shape[0] == 0:
        result = np.zeros((0,), dtype=np.bool)
        if pass_additional_info:
            return result, (
                np.zeros((0,points.shape[1]), dtype=points.dtype),
                np.zeros((0,), np.int64),
                np.zeros((0,), np.int64)
            )
        else:
            return result
    
    unique_points, unique_index, unique_inverse =             np.unique(points, axis=0,             return_index=True, return_inverse=True)
    
    assert(unique_points.shape[0] == unique_index.shape[0])
    assert(unique_inverse.shape[0] == points.shape[0])
    
    ira = np.arange(points.shape[0])
    first_occ_of_el = unique_index[unique_inverse]
    
    result = ira <= first_occ_of_el
    
    assert len(result.shape) == 1
    
    if pass_additional_info:
        info = (unique_points, unique_index, unique_inverse)
        return result, info
    else:
        return result


def duplicit_mask_with_hot_elements(points, hot):
    '''
        Same as duplicit_mask but only checks if 'hot' elements are previously encountered
        Check the test for more info
        
        points - (N,Dim) of np.* type
        hot - (N,) of np.bool type
    '''
    assert len(hot.shape) == 1
    assert len(points.shape) == 2
    assert hot.shape[0] == points.shape[0]
    
    ind = np.arange(points.shape[0])
    
    hot_points, hot_ind = points[hot, ...], ind[hot]
    nothot_points, nothot_ind = points[~hot, ...], ind[~hot]
    
    hot_mask, info = duplicit_mask(hot_points, pass_additional_info=True)
    (hot_unique, hot_index, hot_inverse) = info
    
    # SEARCH MATCHES: (NH_data, UH_data)
    hot_points_real_index = hot_ind[hot_index]
    
    matches, matches_positive = _sort_matching(nothot_points, hot_unique)
    
    if matches.shape[0]:
        # index to unique hot elements
        #matches = np.argmax(matches, axis=1)
        # first occ. of elem.
        matches_first = hot_points_real_index[matches]
        # where are nontho elements (indexes)
        matches_current = nothot_ind[matches_positive]
        # first occ of hot element is after...
        matches_result = matches_first >= matches_current
    else:
        matches_result = np.zeros((0,), dtype=np.bool)
        matches_current = np.zeros((0,), dtype=np.int64)
    
    # PUTTING IT TOGETHER... 
    result = np.ones(shape=(points.shape[0],), dtype=np.bool)
    # remove cold repetetions
    result[matches_current] = matches_result
    # remove hot repetetions
    result[hot_ind] = hot_mask
    return result


def move_indices_given_boolean_mask(mask, indices):
    assert len(mask.shape) == 1
    assert len(indices.shape) == 1
    
    index_removed = np.arange(len(mask))[~mask]
    b = np.sum(np.less_equal(
            index_removed[np.newaxis, ...], indices[..., np.newaxis]
        ), axis=-1)
    
    return np.maximum(indices - b, 0)



def _0_duplicit_mask_with_hot_elements(points, hot):
    # Original implementation - it's slow
    '''
        Same as duplicit_mask but only checks if 'hot' elements are previously encountered
        Check the test for more info
        
        points - (N,Dim) of np.* type
        hot - (N,) of np.bool type
    '''
    assert len(hot.shape) == 1
    assert len(points.shape) == 2
    assert hot.shape[0] == points.shape[0]
    
    ind = np.arange(points.shape[0])
    
    hot_points, hot_ind = points[hot, ...], ind[hot]
    nothot_points, nothot_ind = points[~hot, ...], ind[~hot]
    
    hot_mask, info = duplicit_mask(hot_points, pass_additional_info=True)
    (hot_unique, hot_index, hot_inverse) = info
    
    # SEARCH MATCHES: (NH_data, UH_data)
    hot_points_real_index = hot_ind[hot_index]
    
    matches = np.all(np.equal(
        hot_unique[np.newaxis, ...],
        nothot_points[:, np.newaxis, ...]
        ), axis=2)
    
    matches_positive = np.any(matches, axis=1)
    matches = matches[matches_positive, ...]
    
    if matches.shape[0]:
        matches = np.argmax(matches, axis=1)
        matches_first = hot_points_real_index[matches]

        matches_current = nothot_ind[matches_positive]

        matches_result = matches_first >= matches_current
    else:
        matches_result = np.zeros((0,), dtype=np.bool)
        matches_current = np.zeros((0,), dtype=np.int64)
    
    # PUTTING IT TOGETHER... 
    result = np.ones(shape=(points.shape[0],), dtype=np.bool)
    result[matches_current] = matches_result
    result[hot_ind] = hot_mask
    return result


if __name__ == '__main__':
    import unittest
    import random
    
    class TestMoveIndeces(unittest.TestCase):
        def test_move_indeces(self):
            mask    = np.array([0,1,0,1,1,0,0], dtype=np.bool)
            indices = np.array([0,1,2,3,4,5,6], dtype=np.int64)
            gt      = np.array([0,0,0,1,2,2,2], dtype=np.int64)
            
            new_indices = move_indices_given_boolean_mask(mask, indices)
            
            self.assertEqual(gt.shape, new_indices.shape)
            
            for t,n in zip(new_indices, gt):
                self.assertEqual(t,n)
                
            mask    = np.array([1,1,0,1,0,1,1], dtype=np.bool)
            indices = np.array([0,1,2,3,4,5,6], dtype=np.int64)
            gt      = np.array([0,1,1,2,2,3,4], dtype=np.int64)
            
            new_indices = move_indices_given_boolean_mask(mask, indices)
            
            self.assertEqual(gt.shape, new_indices.shape)
            
            for t,n in zip(new_indices, gt):
                self.assertEqual(t,n)



if __name__ == '__main__':
    class TestDuplicitMask(unittest.TestCase):
        def test_easy(self):
            data = np.array([
                [1, 2],
                [1, 4],
                [1, 2],
                [3, 3],
                [1, 2],
                [2, 2],
                [3, 3]
            ])
            vysledek = np.array([True, True, False, True, False, True, False])
            v = duplicit_mask(data)
            self.assertTrue(np.all(np.equal(v, vysledek)))

        def test_random(self):
            data = np.random.rand(100,3)
            g = duplicit_mask(data)
            self.assertTrue(np.all(g == True))

        def test_ones(self):
            data = np.ones((100,3))
            g = duplicit_mask(data)
            self.assertTrue(np.all(g[0] == True))
            self.assertTrue(np.all(g[1:] == False))

        def test_two_parts(self):
            data = np.random.rand(100,3)
            data_full = np.concatenate((data, data), axis=0)
            g = duplicit_mask(data)

            self.assertTrue(np.all(g[:100] == True))
            self.assertTrue(np.all(g[100:] == False))

        def test_two_parts_2(self):
            data = np.random.rand(100,3)
            data_full = np.concatenate((data, data), axis=0)
            s = np.arange(data_full.shape[0])
            np.random.shuffle(s)
            data_full = data_full[s, ...]

            g = duplicit_mask(data)
            self.assertEqual(np.sum(g), 100)

        def test_empty(self):
            data = np.random.random((0,3))
            g = duplicit_mask(data)
            self.assertTrue(g.shape[0] == 0)
            self.assertTrue(len(g.shape) == 1)


if __name__ == '__main__':
    class TestDuplicitMaskWithHotElements(unittest.TestCase):
        def test_only_hot(self):
            data = np.array([
                [1, 2], [1, 4], [1, 2], [3, 3], [1, 2], [2, 2], [3, 3]
            ])
            hotness = np.ones((data.shape[0],), dtype=np.bool)

            vysledek = np.array([True, True, False, True, False, True, False])
            v = duplicit_mask_with_hot_elements(data, hotness)
            self.assertTrue(np.all(np.equal(v, vysledek)))

        def test_only_cold(self):
            data = np.array([
                [1, 2], [1, 4], [1, 2], [3, 3], [1, 2], [2, 2], [3, 3]
            ])
            hotness = np.zeros((data.shape[0],), dtype=np.bool)

            v = duplicit_mask_with_hot_elements(data, hotness)
            self.assertTrue(np.all(v == True))

        def test_custom(self):
            # [1,2] - N T N T
            # [1,4] - N N
            # [3,3] - T T
            # [2,2] - T T
            data = np.array([
                [1, 2], [1, 4], [1, 2], [3, 3], [1, 2], [2, 2], [3, 3], [2, 2], [1, 2], [1, 4]
            ])
            hotness = np.array([
                False, False, True, True, False, True, True, True, True, False
            ])
            result = np.array([
                True, True, True, True, False, True, False, False, False, True
            ])
            v = duplicit_mask_with_hot_elements(data, hotness)
            self.assertTrue(np.all(v == result))


        def test_random(self):
            ELEMENTS = 10000
            DIMENSIONS = 3

            COLDS_BEFORE_MIN = 0
            COLDS_BEFORE_MAX = 2
            HOTS_MIN = 0 
            HOTS_MAX = 3 
            COLDS_AFTER_MIN = 0
            COLDS_AFTER_MAX = 2

            N = ELEMENTS*(COLDS_AFTER_MAX + COLDS_BEFORE_MAX + HOTS_MAX)
            uniq_elems = np.random.rand(ELEMENTS, DIMENSIONS)

            empty_indexes = list(range(N))
            random.shuffle(empty_indexes)

            used = np.zeros((N,), dtype=np.bool)
            data = np.empty((N,DIMENSIONS))
            hotness = np.zeros((N,), dtype=np.bool)
            result = np.ones((N,), dtype=np.bool)

            for elem in uniq_elems:
                c_before = random.randint(COLDS_BEFORE_MIN, COLDS_BEFORE_MAX)
                c_after = random.randint(COLDS_AFTER_MIN, COLDS_AFTER_MAX)
                hots = random.randint(HOTS_MIN, HOTS_MAX)
                
                if hots == 0:
                    c_after = 0
                size = c_before + c_after + hots

                indexes = empty_indexes[:size]
                empty_indexes = empty_indexes[size:]
                uniq_elems = uniq_elems[size:]
                #del uniq_elems[:size]

                indexes.sort()

                for cbi in indexes[:c_before]:
                    # colds
                    used[cbi] = True
                    data[cbi, ...] = elem
                    hotness[cbi] = False
                    result[cbi] = True
                indexes = indexes[c_before:]

                if hots:
                    # first_hot
                    used[indexes[0]] = True
                    data[indexes[0], ...] = elem
                    hotness[indexes[0]] = True
                    result[indexes[0]] = True

                    indexes = indexes[1:]

                random.shuffle(indexes)
                
                if hots:
                    assert len(indexes) == hots - 1 + c_after
                else:
                    assert len(indexes) == c_after
                
                for caf in indexes[:c_after]:
                    # colds after
                    used[caf] = True
                    data[caf, ...] = elem
                    hotness[caf] = False
                    result[caf] = False
                    pass
                indexes = indexes[c_after:]
                #del indexes[:c_after]

                assert len(indexes) == hots - 1 or (hots == 0 and len(indexes) == 0)
                for other_hots in indexes:
                    # other hots
                    used[other_hots] = True
                    data[other_hots, ...] = elem
                    hotness[other_hots] = True
                    result[other_hots] = False
                    pass

            data = data[used, ...]
            hotness = hotness[used]
            result = result[used]

            tested_res = duplicit_mask_with_hot_elements(data, hotness)

            self.assertTrue( np.all(tested_res == result) )


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False, verbosity=2)   

