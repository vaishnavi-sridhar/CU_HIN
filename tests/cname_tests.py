import unittest
import sys

import scipy.sparse

sys.path.append('../src/')
from src import hin
from scipy.sparse import csc_matrix, csr_matrix
import numpy as np
from src import PathSim


class TestCNameImplementation(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_cname_matrix(self):
        domain2index_test = dict()
        domain2index_test['mdw-efz.ms-acdc.office.com'] = 0
        domain2index_test['outlook.ms-acdc.office.com'] = 1
        domain2index_test['libraries.colorado.edu'] = 2
        domain2index_test['config.teams.trafficmanager.net'] = 3
        domain_pairs_test = [(0, 1), (0, 0), (1, 1)]

        # expected value
        row = np.array([0, 0, 1, 1])
        col = np.array([0, 1, 0, 1])
        data = np.array([1, 1, 1, 1])
        exp_cname_sparse = csc_matrix((data, (row, col)), shape=(4, 4))
        act_cname_sparse = hin.generate_cname_csr(domain2index_test, domain_pairs_test)
        print("Expected CName matrix value:", exp_cname_sparse)
        print("Expected CName matrix value:", act_cname_sparse)
        self.assertEqual(exp_cname_sparse.shape, act_cname_sparse.shape)
        np.ndarray.__eq__(exp_cname_sparse.toarray(), act_cname_sparse.toarray()).all()

    def test_pathsim_computation(self):
        row = np.array([0, 0, 2, 2])
        col = np.array([0, 2, 0, 2])
        data = np.array([1, 1, 1, 1])
        input_m_test = csr_matrix((data, (row, col)), shape=(3, 3))
        exp_data = np.array([1 / 4, 1 / 4, 1 / 4, 1 / 4])
        exp_sim_matrix = csr_matrix((exp_data, (row, col)), shape=(3, 3))
        actual_sim_matrix = PathSim.PathSim(input_m_test)
        print("Expected path-sim output value:", exp_sim_matrix)
        print("Actual path-sim output value:", actual_sim_matrix)
        self.assertEqual(exp_sim_matrix.shape, actual_sim_matrix.shape)
        np.ndarray.__eq__(exp_sim_matrix.toarray(), actual_sim_matrix.toarray()).all()


if __name__ == '__main__':
    unittest.main()
