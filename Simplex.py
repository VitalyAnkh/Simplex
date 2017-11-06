# -*- coding: utf-8 -*-
# @Date    : 2017/11/04
# @Author  : Vitaly

import numpy as np


class Simplex(object):
    def __init__(self, obj, max_mode=False):
        self.max_mode = max_mode  # default is solve min LP, if want to solve max lp,should * -1
        self.mat = np.array([[0] + obj]) * (-1 if max_mode else 1)

    def add_constraint(self, a, b):
        self.mat = np.vstack([self.mat, [b] + a])

    def solve(self):
        m, n = self.mat.shape  # m - 1 is the number slack variables we should add
        temp, B = np.vstack([np.zeros((1, m - 1)), np.eye(m - 1)]
                            ), list(range(n - 1, n + m - 1))  # add diagonal array
        mat = self.mat = np.hstack([self.mat, temp])  # combine them!
        while mat[0, 1:].min() < 0:
            # use Bland's method to avoid degeneracy. use mat[0].argmin() ok?
            col = np.where(mat[0, 1:] < 0)[0][0] + 1
            row = np.array([mat[i][0] / mat[i][col] if mat[i][col] > 0 else 0x7fffffff for i in
                            range(1, mat.shape[0])]).argmin() + 1  # find the theta index
            if mat[row][col] <= 0:
                return None  # the theta is ∞, the problem is unbounded
            mat[row] /= mat[row][col]
            ids = np.arange(mat.shape[0]) != row
            # for each i!= row do: mat[i]= mat[i] - mat[row] * mat[i][col]
            mat[ids] -= mat[row] * mat[ids, col:col + 1]
            B[row] = col
        return mat[0][0] * (1 if self.max_mode else -1), {B[i]: mat[i, 0] for i in range(1, m) if B[i] < n}


t = Simplex([-5,-3,0,0])
t.add_constraint([1,5,1,0], 10)
t.add_constraint([1,0,0,1], 2)
print(t.solve())
print(t.mat)
