# -*- coding: utf-8 -*-
# @Date    : 2017/11/04
# @Author  : Vitaly
import numpy as np


class Simplex(object):
    # default is solve min LP, if want to solve max lp,should * -1
    def __init__(self, obj, max_mode=False):
        self.mat, self.max_mode = np.array(
            [[0] + obj]) * (-1 if max_mode else 1), max_mode

    def add_constraint(self, a, b):
        self.mat = np.vstack([self.mat, [b] + a])

    def _simplex(self, mat, B, m, n):
        while mat[0, 1:].min() < 0:
            # use Bland's method to avoid degeneracy. use mat[0].argmin() ok?
            col = np.where(mat[0, 1:] < 0)[0][0] + 1
            row = np.array([mat[i][0] / mat[i][col] if mat[i][col] > 0 else 0x7fffffff for i in
                            range(1, mat.shape[0])]).argmin() + 1  # find the theta index
            if mat[row][col] <= 0:
                return None  # the theta is ∞, the problem is unbounded
            self._pivot(mat, B, row, col)
        return mat[0][0] * (1 if self.max_mode else -1), {B[i]: mat[i, 0] for i in range(1, m) if B[i] < n}

    def _pivot(self, mat, B, row, col):
        mat[row] /= mat[row][col]
        ids = np.arange(mat.shape[0]) != row
        # for each i!= row do: mat[i]= mat[i] - mat[row] * mat[i][col]
        mat[ids] -= mat[row] * mat[ids, col:col + 1]
        B[row] = col

    def solve(self):
        m, n = self.mat.shape  # m - 1 is the number slack variables we should add
        temp, B = np.vstack([np.zeros((1, m - 1)), np.eye(m - 1)]
                            ), list(range(n - 1, n + m - 1))  # add diagonal array
        mat = self.mat = np.hstack([self.mat, temp])  # combine them!
        if mat[1:, 0].min() < 0:  # is the initial basic solution feasible?
            row = mat[1:, 0].argmin() + 1  # find the index of min b
            # set first row value to zero, and store the previous value
            temp, mat[0] = np.copy(mat[0]), 0
            mat = np.hstack([mat, np.array([1] + [-1] * (m - 1)).reshape((-1, 1))])
            self._pivot(mat, B, row, mat.shape[1] - 1)
            if self._simplex(mat, B, m, n)[0] != 0:
                return None  # the problem has no answer

            if mat.shape[1] - 1 in B:  # if the x0 in B, we should pivot it.
                self._pivot(mat, B, B.index(
                    mat.shape[1] - 1), np.where(mat[0, 1:] != 0)[0][0] + 1)
            # recover the first line
            self.mat = np.vstack([temp, mat[1:, :-1]])
            for i, x in enumerate(B[1:]):
                self.mat[0] -= self.mat[0, x] * self.mat[i + 1]
        return self._simplex(self.mat, B, m, n)
