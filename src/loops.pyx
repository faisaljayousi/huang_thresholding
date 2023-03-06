#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as cnp

from libc.math cimport fabs
from libc.math cimport log as clog

from fused_numerics cimport np_anyint as any_int

cnp.import_array()


def _mu_loop(any_int[:] data, cnp.float64_t[:] mu0, cnp.float64_t[:] mu1, Py_ssize_t first_bin, Py_ssize_t last_bin):

    cdef:
        Py_ssize_t i
        cnp.float64_t sum_pix_0, sum_pix_1
        any_int num_pix_0, num_pix_1

    with nogil:

        num_pix_0 = 0
        sum_pix_0 = 0.
        num_pix_1 = 0
        sum_pix_1 = 0.

        for i in range(first_bin, 255):
            num_pix_0 += data[i]
            sum_pix_0 += i * data[i]
            mu0[i] = sum_pix_0 / num_pix_0

        for i in range(254, 1, -1):
            sum_pix_1 += i * data[i]
            num_pix_1 += data[i]
            mu1[i-1] = sum_pix_1 / num_pix_1


def _entropy_loop(any_int[:] data, cnp.float64_t length_inv, cnp.float64_t[:] mu0, cnp.float64_t[:] mu1, cnp.float64_t min_entropy, Py_ssize_t threshold):

    cdef:
        Py_ssize_t i, j
        cnp.float64_t entropy
        double length_inv_c = length_inv

    with nogil:

        for j in range(254):
            entropy = 0.0
            for i in range(j+1):
                mu_x = 1.0 / (1.0 + length_inv_c * fabs(i - mu0[j]))
                if mu_x > 1e-06 and mu_x < 0.999999:
                    entropy += data[i] * (-mu_x * clog(mu_x) - (1.0 - mu_x) * clog(1.0 - mu_x))

            for i in range(j + 1, 255):
                mu_x = 1.0 / (1.0 + length_inv_c * fabs(i - mu1[j]))
                if mu_x > 1e-06 and mu_x < 0.999999:
                    entropy += data[i] * (-mu_x * clog(mu_x) - (1.0 - mu_x) * clog(1.0 - mu_x))

            if entropy < min_entropy:
                min_entropy = entropy
                threshold = j

    return threshold, min_entropy


