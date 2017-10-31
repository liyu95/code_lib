#!/usr/bin/env python

import numpy as np
from scipy.special import comb

def chebfit(deg):
	x = np.arange(-1, 1, 0.001)
	y = np.sign(x)
	y [y == -1] =0
	c = np.polynomial.chebyshev.chebfit(x, y, deg)
	return c

def ComputePoly(coef_poly, x):
    res = 0
    for idx, c in enumerate(coef_poly):
        res += c * np.power(x, idx)
    return res

def compute_alpha(degree):
	cheb_c = chebfit(degree)
	poly_c = np.polynomial.chebyshev.cheb2poly(cheb_c)
	# print poly_c
	alpha = np.zeros((degree+1, degree+1))
	for k in xrange(degree+1):
		for l in xrange(k+1):
			alpha[k,l] = poly_c[k] * comb(k,l) * np.power(-1, k-l)
	return alpha


# D0, D1 are the samples with true label 0 and 1 respectively
# index in this case
def auc_loss(Y_pred, D0, D1, degree = 7):
	alpha = compute_alpha(degree)
	# print alpha
	auc_poly = 0
	for k in range(degree+1):
		for l in range(k+1):
			auc_poly += alpha[k,l] * np.sum( np.power( Y_pred[D1], l) ) * np.sum( np.power( Y_pred[D0], k-l ) )
	auc_poly /= len(D0) * len(D1) * 1.0
	return -auc_poly
