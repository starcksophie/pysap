# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import

import warnings

# Package import
import pysap
import pysap.base.utils as utils

from pysap.base.transform import MetaRegister  # for the metaclass
from pysap.base import image


try:
    import pysparse
except ImportError:  # pragma: no cover
    warnings.warn("Sparse2d python bindings not found, use binaries.")
    pysparse = None

# Third party import
import numpy as np
import matplotlib.pyplot as plt


class Filter():
    """ Define the structure that will be used to store the filter result.
    """
    def __init__(self, **kwargs):
        """ Define the filter.

        Parameters
        ----------

        type_of_filtering: int
        coef_detection_method: int
        type_of_multiresolution_transform: float
        type_of_filters: float
        type_of_non_orthog_filters: float
        sigma_noise: float
        type_of_noise: int
        number_of_scales: int
        iter_max: double
        epsilon: float
        verbose: Boolean
        tab_n_sigma: ndarray
        suppress_isolated_pixels: Boolean

        """
        self.data = None
        self.flt = pysparse.MRFilters(**kwargs)

    def filter(self, data):
        """ Execute the filter operation.

        Parameters
        ----------
        data: ndarray
            the input data.
        """
        self.data = pysap.Image(data=self.flt.filter(data))

    def show(self):  # pragma: no cover
        """ Show the filtered data.
        """
        if self.data is None:
            raise AttributeError("The data must be filtered first !")
        self.data.show()


class Deconvolve():
    """ Define the structure that will be used to
        store the deconvolution result.
    """
    def __init__(self, **kwargs):
        """ Define the deconvolution.

        Parameters
        ----------

        type_of_deconvolution: int
        type_of_multiresolution_transform: int
        type_of_filters: int
        number_of_undecimated_scales: int
        sigma_noise: float
        type_of_noise: int
        number_of_scales: int
        nsigma: float
        number_of_iterations: int
        epsilon: float
        psf_max_shift: bool
        verbose: bool
        optimization: bool
        fwhm_param: float
        convergence_param: float
        regul_param: float
        first_guess: string
        icf_filename: string
        rms_map: string
        kill_last_scale: bool
        positive_constraint: bool
        keep_positiv_sup: bool
        sup_isol: bool
        pas_codeur: float
        sigma_gauss: float
        mean_gauss: float
        """
        self.data = None
        self.deconv = pysparse.MRDeconvolve(**kwargs)

    def deconvolve(self, img, psf):
        """ Execute the deconvolution operation.

        Parameters
        ----------
        img: ndarray
            the input image.
        psf: ndarray
            the input psf
        """
        self.data = pysap.Image(data=self.deconv.deconvolve(img, psf))

    def show(self):  # pragma: no cover
        """ Show the deconvolved data.
        """
        if self.data is None:
            raise AttributeError("The data must be deconvolved first !")
        self.data.show()


class MR2D1D():
    """ Define the structure that will be used to
        store the MR2D1D transform and reconstruction
        results.
    """
    def __init__(self, **kwargs):
        """ Define the transform.

        Parameters
        ----------

        type_of_transform_2d: int 
              1: linear wavelet transform: a trous algorithm 
              2: bspline wavelet transform: a trous algorithm 
              3: wavelet transform in Fourier space 
              4: morphological median transform 
              5: morphological minmax transform 
              6: pyramidal linear wavelet transform 
              7: pyramidal bspline wavelet transform 
              8: pyramidal wavelet transform in Fourier space: algo 1 (diff. between two resolutions) 
              9: Meyer's wavelets (compact support in Fourier space) 
              10: pyramidal median transform (PMT) 
              11: pyramidal laplacian 
              12: morphological pyramidal minmax transform 
              13: decomposition on scaling function 
              14: Mallat's wavelet transform (7/9 filters) 
              15: Feauveau's wavelet transform 
              16: Feauveau's wavelet transform without undersampling 
              17: Line Column Wavelet Transform (1D+1D) 
              18: Haar's wavelet transform 
              19: half-pyramidal transform 
              20: mixed Half-pyramidal WT and Median method (WT-HPMT) 
              21: undecimated diadic wavelet transform (two bands per scale) 
              22: mixed WT and PMT method (WT-PMT) 
              23: undecimated Haar transform: a trous algorithm (one band per scale) 
              24: undecimated (bi-) orthogonal transform (three bands per scale) 
              25: non orthogonal undecimated transform (three bands per scale) 
              26: Isotropic and compact support wavelet in Fourier space 
              27: pyramidal wavelet transform in Fourier space: algo 2 (diff. between the square of two resolutions) 
              28: Fast Curvelet Transform 
              29: Wavelet transform via lifting scheme 
              30: 5/3 on line and 4/4 on column 
              31: 4/4 on line and 5/3 on column 
             default is bspline wavelet transform: a trous algorithm
        filter_1d: int
              1: Biorthogonal 7/9 filters 
              2: Daubechies filter 4 
              3: Biorthogonal 2/6 Haar filters 
              4: Biorthogonal 2/10 Haar filters 
              5: Odegard 9/7 filters 
              6: 5/3 filter 
              7: Battle-Lemarie filters (2 vanishing moments) 
              8: Battle-Lemarie filters (4 vanishing moments) 
              9: Battle-Lemarie filters (6 vanishing moments) 
              10: User's filters 
              11: Haar filter 
              12: 3/5 filter 
              13: 4/4 Linar spline filters 
              14: Undefined sub-band filters 
             default is Biorthogonal 7/9 filters
        normalize: bool
        verbose: bool
        NbrScale2d: int
            number of scales 2d
        Nbr_Plan: int
            number of scales 1d

        """
        self.cube = None
        self.recons = None
        self.trf = pysparse.MR2D1D(**kwargs)

    def transform(self, data):
        """ Execute the transform operation.

        Parameters
        ----------
        data: ndarray
            the input data.
        """
        self.cube = self.trf.transform(data)

    def reconstruct(self, data):
        """ Execute the reconstructiom operation.

        Parameters
        ----------
        data: ndarray
            the input data.
        """
        self.recons = self.trf.reconstruct(data)
    
    def info(self):
        self.trf.info()
