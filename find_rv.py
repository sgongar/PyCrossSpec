#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Versions:
- 0.1: Initial release.

Description:
    This code finds the radial velocity of a target when supplied with data
    for the target and data for a standard object whose radial velocity
    is known.

Usage:
    Note: Data is not corrected for heliocentric velocities.

    Inputs:
        wv_obj, fx_obj, and sig_obj are arrays containing data for the
        wavelength, flux, and flux uncertainty of the target.
        wv_std, fx_std, and sig_std are arrays containing data for the
        wavelength, flux, and flux uncertainty of the standard.

    Example:
        >>> import find_rv
        >>> find_rv.radial_velocity(wv_obj, fx_obj, sig_obj, wv_std, fx_std,
                                    sig_std, rv_std, rv_std_err, obj_name,
                                    std_name)

Todo:
    * Unit tests.
    * FIX random noise generator


*GNU Terry Pratchett*
"""

from array import array
import math
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import scipy
import scipy.optimize as op
import scipy.ndimage


# Particular options
np.set_printoptions(threshold=np.inf)


def radial_velocity(wv_obj, fx_obj, sig_obj, wv_std, fx_std, sig_std, obj_name, std_name,
                    rv_std, rv_std_err, order, xcorr_width, cut, cutstart, cutend):
    """Calculates a radial velocity
    @param: wv_obj, wavelength_object
    @param: fx_obj, flux_object
    @param: sig_obj, uncertainty_object
    @param: wv_std, wavelength_standard
    @param: fx_std, flux_standard
    @param: sig_std, uncertainty_standard
    @param: object_name
    @param: rv_standard
    @param: error_rv_standard
    @param: spectral_order
    @param: xcorr_width
    @param: cut
    @param: cutstart
    @param: cutend

    @returns rv_measurement, error_rv_measurement:
    """
    plot_ = False
    noise_ = False
    # The more random iterations, the better... but it takes longer
    n_iter = 1500  # Was 1000

    # Step 1: Fix the spectra:
    # * Select only the region in which they overlap
    # * Make a new stretched wavelength array (for sub-pixel precision work)
    # * Interpolate the data onto the new wavelength array
    # * Remove large scale slopes so we only compare line and band features

    # Find where standard and object overlap 
    wv_min = max([min(wv_std), min(wv_obj)])
    wv_max = min([max(wv_std), max(wv_obj)])

    n_pix_std = len(wv_std)  # Number of pixels of studied spectrum

    # Creates ln standard wavelength array
    # The wavelength array only covers the overlap region.
    # Also, I'm folding the rebinning by 10 into this statement.
    acoef_std = (n_pix_std * 10 - 1) / (math.log(wv_max) - math.log(wv_min))
    bcoef_std = (n_pix_std * 10) - (acoef_std * math.log(wv_max))

    arr = np.arange(n_pix_std * 10) + 1  # Array of 10240 values [1, 2, ..., 10240]
    wv_ln_std = np.exp((arr - bcoef_std) / acoef_std)

    # AR 2012.1018: Find the conversion between pixels and velocity.
    # This will vary from instrument to instrument and spectral order to
    # spectral order, so we should preferentially calculate this based on the
    # actual input spectrum.
    # AR 2013.0422: Change the calculation to happen AFTER the corrected
    # wavelength scale has been made
    # Find the average pixel/spectrum offset
    # Note: even though it's called micron_per_pix, it will still work if the
    # wavelengths are angstroms instead (it really converts <wavelength unit>
    # to km/s)

    # Interpolate data onto same ln wavelength scale
    fx_interp_std = np.interp(wv_ln_std, wv_std, fx_std)
    fx_interp_obj = np.interp(wv_ln_std, wv_obj, fx_obj)

    # AR 2012.1018 Also need to rebin sig
    sig_interp_std = np.interp(wv_ln_std, wv_std, sig_std)
    # AR 2012.1018 Also need to rebin sig 
    sig_interp_obj = np.interp(wv_ln_std, wv_obj, sig_obj) 

    # Rebin Data ----------------------------
    wv_arr_std = np.asarray(wv_ln_std, dtype=float)
    fx_arr_obj = np.asarray(fx_interp_obj, dtype=float)
    fx_arr_std = np.asarray(fx_interp_std, dtype=float)
    sig_arr_obj = np.asarray(sig_interp_obj, dtype=float)
    sig_arr_std = np.asarray(sig_interp_std, dtype=float)

    # plt.title("Template (blue) and data (red)")
    # plt.plot(wv_arr_std, fx_arr_std, 'b.-')
    # plt.plot(wv_arr_std, fx_arr_obj, 'r.-')
    # plt.show()

    datalen = len(fx_arr_obj)

    # Cross correlation loop --------------------------------
    pix_shift = np.array([])  # initialize array for pixel shift values
    pix_width = np.zeros(n_iter)  # initialize array for pixel width values
    l = 0

    # using the xrange generator rather than making a full list saves memory
    while len(pix_shift) < n_iter:
        # prepare the randomized data
        # GETTING ARRAYS READY FOR CROSS CORRELATION

        # Randomize noise:
        # create gaussian distribution of random numbers b/t 1 and -1,
        # multiply err by numbers, add numbers to flux
        # I have drastically simplified the arrays here AR 2013.0319
        # AR 2013.0318: There was a problem, previously: noise was a fixed
        # value, not linked to the known error values

        # AR 2013.0321: Speed fix.  Rather than step through the array and
        #  generate one normally-distributed error value scaled to the SNR at
        #  that point, I will generate an array of normally-distributed error
        #  values scaled to 1, and then multiply by the SNR:
        #  One array generation, one array multiplication.

        if noise_:
            rand_dist = np.random.normal(loc=0.0, scale=1.0, size=datalen)
            rand_dist2 = np.random.normal(loc=0.0, scale=1.0, size=datalen)

            fx_temp_obj = np.asarray(fx_arr_obj + rand_dist * sig_arr_obj)
            fx_temp_std = np.asarray(fx_arr_std + rand_dist2 * sig_arr_std)
        else:
            rand_dist = np.random.normal(loc=0.0, scale=1.0, size=datalen)
            rand_dist2 = np.random.normal(loc=0.0, scale=1.0, size=datalen)

            fx_temp_obj = np.asarray(fx_arr_obj + rand_dist * sig_arr_obj)
            fx_temp_std = np.asarray(fx_arr_std + rand_dist2 * sig_arr_std)

        mean_obj = np.mean(fx_temp_obj)
        mean_std = np.mean(fx_temp_std)
        stddev_obj = np.std(fx_temp_obj, ddof=1)
        stddev_std = np.std(fx_temp_std, ddof=1)

        # Regularize data (subtract mean, divide by std dev) (Should definitely
        # be done AFTER noise was added)
        # For image-processing applications in which the brightness of the
        # image and template can vary due to lighting and exposure conditions,
        # the images can be first normalized. This is typically done at every
        # step by subtracting the mean and dividing by the standard deviation.
        fx_reg_temp_obj = fx_temp_obj - mean_obj
        fx_reg_temp_obj = fx_reg_temp_obj / stddev_obj
        
        fx_reg_temp_std = fx_temp_std - mean_std
        fx_reg_temp_std = fx_reg_temp_std / stddev_std

        # plt.title("Template (blue) and data (red)")
        # plt.plot(wv_arr_std, fx_reg_temp_obj, 'b.-')
        # plt.plot(wv_arr_std, fx_reg_temp_std, 'r.-')
        # plt.grid(True)
        # plt.show()

        # CROSS CORRELATION
        # compute the cross-correlation between the two spectra
        ycorr = np.correlate(fx_reg_temp_obj, fx_reg_temp_std, mode='full')

        ## slight smoothing AR 2013.0315
        # ycorr = scipy.ndimage.filters.gaussian_filter1d(ycorr, 11)

        # create the x offset axis (same length as ycorr, with 0 in the MIDDLE)
        length = len(ycorr)  # Total length of correlation data
        xcorr = np.arange(length) - length // 2
        # Select a tiny piece around the maximum to fit with a gaussian.
        xmid = np.argmax(ycorr)  # x position of y cross-correlation max value
        ymax = np.max(ycorr)  # y cross-correlation max value

        """
        plt.title("Template (blue) and data (red)")
        plt.plot(wv_arr_std, fx_reg_temp_obj, 'b.-')
        plt.plot(wv_arr_std, fx_reg_temp_std, 'r.-')
        plt.plot(wv_arr_std[int(xmid/2)], 4, 'bx')
        plt.grid(True)
        plt.show()
        """

        # Now take just the portion of the array that matters
        xcorr_min = int(xmid - xcorr_width)  # xcorr_width = 200 pixels
        xcorr_max = int(xmid + xcorr_width)  # xcorr_width = 200 pixels
        # Isolate section of array with gaussian
        ycorr1 = ycorr[xcorr_min:xcorr_max]
         # isolate the same section of the pixel range
        pixel_width = 200  # Was 50
        xcorr1 = xcorr[xcorr_min:xcorr_max]
        ycorr2 = ycorr[xcorr_min - pixel_width:xcorr_max + pixel_width]
        xcorr2 = xcorr[xcorr_min - pixel_width:xcorr_max + pixel_width]

        mean = xcorr[xmid]
        # set up initial values for chi2
        sig = 10
        sky = np.min(ycorr1) / 1.2
        sky2 = (ycorr1[-1] - ycorr1[0]) / (xcorr1[-1] - xcorr1[0])
        lnamp = np.log(ymax / 1.2 - sky)  # guess some values

        amp = np.exp(lnamp)
        sig2 = sig ** 2
        # suggestion from D. Hogg 12/15/12: Add extra linear feature to fit.
        # suggestion from D. Hogg 12/15/12: operate on ln(amp) so that the
        # amplitude CANNOT be negative.
        def chi2(p):  # define gaussian function for fitting
            sig2 = p[2] ** 2
            m = (np.exp(p[0]) * np.exp(-0.5 * (xcorr1 - p[1]) ** 2 / sig2)) + p[3] + p[4] * xcorr1
            return (ycorr1 - m)

        popt, ier = op.leastsq(chi2, [lnamp, mean, sig, sky, sky2])
        lnamp, mean, sig, sky, sky2 = popt

        amp = np.exp(lnamp)

        l += 1
        if (cut == 0) | (mean > np.float(cutstart)) & (mean < np.float(cutend)):
            pix_shift = np.append(pix_shift, mean)

        # plt.title("Template (blue) and data (red)")
        # plt.plot(wv_arr_std, fx_reg_temp_obj, 'b.-')
        # plt.plot(wv_arr_std, fx_reg_temp_std, 'r.-')
        # plt.plot(wv_arr_std[int(xmid/2)], -4, 'bx')
        # plt.plot(wv_arr_std[int(xmid/2)] + mean/2, -4, 'rx')
        # plt.grid(True)
        # plt.show()

    # 4. Find the RV
    # All 5000 rv fits have been calculated and stored in arrays
    # 4a. Cut out outlier RVs. Useful if the cross-correlation produces
    #     occasional bad results. Use cutstart and cutend to force the code
    #     to only fit a gaussian to a certain region. Don't over-use this to
    #     force the result you want, though.
    # 4b. Compute the mean pixel shift and pixel shift uncertainty.
    # 4c. Convert pixel shift into RV
    # 4d. Shift the wavelength array appropriately - all lines should now line up.

    # Turn the list of pixel shifts into a numpy array
    pix_shift = np.asarray(pix_shift)

    # 4a. Cut out outliers from the pixel shift
    if cut == 1:
        pix_shift = pix_shift[np.where((pix_shift > np.float(cutstart)) & (pix_shift < np.float(cutend)))]

    # 4b. Compute the mean pixel shift (rv value) and pixel shift uncertainty
    # (RV uncertainty).
    mu = np.mean(pix_shift)
    sigma = np.std(pix_shift, ddof=1)

    # 4c. Transform pixel shift to shift in radial velocity

    # AR 2013.0423: The actually appropriate method requires a speed-of-light
    # correction. This works for both angstroms and microns.
    # 2.99792458 * 10 ** 5 -> ligthspeed
    # acoef_std ->
    # mu -> pixel shift
    # sigma -> error en pixel shift
    rv_meas = (2.99792458 * 10 ** 5 * mu) / acoef_std
    rv_meas_err = (2.99792458 * 10 ** 5 * sigma) / acoef_std

    return rv_meas, rv_meas_err
