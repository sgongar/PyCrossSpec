#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Versions:
- 0.1: Initial release.

Todo:
    * Unit tests.
    *

Columns definitions:
    * name = 'SPORDER'; format = '1I'; null = -32767; disp = 'I11'
    spectral order
    * name = 'NELEM'; format = '1I'; null = -32767; disp = 'I11'
    * name = 'WAVELENGTH'; format = '1024D'; unit = 'Angstroms'; disp = 'G25.16'
    * name = 'GROSS'; format = '1024E'; unit = 'Counts/s'; disp = 'G15.7'
    * name = 'BACKGROUND'; format = '1024E'; unit = 'Counts/s'; disp = 'G15.7'
    * name = 'NET'; format = '1024E'; unit = 'Counts/s'; disp = 'G15.7'
    * name = 'FLUX'; format = '1024E'; unit = 'erg/s/cm**2/Angstrom'; disp = 'G15.7'
    * name = 'ERROR'; format = '1024E'; unit = 'erg/s/cm**2/Angstrom'; disp = 'G15.7'
    * name = 'NET_ERROR'; format = '1024E'; unit = 'Counts/s'; disp = 'G15.7'
    * name = 'DQ'; format = '1024I'; null = -32767; disp = 'I11'
    * name = 'A2CENTER'; format = '1E'; unit = 'pixel'; disp = 'G15.7'
    * name = 'EXTRSIZE'; format = '1E'; unit = 'pixel'; disp = 'G15.7'
    * name = 'MAXSRCH'; format = '1I'; unit = 'pixel'; null = -32767; disp = 'I11'
    * name = 'BK1SIZE'; format = '1E'; unit = 'pixel'; disp = 'G15.7'
    * name = 'BK2SIZE'; format = '1E'; unit = 'pixel'; disp = 'G15.7'
    * name = 'BK1OFFST'; format = '1E'; unit = 'pixel'; disp = 'G15.7'
    * name = 'BK2OFFST'; format = '1E'; unit = 'pixel'; disp = 'G15.7'
    * name = 'EXTRLOCY'; format = '1024E'; unit = 'pixel'; disp = 'G15.7'
    * name = 'OFFSET'; format = '1E'; unit = 'pixel'; disp = 'G15.7'

*GNU Terry Pratchett*
"""
from __future__ import print_function, division
from PyAstronomy import pyasl
import numpy as np
import matplotlib.pylab as plt
from astropy.io import fits
from os import listdir

from find_rv import radial_velocity

test = False

# Loads HR 260
hdu_hr_260 = fits.open('/home/sgongora/Dev/PyCrossSpec/files/HR_260/sx1/o4qa07020_sx1.fits')
hdu_hr_4521 = fits.open('/home/sgongora/Dev/PyCrossSpec/files/HR_4521/sx1/o8qs04030_sx1.fits')

if test:
    # Create the template
    tw = np.linspace(5000,5010,1000)
    tf = np.exp(-(tw-5004.0)**2/(2.*0.1**2))

    # Create data, which are not that well sampled
    dw = np.linspace(5000,5010,200)
    df = np.exp(-(dw-5004.17)**2/(2.*0.1**2))
else:
    # Create the template
    tw = hdu_hr_260[1].data.field('WAVELENGTH')[0]
    tf = hdu_hr_260[1].data.field('FLUX')[0]
    te = hdu_hr_260[1].data.field('ERROR')[0]

    # Create data, which are not that well sampled
    dw = hdu_hr_4521[1].data.field('WAVELENGTH')[0]
    df = hdu_hr_4521[1].data.field('FLUX')[0]
    de = hdu_hr_4521[1].data.field('ERROR')[0]

# Plot template and data
plt.title("Template (blue) and data (red)")
plt.plot(tw, tf, 'b.-')
plt.plot(dw, df, 'r.-')
plt.show()

"""
test = radial_velocity(dw, df, de, # spectral data on your target (uncertainty is NOT SNR)
                       tw, tf, te, # spectral data on a star with known RV, in the same (or at least overlapping) wavelength region
                       'HR_4521', 'HR_260', # their names, for the plots (see point 2 below for details)
                       0.3, 0.01, # the radial velocity of the standard, for the plots
                       1, # for the plots. Should be the same for both.
                       200, # set to 200 for 'default', see point 3 below for what this means
                       0, 0, 0) # set all to zero by default. See point 4 below for what this means

print(test)
"""

# Carry out the cross-correlation.
# The RV-range is -30 - +30 km/s in steps of 0.6 km/s.
# The first and last 20 points of the data are skipped.
# rv, cc = pyasl.crosscorrRV(dw, df, tw, tf, -30., 30., 30./50., skipedge=20)
rv, cc = pyasl.crosscorrRV(dw_cut, df_cut, tw, tf, -100., 100., 30./50, skipedge=20)

# Find the index of maximum cross-correlation function
maxind = np.argmax(cc)

print("Cross-correlation function is maximized at dRV = ", rv[maxind], " km/s")
if rv[maxind] > 0.0:
  print("  A red-shift with respect to the template")
else:
  print("  A blue-shift with respect to the template")

plt.plot(rv, cc, 'bp-')
plt.plot(rv[maxind], cc[maxind], 'ro')
plt.show()
