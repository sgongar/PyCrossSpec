#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Versions:
- 0.1: Initial release.

Todo:
    * Unit tests.
    *

ColDefs(
    name = 'SPORDER'; format = '1I'; null = -32767; disp = 'I11'
    name = 'NELEM'; format = '1I'; null = -32767; disp = 'I11'
    name = 'WAVELENGTH'; format = '1024D'; unit = 'Angstroms'; disp = 'G25.16'
    name = 'GROSS'; format = '1024E'; unit = 'Counts/s'; disp = 'G15.7'
    name = 'BACKGROUND'; format = '1024E'; unit = 'Counts/s'; disp = 'G15.7'
    name = 'NET'; format = '1024E'; unit = 'Counts/s'; disp = 'G15.7'
    name = 'FLUX'; format = '1024E'; unit = 'erg/s/cm**2/Angstrom'; disp = 'G15.7'
    name = 'ERROR'; format = '1024E'; unit = 'erg/s/cm**2/Angstrom'; disp = 'G15.7'
    name = 'NET_ERROR'; format = '1024E'; unit = 'Counts/s'; disp = 'G15.7'
    name = 'DQ'; format = '1024I'; null = -32767; disp = 'I11'
    name = 'A2CENTER'; format = '1E'; unit = 'pixel'; disp = 'G15.7'
    name = 'EXTRSIZE'; format = '1E'; unit = 'pixel'; disp = 'G15.7'
    name = 'MAXSRCH'; format = '1I'; unit = 'pixel'; null = -32767; disp = 'I11'
    name = 'BK1SIZE'; format = '1E'; unit = 'pixel'; disp = 'G15.7'
    name = 'BK2SIZE'; format = '1E'; unit = 'pixel'; disp = 'G15.7'
    name = 'BK1OFFST'; format = '1E'; unit = 'pixel'; disp = 'G15.7'
    name = 'BK2OFFST'; format = '1E'; unit = 'pixel'; disp = 'G15.7'
    name = 'EXTRLOCY'; format = '1024E'; unit = 'pixel'; disp = 'G15.7'
    name = 'OFFSET'; format = '1E'; unit = 'pixel'; disp = 'G15.7'
)

*GNU Terry Pratchett*
"""
from __future__ import print_function, division
from multiprocessing import Process
import threading
from PyAstronomy import pyasl
import numpy as np
import matplotlib.pylab as plt
from astropy.io import fits
from os import listdir, getcwd

from find_rv import radial_velocity

cut_ = True
test = False
plot_ = False
np.set_printoptions(suppress=True)

cores_number = 4

total_rv = []

total_te = []
total_de = []
total_error = []

# def spectra_analysis(files_dir, spectra_reference, spectra_references):
#     """
#
#     :return:
#     """
#     hdu_hr_260 = fits.open('{}HR_260/sx1/{}'.format(files_dir,
#                                                     spectra_reference))
#     print('Starting test {} for {} spectra'.format(spectra_references.index(spectra_reference),
#                                                    len(spectra_references)))
#     # Loads HR 4521
#     spectra_list = listdir('{}HR_4521/sx1'.format(files_dir))
#     for spectra in spectra_list:
#         # print('New test')
#         hdu_hr_4521 = fits.open('{}HR_4521/sx1/{}'.format(files_dir, spectra))
#
#         # print(hdu_hr_4521[1].data.columns)
#         # print('Reference: {}'.format(spectra_reference))
#         # print('Target: {}'.format(spectra))
#
#         if test:
#             # Create the template
#             tw = np.linspace(5000,5010,1000)
#             tf = np.exp(-(tw-5004.0)**2/(2.*0.1**2))
#
#             # Create data, which are not that well sampled
#             dw = np.linspace(5000,5010,200)
#             df = np.exp(-(dw-5004.17)**2/(2.*0.1**2))
#         else:
#             # Create the template
#             tw = hdu_hr_260[1].data.field('WAVELENGTH')[0]
#             tf = hdu_hr_260[1].data.field('FLUX')[0]
#             te = hdu_hr_260[1].data.field('ERROR')[0]
#
#             # Create data, which are not that well sampled
#             dw = hdu_hr_4521[1].data.field('WAVELENGTH')[0]
#             df = hdu_hr_4521[1].data.field('FLUX')[0]
#             de = hdu_hr_4521[1].data.field('ERROR')[0]
#
#         total_te.append(np.mean(te))
#         total_de.append(np.mean(de))
#         total_error.append(np.mean(te) + np.mean(de))
#
#         # Plot template and data
#         if plot_:
#             plt.title("Template (blue) and data (red)")
#             plt.plot(tw, tf, 'b.-')
#             plt.plot(dw, df, 'r.-')
#             plt.show()
#
#         outlier_margin = 75
#         # Removes outliers
#         if cut_:
#             tw = tw[outlier_margin:-outlier_margin]
#             tf = tf[outlier_margin:-outlier_margin]
#             te = te[outlier_margin:-outlier_margin]
#
#             dw = dw[outlier_margin:-outlier_margin]
#             df = df[outlier_margin:-outlier_margin]
#             de = de[outlier_margin:-outlier_margin]
#
#         output_test = radial_velocity(dw, df, de,  # spectral data on your target (uncertainty is NOT SNR)
#                                       tw, tf, te,  # spectral data on a star with known RV, in the same (or at
#                                                    # least overlapping) wavelength region
#                                       'HR_4521', 'HR_260',  # their names, for the plots (point 2 below for details)
#                                       0.3, 0.01,  # the radial velocity of the standard, for the plots
#                                       1,  # for the plots. Should be the same for both.
#                                       2000,  # set to 200 for 'default', see point 3 below for what this means
#                                       0, 0, 0)  # set all to zero by default. See point 4 below for what this means
#
#         total_rv.append('{}-{}'.format(spectra_references.index(spectra_reference), output_test[0]))


class Thread_S(threading.Thread):
    def __init__(self, name, files_dir, spectra_reference, spectra_references):
        threading.Thread.__init__(self)
        self.name = name
        self.files_dir = files_dir
        self.spectra_reference = spectra_reference
        self.spectra_references = spectra_references

    def run(self):
        global total_rv
        """
        for i in range(0, 10, 1):
            c.acquire()
            if flag == 0:
                print("A: val=" + str(val))
                print("A {}".format(lista))
                flag = 1
                val = 30
                lista.append('A')
                c.notify_all()
            else:
                c.wait()
            c.release()
        """
        hdu_hr_260 = fits.open('{}HR_260/sx1/{}'.format(self.files_dir,
                                                        self.spectra_reference))
        print('Starting test {} for {} spectra'.format(self.spectra_references.index(self.spectra_reference),
                                                       len(self.spectra_references)))
        # Loads HR 4521
        spectra_list = listdir('{}HR_4521/sx1'.format(self.files_dir))
        for spectra in spectra_list:
            hdu_hr_4521 = fits.open('{}HR_4521/sx1/{}'.format(self.files_dir, spectra))

            if test:
                # Create the template
                tw = np.linspace(5000, 5010, 1000)
                tf = np.exp(-(tw - 5004.0) ** 2 / (2. * 0.1 ** 2))

                # Create data, which are not that well sampled
                dw = np.linspace(5000, 5010, 200)
                df = np.exp(-(dw - 5004.17) ** 2 / (2. * 0.1 ** 2))
            else:
                # Create the template
                tw = hdu_hr_260[1].data.field('WAVELENGTH')[0]
                tf = hdu_hr_260[1].data.field('FLUX')[0]
                te = hdu_hr_260[1].data.field('ERROR')[0]

                # Create data, which are not that well sampled
                dw = hdu_hr_4521[1].data.field('WAVELENGTH')[0]
                df = hdu_hr_4521[1].data.field('FLUX')[0]
                de = hdu_hr_4521[1].data.field('ERROR')[0]

            total_te.append(np.mean(te))
            total_de.append(np.mean(de))
            total_error.append(np.mean(te) + np.mean(de))

            # Plot template and data
            if plot_:
                plt.title("Template (blue) and data (red)")
                plt.plot(tw, tf, 'b.-')
                plt.plot(dw, df, 'r.-')
                plt.show()

            outlier_margin = 75
            # Removes outliers
            if cut_:
                tw = tw[outlier_margin:-outlier_margin]
                tf = tf[outlier_margin:-outlier_margin]
                te = te[outlier_margin:-outlier_margin]

                dw = dw[outlier_margin:-outlier_margin]
                df = df[outlier_margin:-outlier_margin]
                de = de[outlier_margin:-outlier_margin]

            output_test = radial_velocity(dw, df, de,  # spectral data on your target (uncertainty is NOT SNR)
                                          tw, tf, te,
                                          # spectral data on a star with known RV, in the same (or at least
                                          # overlapping) wavelength region
                                          'HR_4521', 'HR_260',
                                          # their names, for the plots (see point 2 below for details)
                                          0.3, 0.01,  # the radial velocity of the standard, for the plots
                                          1,  # for the plots. Should be the same for both.
                                          2000,  # set to 200 for 'default', see point 3 below for what this means
                                          0, 0, 0)  # set all to zero by default. See point 4 below for what this means

            c.acquire()
            total_rv.append('{}-{}'.format(self.spectra_references.index(self.spectra_reference),
                                           output_test[0]))
            c.release()


def main():
    # c = threading.Condition()
    spectra_j = []
    files_dir = '{}/files/'.format(getcwd())
    spectra_references = listdir('{}HR_260/sx1'.format(files_dir))
    for spectra_reference in spectra_references:
        spectra_p = Thread_S(spectra_reference, files_dir, spectra_reference, spectra_references)
        spectra_p.start()
        spectra_j.append(spectra_p)

    active_spectra = list([job.is_alive() for job in spectra_j])
    while True in active_spectra:
        active_spectra = list([job.is_alive() for job in spectra_j])
        pass

    """
    for spectra_reference in spectra_references:
        spectra_p = Process(target=spectra_analysis,
                            args=(files_dir, spectra_reference, spectra_references))
        spectra_j.append(spectra_p)
        spectra_p.start()

    active_spectra = list([job.is_alive() for job in spectra_j])
    while True in active_spectra:
        active_spectra = list([job.is_alive() for job in spectra_j])
        pass
    """
    print('test')
    print(total_rv)

"""
import pandas
total_dict = {'total_te': total_te, 'total_de': total_de, 'total_rv': total_rv,
              'total_error': total_error}
for key_ in total_dict.keys():
    print('key {} - len {}'.format(key_, len(total_dict[key_])))

total_df = pandas.DataFrame.from_dict(total_dict)

total_df.to_csv('total.csv')

print(np.mean(total_rv))
"""

if __name__ == "__main__":
    c = threading.Condition()
    total_rv = []
    main()
