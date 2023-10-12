# Code for fitting spectra, using the models in spectral_model.py
from __future__ import absolute_import, division, print_function # python2 compatibility
import numpy as np
from scipy.optimize import curve_fit
from numpy.polynomial.chebyshev import chebval
from scipy.ndimage import gaussian_filter1d

from zetapayne.common_functions import doppler_shift
from zetapayne.sobol_seq import i4_sobol_generate
from Network import Network


class FitResult:
    pass

class Fit:

    def __init__(self, network:Network, Cheb_order, N_presearch_iter=1, N_pre_search=4000, tol=5.e-4):
        self.tol = tol # tolerance for when the optimizer should stop optimizing.
        self.network = network
        self.Cheb_order = Cheb_order
        self.N_presearch_iter = N_presearch_iter
        self.N_pre_search = N_pre_search
        self.RV_presearch_range = 100.0 # Search RV in [-50,50] km/s

    def run(self, wave, spec, spec_err, p0=None):
        '''
        Fit a single-star model to an observed spectrum.
        Fit parameters are: Teff, logg, vsini, vmicro and [M/H]
        -----Parameters-----
        wave: list
            Wavelengths of observed spectrum
        spec: list
            Flux values of observed spectrum
        spec_err: list
            Uncertainty values of observed spectrum
        p0: list
            An initial guess for where to initialize the optimizer. Because
            this is a simple model, having a good initial guess is usually not
            important.
        -----Returns-----
        popt: the best-fit labels
        pcov: the covariance matrix, from which you can get formal fitting uncertainties
        model_spec: the model spectrum corresponding to popt
        '''

        # Number of labels + number of Chebyshev coefficients + radial velocity
        nnl = self.network.num_labels()
        num_labels = nnl + self.Cheb_order + 1

        # Compute kernel to convolve spectrum to resolution of the spectrograph
        FWHM_factor = 2 * np.sqrt(2* np.log(2))
        R = self.lsf_R
        center_lambda = 0.5 * (max(self.network.wave) + min(self.network.wave))
        delta_lambda  = center_lambda / R
        sigma = delta_lambda / FWHM_factor
        pixel_width = (max(self.network.wave) - min(self.network.wave))/len(self.network.wave)
        kernel_sigma = sigma / pixel_width

        def fit_func(dummy_variable, *labels):
            ## Uncommend some of these lines if some parameters are fixed to a certain value.
            # lst = list(labels[:nnl])
            # Teff_scaled = (23900 - self.network.x_min[0])/(self.network.x_max[0] - self.network.x_min[0]) - 0.5
            # logg_scaled = (3.65 - self.network.x_min[1])/(self.network.x_max[1] - self.network.x_min[1]) - 0.5
            # #vsini_scaled = (20 - self.network.x_min[2])/(self.network.x_max[2] - self.network.x_min[2]) - 0.5
            # #vmicro_scaled = (2 - self.network.x_min[3])/(self.network.x_max[3] - self.network.x_min[3]) - 0.5
            # lst_fixed = [Teff_scaled] + [logg_scaled] +  [lst[2]] + [lst[3]] + [lst[4]]
            # labels_fixed = tuple(lst_fixed)
            # nn_spec = self.network.get_spectrum_scaled(scaled_labels = labels_fixed)

            # Model spectrum for labels[:nnl] = Teff, logg, vsini, vmicro, metallicity
            nn_spec = self.network.get_spectrum_scaled(scaled_labels = labels[:nnl])

            # Doppler shift synthetic spectrum
            nn_spec = doppler_shift(self.network.wave, nn_spec, labels[nnl])

            # Convolve to resolution of spectrograph
            nn_conv = gaussian_filter1d(nn_spec, kernel_sigma)
            nn_resample = np.interp(wave, self.network.wave, nn_conv)

            # Introduce response function
            Cheb_coefs = labels[nnl + 1 : nnl + 1 + self.Cheb_order]
            Cheb_x = np.linspace(-1, 1, len(nn_resample))
            Cheb_poly = chebval(Cheb_x, Cheb_coefs)
            spec_with_resp = nn_resample * Cheb_poly
            return spec_with_resp


        x_min = self.network.x_min
        x_max = self.network.x_max

        # If no initial guess is supplied, initialize with the median value
        if p0 is None:
            p0 = np.zeros(num_labels)
        else:
            p0[:nnl] = (p0[:nnl]-x_min)/(x_max-x_min)-0.5


        # Prohibit the minimimizer to go outside the range of training set
        bounds = np.zeros((2,num_labels))
        if not hasattr(self, 'bounds_unscaled'):
            bounds[0,:nnl] = -0.5
            bounds[1,:nnl] = 0.5
        else:
            for i in [0,1]:
                bounds[i,:nnl] = (self.bounds_unscaled[i,:]-x_min)/(x_max-x_min) - 0.5
        bounds[0, nnl:] = -np.inf
        bounds[1, nnl:] = np.inf

        # Make sure the starting point is within bounds
        for i in range(num_labels):
            if not bounds[0,i]  < p0[i] < bounds[1,i]:
                p0[i] = 0.5*(bounds[0,i] + bounds[1,i])

        # Pre-searching on sobol grid
        for it in range(self.N_presearch_iter):
            # Fit spectrum to get response function
            popt, pcov = curve_fit(fit_func, xdata=[], ydata = spec, sigma = spec_err, p0 = p0,
                        bounds = bounds, ftol = self.tol, xtol = self.tol, absolute_sigma = True, method = 'trf')

            best = [np.inf, None]
            N_sobol = i4_sobol_generate(nnl+1, self.N_pre_search) # 6D: 5 params + RV
            for i in range(self.N_pre_search):
                lbl = np.copy(popt)
                lbl[:nnl] = bounds[0,:nnl] + N_sobol[i,:nnl]*(bounds[1,:nnl] - bounds[0,:nnl])
                lbl[nnl] = self.RV_presearch_range * (N_sobol[i,nnl] - 0.5) #RV

                model = fit_func([], *lbl)
                diff = (spec - model) / spec_err
                chi2 = np.sum(diff**2)
                if chi2 < best[0]:
                    best[0] = chi2
                    best[1] = lbl

            p0 = best[1]


        # Run the optimizer with best result of pre-search as initial values
        popt, pcov = curve_fit(fit_func, xdata=[], ydata = spec, sigma = spec_err, p0 = p0,
                    bounds = bounds, ftol = self.tol, xtol = self.tol, absolute_sigma = True, method = 'trf')
        model_spec = fit_func([], *popt)

        res = FitResult()
        res.model = model_spec
        res.popt_scaled = np.copy(popt)
        res.pcov_scaled = np.copy(pcov)

        # Rescale the result back to original unit
        popt[:nnl] = (popt[:nnl]+0.5)*(x_max-x_min) + x_min
        pcov[:nnl,:nnl] = pcov[:nnl,:nnl]*(x_max-x_min)

        res.popt = popt
        res.pcov = pcov

        def chi2_func(labels):
            labels_sc = np.copy(labels)
            labels_sc[:nnl] = (labels_sc[:nnl] - x_min)/(x_max - x_min) - 0.5
            model = fit_func([], *labels_sc)
            diff = (spec - model) / spec_err
            chi2 = np.sum(diff**2)
            return chi2

        res.chi2_func = chi2_func
        return res
