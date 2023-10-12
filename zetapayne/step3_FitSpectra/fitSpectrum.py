import sys,os
import numpy as np
import matplotlib.pyplot as plt
from bisect import bisect
from numpy.polynomial.chebyshev import chebval
import astropy.io.fits as pf

from zetapayne.common_functions import param_names, param_units, parse_inp, DER_SNR, doppler_shift
from Fit import Fit
from Network import Network
from UncertFit import UncertFit
from SpectrumLoader import SpectrumLoader

import warnings
warnings.filterwarnings('ignore')


def fit_spectrum(filename, folder_name, NN, opt):
    '''
    Fit a spectrum with the ZETA-PAYNE.
    -----Parameters-----
    filename: string
        Path to the observed spectrum
    folder_name: string
        Path to the output folder
    NN: NN object
        Contains the NN
    opt: list
        List with user input from 'inputfile.conf'
    '''

    # User defined wavelength ranges, number of Chebyshev coefficients and presearch iterations
    wave_start = float(opt['wave_range'][0])
    wave_end = float(opt['wave_range'][1])
    Cheb_order = int(opt['N_chebyshev'][0]) #int(sys.argv[2])
    N_presearch_iter = int(opt['N_presearch_iter'][0])
    N_pre_search = int(opt['N_pre_search'][0])

    obj_name = filename.split('/')[-1].split('.')[0]
    print('-'*25)
    print('Object:', obj_name)

    # Load observed spectrum and take region between wave_start and wave_stop
    spectrum = SpectrumLoader(opt['spectrum_format'][0], filename)
    wave = spectrum.wave
    flux = spectrum.flux
    flux_error = spectrum.err

    start_idx = bisect(wave, wave_start)
    end_idx = bisect(wave, wave_end)
    wave = wave[start_idx:end_idx]
    flux = flux[start_idx:end_idx]
    flux_error = flux_error[start_idx:end_idx]
    index = np.where((np.isfinite(flux)) & (flux!=0) & (np.isfinite(flux_error)))
    wave = wave[index]
    flux = flux[index]
    flux_error = flux_error[index]

    f_mean = np.mean(flux)
    flux /= f_mean
    flux_error /= f_mean

    SNR = DER_SNR(flux)
    print('SNR:', SNR)

    # Optionally remove spectrum region by increasing the flux error to 9999
    #l1,l2,l3,l4,l5,l6 = int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]),int(sys.argv[6]),int(sys.argv[7])
    #flux_error[np.where((wave>l1) & (wave<l2))] = 9999
    #flux_error[np.where((wave>l3) & (wave<l4))] = 9999
    #flux_error[np.where((wave>l5) & (wave<l6))] = 9999

    fit = Fit(NN, Cheb_order, N_presearch_iter, N_pre_search)

    # For the convolution of synthetic spectra to the resolution of the spectrograph
    R = float(opt['spectral_R'][0])
    fit.lsf_R = R

    # Start the fitting and compute uncertainties
    unc_fit = UncertFit(fit, R)

    # Add 'p0=[<Teff>, <logg>, <vsini>, <vmicro>, <[M/H]>] + [1]*(Cheb_order+1))' if fit is stuck in local minimum
    # with initial guess of <Teff>, <logg>, <vsini>, <vmicro>, <[M/H]>
    fit_res = unc_fit.run(wave, flux, flux_error) #, p0=[7000, 4, 100, 2, 0] + [1]*(Cheb_order+1))

    # Print chi^2, parameter values and RV of best fitting model and save values to LOG file
    CHI2 = fit_res.chi2_func(fit_res.popt)
    CHI2_red = CHI2/(len(flux) - len(fit_res.popt))
    print('Chi^2:', '%.2e'%CHI2)
    print('Chi^2_red:', '%.4e'%CHI2_red)

    nnl = NN.num_labels() # index of RV

    with open('Output/' + folder_name + '/LOG', 'a') as flog:
    	L = [obj_name, str(Cheb_order), '%.2e'%CHI2, '%.4e'%CHI2_red, '%.2e'%SNR, '%.2f'%fit_res.popt[nnl], '%.2f'%fit_res.RV_uncert]
    	L.extend( [str('%.2f'%x) + ' ' + str('%.2f'%y) for x,y in zip(fit_res.popt,fit_res.uncert)] )
    	L.extend( [str('%.8f'%x) for x in fit_res.popt[nnl+1:]] )
    	s = ' '.join(L)
    	flog.write(s+'\n')

    print('-'*25)
    k = 0
    for i,v in enumerate(param_names):
    	if NN.grid[v][0]!=NN.grid[v][1]:
    		print(v, ':', '%.2f'%fit_res.popt[k], '+/-', '%.2f'%fit_res.uncert[k], param_units[i])
    		k += 1

    print('RV:', '%.2f'%fit_res.popt[nnl],  '+/-', '%.2f'%fit_res.RV_uncert, 'km/s')
    print('-'*25)

    # Normalise and RV shift observed spectrum
    che_coef = fit_res.popt[-Cheb_order:]
    che_x = np.linspace(-1, 1, len(flux))
    che_poly = chebval(che_x, che_coef)
    norm_flux = doppler_shift(wave, flux/che_poly, -fit_res.popt[nnl])

    # Save normalised spectra to fits file
    col1 = pf.Column(name='wave', format='E', array=np.array(wave))
    col2 = pf.Column(name='flux', format='E', array=np.array(flux))
    col3 = pf.Column(name='fit', format='E', array=np.array(fit_res.model))
    col4 = pf.Column(name='flux_norm', format='E', array=np.array(norm_flux))
    col5 = pf.Column(name='fit_norm', format='E', array=np.array(doppler_shift(wave, fit_res.model, -fit_res.popt[nnl]) / che_poly))
    col6 = pf.Column(name='che_poly', format='E', array=np.array(che_poly))
    cols = pf.ColDefs([col1, col2, col3, col4, col5, col6])
    tbhdu = pf.BinTableHDU.from_columns(cols)
    tbhdu.writeto('Output/' + folder_name + '/' + obj_name + '_Cheb' + str(Cheb_order) + '_wave' + str(int(wave_start)) + '-' + str(int(wave_end)) + '_norm.fits', overwrite=True)

    # Plot observed spectrum with fitting model
    plt.figure(figsize=(15,5))
    plt.plot(wave, flux)
    plt.plot(wave, fit_res.model)
    plt.xlabel('Wavelength ($\mathrm{\AA}$)', fontsize=16)
    plt.ylabel('Flux', fontsize=16)
    plt.title(obj_name + ' -- Cheb' + str(Cheb_order) + ' -- $\chi^2$ = ' + str(round(CHI2,0)), fontsize=18)
    plt.text(5400, (max(flux)-min(flux))/2 + min(flux), '$T_{\mathrm{eff}}$ = ' + str(round(fit_res.popt[0],0)) + '$\pm$' + str(round(fit_res.uncert[0],0)) + ' K \n' + '$\log g$ = ' + str(round(fit_res.popt[1],2)) + '$\pm$' + str(round(fit_res.uncert[1],2)) + ' dex \n' + '$v\sin i$ = ' + str(round(fit_res.popt[2],1)) + '$\pm$' + str(round(fit_res.uncert[2],1)) + ' km/s \n' + r'$\xi$ = ' + str(round(fit_res.popt[3],2)) + '$\pm$' + str(round(fit_res.uncert[3],2)) + ' km/s \n' + '[M/H] = ' + str(round(fit_res.popt[4],2)) + '$\pm$' + str(round(fit_res.uncert[4],2)) + ' dex \n', fontsize=14)
    plt.tight_layout()
    plt.savefig('Output/' + folder_name + '/' + obj_name + '_Cheb' + str(Cheb_order) + '_wave' + str(int(wave_start)) + '-' + str(int(wave_end)) + '.pdf')
    plt.show()

    return


if __name__=='__main__':

	if len(sys.argv)<2:
		print('Use:', sys.argv[0], '<filename>')
		exit()

	filename = sys.argv[1]
	opt = parse_inp('input.conf')

	NN_path = opt['NN_path'][0]
	NN = Network()
	NN.read_in(NN_path)
	print('NN: ', NN_path.split('/')[-1])

	folder_name = opt['log_dir'][0]
	if not os.path.exists('Output'):
		os.makedirs('Output')
	if not os.path.exists('Output/' + folder_name):
		os.makedirs('Output/' + folder_name)

	fit_spectrum(filename, folder_name, NN, opt)
