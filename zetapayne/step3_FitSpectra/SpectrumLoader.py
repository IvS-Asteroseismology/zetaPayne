import os
import numpy as np
import astropy.io.fits as pf

from zetapayne.common_functions import DER_SNR

def load_ASCII(path):
	data = np.loadtxt(path)
	wave = data[:,0]
	flux = data[:,1]
	if data.shape[1] > 2:
	    err = data[:,2]
	else:
	    SNR = DER_SNR(flux)
	    err = flux/SNR
	obj_id = os.path.basename(path)
	return wave, flux, err, obj_id

def load_NPZ(path):
	npz = np.load(path)
	flux = np.squeeze(npz['flux'])
	wave = npz['wave']
	SNR = DER_SNR(flux)
	err = 1e-3 + np.zeros(len(flux)) # these are normalised synthetic spectra so small error
	obj_id = os.path.basename(path)
	return wave, flux, err, obj_id

def load_FITS(path):
	spec = pf.getdata(path)
	wave = spec['wavelength']
	flux = spec['flux']
	err = spec['error']
	obj_id = path.split('/')[-1].split('.')[0]
	return wave, flux, err, obj_id

def load_FEROS(path):
	spec = pf.getdata(path)
	wave = spec[0]['WAVE']
	flux = spec[0]['FLUX']
	err = spec[0]['ERR']
	obj_id = path.split('/')[-1].split('.')[0]
	return wave, flux, err, obj_id

def load_HERMES(path):
	flux = pf.getdata(path)
	header = pf.getheader(path)

	ref_pix = int(header['CRPIX1'])-1
	ref_val = float(header['CRVAL1'])
	ref_del = float(header['CDELT1'])
	numberpoints = flux.shape[0]
	unitx = header['CTYPE1']
	wave_start = ref_val - ref_pix*ref_del
	wave_end = wave_start + (numberpoints-1)*ref_del
	wave = np.linspace(wave_start, wave_end, numberpoints)
	wave = np.exp(wave)
	SNR = DER_SNR(flux)
	err = flux/SNR
	obj_id = os.path.basename(path)
	return wave, flux, err, obj_id


class SpectrumLoader():
	def __init__(self, format, path):
		selector = {'ASCII':load_ASCII, 'FITS':load_FITS, 'NPZ':load_NPZ, 'HERMES':load_HERMES, 'FEROS':load_FEROS}

		if not format in selector:
		    raise Exception('Unknown spectrum format ' + format)

		self.load_func = selector[format]
		self.wave, self.flux, self.err, self.obj_id = self.load_func(path)
