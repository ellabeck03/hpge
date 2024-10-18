# module for help with openmc simulations
import glob
import numpy as np
import openmc
import os
import scipy
from scipy.special import erf
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

def find_decay_probabilities(intensities):
    #intensities must be np array
    #normalises all intensities to create table of probabilities that sums to 1
    """
    Calculates the position, velocity and acceleration for a particle for a 
    given mass, spring constant, and damping constant using Euler's method.
    Returns arrays of the position and velocity of the particle at the times
    given by the 'time_array' array.
    
    Parameters
    ----------
    k : float
        A parameter which represents the spring constant of the system.
    m : float
        A parameter which represents the mass of the particle.
    b : float
        A parameter which represents the damping constant of the system.

    Returns
    -------
    position_array : array
        An array showing the particle's position at times given by the 
        'time_array' array.
    velocity_array : array
        An array showing the particle's velocity at times given by the 
        'time_array' array.
    """
    total = np.sum(intensities)
    probabilities = intensities / total
    
    return probabilities.tolist()
    

def find_current_activity(initial_activity, time_passed_s, half_life):
    #finds the current activity of the source based on the initial activity and the time passed

    decay_const = np.log(2) / half_life
    activity = initial_activity * np.exp(-1 * decay_const * time_passed_s)

    return activity
    

def create_hpge_settings(num_particles, num_batches, source_location, decay_peaks, decay_probabilities):
    #creates settings for an isotropic point source
    #source_location is a tuple of 3d points in cartesian coordinates

    settings = openmc.Settings()
    settings.batches = num_batches
    settings.particles = num_particles
    settings.run_mode = 'fixed source'
    
    source = openmc.IndependentSource()
    source.space = openmc.stats.Point(source_location)
    source.particle = 'photon'
    source.angle = openmc.stats.Isotropic()
    energy_distribution = openmc.stats.Discrete(decay_peaks, decay_probabilities)
    source.energy = energy_distribution
    settings.source = source
    
    return settings


def create_hpge_spectrum_tally(crystal_cell, energy_filter_bins):
    #creates pulse_height tally for a hpge simulation
    
    energy_filter = openmc.EnergyFilter(energy_filter_bins)
    cell_filter = openmc.CellFilter(crystal_cell)
    
    tally = openmc.Tally(name = 'pulse_height')
    tally.filters = [cell_filter, energy_filter]
    tally.scores = ['pulse-height']
    
    tallies = openmc.Tallies()
    tallies.append(tally)
    tallies.export_to_xml()
    
    return tallies
    

def remove_files(pattern):
    for filename in glob.glob(pattern):
        os.remove(filename)
    return


def openmc_sim(geometry, materials, settings, tallies):
    #runs an openmc simulation for certain settings
    
    model = openmc.model.Model(geometry, materials, settings, tallies)
    remove_files('*.h5')
    
    results_filename = model.run(threads = 4)
    results = openmc.StatePoint(results_filename)
    
    tally_results = {}
    
    for tally in tallies:
        tally_result = results.get_tally(name = tally.name)
        tally_results[tally.name] = tally_result.mean.ravel()
        
    return tally_results


def scale_pulse_height_spectra(spectrum, current_activity, initial_activity):
    #scales the pulse height spectra for the current activity of the source
    #spectrum must be a numpy array

    scaled_spectrum = spectrum *  (current_activity / initial_activity)

    return scaled_spectrum


def get_background_removed_spectrum(spectrum, background):
    #removes the background from a hoge spectrum given a spectra and a background
    background_removed_spectrum = []

    for i in range(len(spectrum)):
        background_removed_data = spectrum[i] - background[i]
    
        if background_removed_data < 1E-7:
            background_removed_spectrum.append(0)

        else:
            background_removed_spectrum.append(background_removed_data)

    return background_removed_spectrum


def find_fwhm(measured_bins, measured_spectrum):

    max_index = np.argmax(measured_spectrum)
    max_value = measured_spectrum[max_index]

    left_boundary = max_index
    right_boundary = max_index

    for i in range(max_index - 1, -1, -1):
        if measured_spectrum[i] < max_value / 2:
            left_boundary = i + 1
            break

    for i in range(max_index + 1, len(measured_spectrum)):
        if measured_spectrum[i] < max_value / 2:
            right_boundary = i - 1
            break

    if left_boundary == max_index or right_boundary == max_index:
        print("peak is too narrow, assuming fwhm equal to half the bin width")
        return (measured_bins[1] - measured_bins[0]) / 2

    fwhm = measured_bins[right_boundary] - measured_bins[left_boundary]

    return fwhm


def apply_gaussian_broadening(energy_bins, spectrum, fwhm):
    # applies gaussian broadening to a hpge spectrum
    
    std = fwhm / (2 * np.sqrt(2 * np.log(2)))

    # broadened_spectrum = np.zeros_like(spectrum)

    # for i in range(len(spectrum)):
    #     gaussian = np.exp(-0.5 * ((energy_bins - energy_bins[i]) / std) ** 2)
    #     gaussian /= gaussian.sum()
    #     broadened_spectrum += spectrum[i] * gaussian

    
    # energy_diff = energy_bins[:, np.newaxis] - energy_bins[np.newaxis, :]
    # gaussian_matrix = np.exp(-0.5 * (energy_diff / std) ** 2)
    # gaussian_matrix /= gaussian_matrix.sum(axis = 0)

    # broadened_spectrum = np.dot(gaussian_matrix, spectrum)

    # Estimate the energy bin width
    bin_width = energy_bins[1] - energy_bins[0]
    
    # Apply Gaussian convolution (standard deviation must be in units of bins)
    broadened_spectrum = gaussian_filter1d(spectrum, std / bin_width)
    

    return broadened_spectrum


def find_peaks(energy_bins, spectrum, expected_peaks, tolerance = 0.05):
    #finds peaks in a hpge simulation
    #includes sanity check to make sure peaks are in the right place
    max_val = max(spectrum)
    avg_val = sum(spectrum) / len(spectrum)

    peak_locations = list(scipy.signal.find_peaks(spectrum, distance = 10, prominence = avg_val * 2))[0]
    print(len(peak_locations))
    #peak_locations = list(scipy.signal.find_peaks(spectrum, distance = 10, prominence = 0.0001))[0]
    peaks = []
    current = []
    
    for location in peak_locations:
        peak_energy = float(energy_bins[location])
        for expected_peak in expected_peaks:
            if abs(peak_energy - expected_peak) < tolerance * expected_peak:
                peaks.append(float(energy_bins[location]))
                current.append(float(spectrum[location]))

    #handle expected peaks that were missed in scipy peak detection
    # for expected_peak in expected_peaks:
    #     if not any(abs(peak - expected_peak) < tolerance * expected_peak for peak in peaks):
    #         #find the closest bin to the expected peak
    #         closest_bin_index = np.argmin(np.abs(np.array(energy_bins) - expected_peak))
    #         closest_bin_energy = float(energy_bins[closest_bin_index])
    #         peaks.append(closest_bin_energy)
    #         current.append(float(spectrum[closest_bin_index]))
    #         print(f"forcing peak at {closest_bin_energy} keV (closest to expected {expected_peak} keV)")
    
    return peaks, current


def find_absolute_efficiency(energies, peaks, expected_energies, expected_peaks, num_particles):
    #finds absolute efficiency of a hpge spectrum
    efficiency_results = []

    peak_indices = [np.abs(expected_energies - peak).argmin() for peak in energies]
    
    for i, idx in enumerate(peak_indices):
        photon_counts = expected_peaks[idx]
        spectra_counts = peaks[i]
        efficiency_results.append(float(spectra_counts / (photon_counts))) #normalising denominator for photons emitted

    errors = find_peak_error(peaks, num_particles)
    
    return efficiency_results, errors


def find_background_spectrum(energy_bins, spectrum, expected_peaks, tolerance = 0.05):
    #finds the background in a given spectra
    #dynamically chooses between linear and quadratic interpolation based on the slope of the background around each peak

    #find the locations of the peaks
    peaks, current = find_peaks(energy_bins, spectrum, expected_peaks, tolerance)

    background_spectrum = []

    def is_peak(index):
        #helper function to check if an index correponds to a peak
        return energy_bins[index] in peaks

    def gradient_change(energy_bins, spectrum, peak_index):
        # Calculate the gradient around the peak to determine the slope
        left_grad = spectrum[peak_index] - spectrum[peak_index - 1] if peak_index > 0 else 0
        right_grad = spectrum[peak_index + 1] - spectrum[peak_index] if peak_index < len(spectrum) - 1 else 0
        return abs(right_grad - left_grad)


    for i in range(len(spectrum)):
        if energy_bins[i] in peaks:
            background_bins = []
            background_energy = []
    
            for offset in list(range(-100, -1)) + list(range(1, 100)):
                adjacent_index = i + offset
                if 0 <= adjacent_index < len(spectrum) and not is_peak(adjacent_index) and spectrum[adjacent_index] > 0:
                    background_bins.append(spectrum[adjacent_index])
                    background_energy.append(energy_bins[adjacent_index])
    
            if len(background_bins) >= 100:
                interp_func = interp1d(background_energy, background_bins, kind = 'quadratic', fill_value = 'extrapolate')
                background = interp_func(energy_bins[i])
                # grad_change = gradient_change(energy_bins, spectrum, i)
                # if grad_change > 0.0000001:  # Threshold to decide if we need quadratic (you can adjust this)
                #     interp_func = interp1d(background_energy, background_bins, kind='quadratic', fill_value='extrapolate')
                # else:
                #     interp_func = interp1d(background_energy, background_bins, kind='linear', fill_value='extrapolate')
                
                # background = interp_func(energy_bins[i])
            
            elif len(background_bins) >= 10:
                interp_func = interp1d(background_energy, background_bins, kind='linear', fill_value='extrapolate')
                background = interp_func(energy_bins[i])

            else:
                background = spectrum[i]  # Use the original spectrum value if not enough non-zero counts
                print(f'warning: background may not be successfully estimated for energy {energy_bins[i]} keV')
        else:
            background = spectrum[i]

        background_spectrum.append(background)

    return background_spectrum


def find_peak_error(spectrum, num_particles):
    # return the error of a peak based on poisson statistics
    # calibrated by total number of counts in the experimental spectrum taken
    errors = []
    peak_counts = []

    for point in spectrum:
        peak_counts.append(point * num_particles)

    for count in peak_counts:
        errors.append(1 / np.sqrt(count))
    
    return errors


def detector_channel_calibration(channels, gradient, intercept):
    """
    performs the linear calibration of detector channels to energy bins (in eV) given the parameters.
    """

    calibration = (channels * gradient + intercept) * 1000

    return calibration


def spectrum_calibration(spectrum):
    """
    calibrates the spectrum so it has the same scale (particles per source particle) as the simulation
    """

    total = np.sum(spectrum)
    print(total)
    
    return spectrum / total