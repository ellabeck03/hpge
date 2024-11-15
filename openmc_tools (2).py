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
    """
    converts decay intensities to probabilities which sum to 1
    
    Parameters
    ----------
    intensities : np.array
        array of decay intensities

    Returns
    -------
    probabilities : list
        list of decay probabilities
    """
    total = np.sum(intensities)
    probabilities = intensities / total
    
    return probabilities.tolist()
    

def find_current_activity(initial_activity, time_passed_s, half_life):
    """
    finds the current activity of the source based on the initial activity and the time passed
    
    Parameters
    ----------
    initial_activity : float
        initial activity of the source (kBq)
    time_passed_s : int
        time passed since the initial activity measurement
    half_life : float
        half life of the source (s)

    Returns
    -------
    activity : list
        activity of the source in kBq / s
    """

    decay_const = np.log(2) / half_life
    activity = initial_activity * np.exp(-1 * decay_const * time_passed_s)

    return activity
    

def create_hpge_settings(num_particles, num_batches, source_location, decay_peaks, decay_probabilities):
    """
    creates settings for an isotropic point source
    
    Parameters
    ----------
    num_particles : int
    num_batches : int
    source_location : tuple
        tuple of 3d point in cartesian coordinates
    decay_peaks : np.array
        array of energy values corresponding to energy peaks
    decay_probabilities : list
        list of decay probabilities corresponding to energy peaks

    Returns
    -------
    settings : openmc.Settings object
        settings corresponding to an isotropic point source
    """

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
    """
    creates pulse height tally for a hpge simulation
    
    Parameters
    ----------
    crystal_cell : openmc.Cell object
        cell corresponding to the germanium crystal
    energy_filter_bins : np.array
        corresponding to desired list of energy bins for the simulation

    Returns
    -------
    tallies : openmc.Tallies object
        tallies corresponding to a hpge pulse height tally
    """
    
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
    """
    removes files with a specified extension from directory
    
    Parameters
    ----------
    pattern : string
        file extension

    Returns
    -------
    None
    """
    for filename in glob.glob(pattern):
        os.remove(filename)
    return


def openmc_sim(geometry, materials, settings, tallies):
    """
    runs an openmc simulation for specified geometry, materials, settings, and tallies
    
    Parameters
    ----------
    geometry : openmc.Geometry object
    materials : openmc.Materials object
    settings : openmc.Settings object
    tallies : openmc.Tallies object

    Returns
    -------
    tally_results : np.array
    """
    
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
    """
    scales the pulse height spectra for the current activity of the source
    
    Parameters
    ----------
    spectrum : np.array
        tally results spectrum
    current_activity : float
    initial_activity : float

    Returns
    -------
    scaled_spectrum : np.array
    """

    scaled_spectrum = spectrum *  (current_activity / initial_activity)

    return scaled_spectrum


def get_background_removed_spectrum(spectrum, background):
    """
    removes the background from a hpge spectrum given a spectra and a background
    
    Parameters
    ----------
    spectrum : np.array
        tally results spectrum
    background : np.array
    
    Returns
    -------
    background_removed_spectrum : np.array
    """
    
    background_removed_spectrum = []

    for i in range(len(spectrum)):
        background_removed_data = spectrum[i] - background[i]
    
        if background_removed_data < 1E-7:
            background_removed_spectrum.append(0)

        else:
            background_removed_spectrum.append(background_removed_data)

    return background_removed_spectrum


def find_fwhm(measured_bins, measured_spectrum):
    """
    estimates the fwhm of a set of peaks from experimental data
    
    Parameters
    ----------
    measured_bins : np.array
        energy bins from the experimental data
    measured_spectrum : np.array
        experimentally measured spectrum
    
    Returns
    -------
    fwhm : float
    """

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
    """
    applies gaussian broadening to a hpge spectrum
    
    Parameters
    ----------
    energy_bins : np.array
        energy bins used in simulation
    spectrum : np.array
        tally results generated by simulation
    fwhm : float
        estimated fwhm of the energy peaks
    
    Returns
    -------
    broadened_spectrum : np.array
    """
    
    std = fwhm / (2 * np.sqrt(2 * np.log(2)))

    bin_width = energy_bins[1] - energy_bins[0]
    broadened_spectrum = gaussian_filter1d(spectrum, std / bin_width)

    return broadened_spectrum


def find_peaks(energy_bins, spectrum, expected_peaks, tolerance = 0.05):
    """
    finds peaks in a hpge simulation
    includes sanity check to make sure peaks are in the right place
    
    Parameters
    ----------
    energy_bins : np.array
        energy bins used in simulation
    spectrum : np.array
        tally results generated by simulation (note: this must be background removed)
    expected_peaks : np.array
        location of expected energy peaks
    tolerance : float
        tolerance to which detected peaks are matched to expected peaks
    
    Returns
    -------
    peaks : np.array
        energy values of the detected peaks
    current : np.array
        pulse height values of the detected peaks
    """
    max_val = max(spectrum)
    avg_val = sum(spectrum) / len(spectrum)

    initial_peak_locations = list(scipy.signal.find_peaks(spectrum, distance = 10, prominence = avg_val * 2))[0]
    unique_peak_locations = list(set(initial_peak_locations))#remove duplicates
        
    peaks = []
    current = []
    
    for location in unique_peak_locations:
        peak_energy = float(energy_bins[location])
        for expected_peak in expected_peaks:
            if abs(peak_energy - expected_peak) < tolerance * expected_peak:
                if peak_energy not in peaks:
                    peaks.append(float(energy_bins[location]))
                    current.append(float(spectrum[location]))

    return peaks, current


def find_absolute_efficiency(energies, peaks, expected_energies, expected_peaks, num_particles):
    """
    finds absolute efficiency of a hpge spectrum
    
    Parameters
    ----------
    energies : np.array
        energy bins used in simulation
    peaks : np.array
        tally results generated by simulation (note: this must be background removed)
    expected_energies : np.array
        location of expected energy peaks
    expected_peaks : float
        pulse height / expected intensity of energy peaks (taken from probabilities)
    num_particles : int
        number of particles used in simulation
    
    Returns
    -------
    efficiency_results : np.array
        absolute efficiency values for each peak
    errors : np.array
        errors on each absolute efficiency
    """
    efficiency_results = []

    peak_indices = [np.abs(expected_energies - peak).argmin() for peak in energies]
    
    for i, idx in enumerate(peak_indices):
        photon_counts = expected_peaks[idx]
        spectra_counts = peaks[i]
        efficiency_results.append(float(spectra_counts / (photon_counts))) #normalising denominator for photons emitted

    errors = find_peak_error(peaks, num_particles)
    
    return efficiency_results, errors


def find_background_spectrum(energy_bins, spectrum, expected_peaks, tolerance = 0.05):
    """
    finds the background in a given spectra
    dynamically chooses between linear and quadratic interpolation based on the amount of data points to fit around each peak
    Parameters
    ----------
    energy_bins : np.array
        energy bins used in simulation
    spectrum : np.array
        tally results generated by simulation
    expected_peaks : np.array
        energy of expected peaks
    tolerance : float
        tolerance to which detected peaks are matched to expected peaks
    
    Returns
    -------
    background_spectrum : np.array
    """
    
    #find the locations of the peaks
    peaks, current = find_peaks(energy_bins, spectrum, expected_peaks, tolerance)

    background_spectrum = []

    def is_peak(index):
        #helper function to check if an index correponds to a peak
        return energy_bins[index] in peaks

    def gradient_change(energy_bins, spectrum, peak_index):
        #calculate the gradient around the peak to determine the slope
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
    """
    finds the error of a peak based on poisson statistics
    calibrated by total number of counts in the experimental spectrum taken
    
    Parameters
    ----------
    spectrum : np.array
    num_particles : int
    
    Returns
    -------
    errors : np.array
    """
    errors = []
    peak_counts = []

    for point in spectrum:
        peak_counts.append(point * num_particles)

    for count in peak_counts:
        errors.append(1 / np.sqrt(count))
    
    return errors


def detector_channel_calibration(channels, x1, x0):
    """
    performs the linear calibration of detector channels to energy bins (in eV) given the parameters
    
    Parameters
    ----------
    channels : np.array
        taken from experimental data
    x1 : float
        taken from experimental data
    x0 : float
        taken from experimental data
    
    Returns
    -------
    errors : np.array
    """

    calibration = (channels * x1 + x0) * 1000

    return calibration


def spectrum_calibration(spectrum):
    """
    calibrates the spectrum so it has the same scale (particles per source particle) as the simulation
    
    Parameters
    ----------
    spectrum : np.array
        taken from experimental data
    
    Returns
    -------
    spectrum / total : np.array
    """

    total = np.sum(spectrum)
    print(total)
    
    return spectrum / total


def peak_height_ratios(energy_bins, simulation, measurement, measured_energy_bins, tolerance = 500):
    """
    finds the ratio of simulated peak height to measured peak height for each peak
    
    Parameters
    ----------
    energy_bins : np.array
        energy bins used in simulation
    simulation : np.array
        simulation tally results
    measurement : np.array
        spectrum taken from experimental data
    measured_energy_bins : np.array
        energy bins taken from experimental data
    
    Returns
    -------
    energies : np.array
        corresponding to peak energies
    ratios : np.array
    """
    
    max_val = max(simulation)
    avg_val = sum(simulation) / len(simulation)

    simulated_peaks = list(scipy.signal.find_peaks(simulation, distance = 10, prominence = avg_val * 2))[0]
    measured_peaks = list(scipy.signal.find_peaks(measurement, distance = 10, prominence = avg_val * 2))[0]

    ratios = []
    energies = []

    for peak_loc in simulated_peaks:
        sim_peak_energy = energy_bins[peak_loc]

        #find closest corresponding energy bin in measured spectrum
        closest_idx = np.abs(measured_energy_bins[measured_peaks] - sim_peak_energy).argmin()
        measured_peak_energy = measured_energy_bins[measured_peaks[closest_idx]]
        measured_peak_value = measurement[measured_peaks[closest_idx]]

        if abs(measured_peak_energy - sim_peak_energy) <= tolerance:
            if simulation[peak_loc] > 0 and measured_peak_value > 0:
                ratio = measured_peak_value / simulation[peak_loc]
                if ratio > 0:
                    ratios.append(ratio)
                    energies.append(sim_peak_energy)

    if len(energies) == 0:
        print('warning: no peaks detected')

    return energies, ratios


def improve_efficiency_result(energy, original_probability, num_particles, num_batches, source_loc, geometry, materials, tallies, energy_bins):
    """
    running another simulation on a single energy in order to improve the uncertainty on the efficiency result
    
    Parameters
    ----------
    energy : float
    original_probability : float
    num_particles: int
    num_batches : int
    source_loc : tuple
        location of the source in 3d cartesian coordinates
    geometry : openmc.Geometry object
    materials : openmc.Materials object
    tallies : openmc.Tallies object
    energy_bins : array
        energy bins used in original simulation
    
    Returns
    -------
    efficiency : float
    error : float
    """
    
    settings = create_hpge_settings(num_particles, num_batches, source_loc, energy, np.array([1]))
    results = openmc_sim(geometry, materials, settings, tallies)
    spectrum = results["pulse_height"]
    background = find_background_spectrum(energy_bins, spectrum, energy)
    background_removed_spectrum = get_background_removed_spectrum(spectrum, background)
    peaks_loc, peaks = find_peaks(energy_bins, background_removed_spectrum, energy)
    efficiency, error = find_absolute_efficiency(peaks_loc, peaks, energy, np.array([1]), num_particles)

    efficiency *= original_probability
    error *= original_probability

    return efficiency, error
