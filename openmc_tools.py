# module for help with openmc simulations
import glob
import numpy as np
import openmc
import os

def find_decay_probabilities(intensities):
    #intensities must be np array
    #normalises all intensities to create table of probabilities that sums to 1
    total = np.sum(intensities)
    probabilities = intensities / total
    
    return probabilities.tolist()


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


def find_background_spectrum(spectrum):
    #finds the background spectrum given a full hpge spectrum
    background = []

    for i in range(len(spectrum)):
        background_counter = 1
        
        while True:
            left_index = i - background_counter
            right_index = i + background_counter
            
            if left_index < 0 or right_index >= len(spectrum):
                background.append(spectrum[i])
                break
            
            if spectrum[right_index] < 1E-3 and spectrum[left_index] < 1E-3:
                background.append((spectrum[left_index] + spectrum[right_index]) / 2)
                break
            
            background_counter += 1
    
    return background


def get_background_removed_spectrum(spectrum, background):
    #removes the background from a hoge spectrum given a spectra and a background
    background_removed_spectrum = []

    for i in range(len(spectrum)):
        background_removed_data = spectrum[i] - background[i]
    
        if background_removed_data < 0:
            background_removed_spectrum.append(0)

        else:
            background_removed_spectrum.append(background_removed_data)

    return background_removed_spectrum


def apply_gaussian_broadening(energy_bins, spectrum, fwhm):
    #applies gaussian broadening to a hpge spectrum
    std = fwhm / (2 * np.sqrt(2 * np.log(2)))

    broadened_spectrum = np.zeros_like(spectrum)

    for i in range(len(spectrum)):
        gaussian = np.exp(-0.5 * ((energy_bins - energy_bins[i]) / std) ** 2)
        gaussian /= gaussian.sum()
        broadened_spectrum += spectrum[i] * gaussian

    return broadened_spectrum


def find_peaks(energy_bins, spectrum):
    #finds peaks in a hpge simulation
    peak_locations = list(scipy.signal.find_peaks(spectrum, distance = 10, prominence = 0.00005, width = 0.01))[0]
    peaks = []
    current = []
    
    for location in peak_locations:
        peaks.append(float(energy_bins[location]))
        current.append(float(spectrum[location]))

    peaks_list.append(peaks)
    current_list.append(current)
    
    return peaks, current


def find_absolute_efficiency(energies, peaks, expected_energies, expected_peaks):
    #finds absolute efficiency of a hpge spectrum
    efficiency_results = []
    peak_indices = [np.abs(expected_energies - peak).argmin() for peak in energies]
    
    for i, idx in enumerate(peak_indices):
        photon_counts = expected_peaks[idx]
        spectra_counts = peaks[i]
        efficiency_results.append(float(spectra_counts / (photon_counts))) #normalising denominator for photons emitted

    return efficiency_results