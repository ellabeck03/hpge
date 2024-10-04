# module for help with openmc simulations
import glob
import numpy as np
import openmc
import os
import scipy
from scipy.special import erf
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def find_decay_probabilities(intensities):
    #intensities must be np array
    #normalises all intensities to create table of probabilities that sums to 1
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


def find_background_spectrum(energy_bins, spectrum):
    #finds the background in a given spectra

    #find the locations of the peaks
    peaks, current = find_peaks(energy_bins, spectrum)

    background_spectrum = []

    def is_peak(index):
        #helper function to check if an index correponds to a peak
        return energy_bins[index] in peaks

    for i in range(len(spectrum)):
        if energy_bins[i] in peaks:
            background_bins = []

            for offset in [-3, -2, -1, 1, 2, 3]:
                adjacent_index = i + offset
                if 0 <= adjacent_index < len(spectrum) and not is_peak(adjacent_index):
                    background_bins.append(spectrum[adjacent_index])

            if background_bins:
                background = sum(background_bins) / len(background_bins)
            else:
                background = spectrum[i]
        else:
            background = spectrum[i]

        background_spectrum.append(background)

    return background_spectrum


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
    peak_locations = list(scipy.signal.find_peaks(spectrum, distance = 10, prominence = 0.00002, width = 0.01))[0]
    peaks = []
    current = []
    
    for location in peak_locations:
        peaks.append(float(energy_bins[location]))
        current.append(float(spectrum[location]))
    
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
    

def create_intrinsic_efficiency_settings(crystal_radius, source_height, photon_beam_energies, num_batches, num_particles):
    #creates a source of isotropically distributed energies incident on the face of the germanium crystal
    spatial_dist = openmc.stats.CylindricalIndependent(
        r = openmc.stats.Uniform(0, crystal_radius),
        phi = openmc.stats.Uniform(0, 2 * np.pi),
        z = openmc.stats.Discrete([source_height], [1])
    )

    direction_dist = openmc.stats.Monodirectional((0, 0, -1))

    photon_beam_probabilities = np.full(len(photon_beam_energies), 1 / len(photon_beam_energies))
    energy_dist = openmc.stats.Discrete(photon_beam_energies, photon_beam_probabilities)

    source = openmc.IndependentSource()
    source.space = spatial_dist
    source.angle = direction_dist
    source.energy = energy_dist
    source.particle = 'photon'

    settings = openmc.Settings()
    settings.source = source
    settings.batches = num_batches
    settings.particles = num_particles
    settings.run_mode = 'fixed source'

    return settings


def create_intrinsic_efficiency_tally(photon_beam_energies, germanium_crystal):
    #creates a tally in order to find the number of photons passing through the crystal
    energy_filter = openmc.EnergyFilter(photon_beam_energies)
    surface_filter = openmc.SurfaceFilter(germanium_crystal)
    #cell_filter = openmc.CellFilter(germanium_crystal)

    efficiency_tally = openmc.Tally(name = 'photon_current')
    efficiency_tally.filters = [surface_filter, energy_filter]
    efficiency_tally.scores = ['current']

    efficiency_tallies = openmc.Tallies([efficiency_tally])
    efficiency_tallies.export_to_xml()

    return efficiency_tallies


def find_background_spectrum(energy_bins, spectrum):
    #finds the background in a given spectra

    #find the locations of the peaks
    peaks, current = find_peaks(energy_bins, spectrum)

    background_spectrum = []

    def is_peak(index):
        #helper function to check if an index correponds to a peak
        return energy_bins[index] in peaks

    for i in range(len(spectrum)):
        if energy_bins[i] in peaks:
            background_bins = []
            background_energy = []

            for offset in [-3, -2, -1, 1, 2, 3]:
                adjacent_index = i + offset
                if 0 <= adjacent_index < len(spectrum) and not is_peak(adjacent_index):
                    background_bins.append(spectrum[adjacent_index])
                    background_energy.append(energy_bins[adjacent_index])

            if len(background_bins) >= 2:
                interp_func = interp1d(background_energy, background_bins, kind = 'linear', fill_value = 'extrapolate')
                background = interp_func(energy_bins[i])
            else:
                background = spectrum[i]
        else:
            background = spectrum[i]

        background_spectrum.append(background)

    return background_spectrum