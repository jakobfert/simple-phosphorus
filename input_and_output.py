# -*- coding: utf-8 -*-
"""
Created in Aug 2020

This file holds functions for reading input data, writing output data, and edit dataframes.

@author: pferdmenges-j
"""

import cmf
import pandas as pd
from pathlib import Path
from datetime import datetime
import statistics
import csv


# ------------------------------------------- READ DATA -------------------------------------------
def read_data(source, irrigation=False, profile=False, drop=False, rename=False):
    df = pd.read_csv(source, sep=';', decimal=',', engine='python', header=0, index_col='Sample')
    if irrigation:
        df = df[df['Irrigation'] == irrigation]
    if profile:
        if df['Profile'].dtype == 'int64':
            df = df[df['Profile'] == profile]
        elif df['Profile'].dtype == 'object':
            df['Profile'] = df['Profile'].tolist()
            df = df[df.Profile.str.contains(str(profile))]

    if isinstance(drop, list):
        df.drop(drop, axis=1, inplace=True)
    if isinstance(rename, dict):
        df.rename(columns=rename, inplace=True)
    return df


# EDITING DATAFRAMES
def cut_df(df, layer_mask, start, stop):
    df = df.isel(time=slice(start, stop)).sel(layer_boundary=layer_mask)
    return df


# ------------------------------------------- INPUT -------------------------------------------
# RAIN
def create_1_min_irrigation_per_6_min(startup=60, repeat=40, fade_out=19 * 60):
    """
    Easy creation of time series by choosing a initialization time (default: 60 min), a period of sprinkling
    (default: 240 min, with one minute of rain (2 mm or 2 l/m2) and 5 without), and a fade out

    Precipitation in cmf is always in mm/day. For 2 mm/min, I actually have to sum it up for a whole day, thus
    2 mm/min * 60 min/h * 24 h/day.
    The area of the cell is not important here.

    IMPORTANT: Rainfall intensity of the irrigation was 20 L/(m2*h) over 4 h = 80 L/m2 [or: 20 mm/h = 80 mm / 4h]
    Therefore, to bring it to mm/day, it is 20 mm/h * 24 h/day = 480 mm/day.
    I calculated this wrong: I went from 20 mm/h to 2 mm/6min (thus far correct), but then to go back I multiplied it by
    60 min/h (so 2mm/min * 60 min/h * 24 h/day), instead of 10 (2 mm/6min * 10 * 6min/h * 24 h/day)!!!

    :param startup: initialization time in minutes
    :param repeat: number of repetitions of 6 minute segments
    :param fade_out: simulated time in minutes after irrigation
    :return: list of irrigation timeseries
    """
    return (startup * [0.]  # 60 minutes dry
            + repeat * ([2. * 10 * 24] + 5 * [0.])  # 40 times 1 minute of rain and 5 without = 4h
            + fade_out * [0.])  # 19 h for washout


def irrigation_experiment(begin, dt, startup=60, repeat=40, fade_out=19 * 60):
    """
    Creates an irrigation timeseries and transforms it to a cmf.timeseries

    :param begin: cmf.Time object, starting point of solute_results
    :param dt: cmf.Time steps, e.g. cmf.min, cmf.h, cmf.day
    :param startup: initialization time in minutes
    :param repeat: number of repetitions of 6 minute segments
    :param fade_out: simulated time in minutes after irrigation
    :return: cmf.timeseries containing irrigation data
    """
    rain = create_1_min_irrigation_per_6_min(startup, repeat, fade_out)
    return cmf.timeseries.from_array(begin, dt, data=rain)


# SOIL
def real_soil_from_csv(soil_file, organic_layer_as_layer=False):
    """
    When already a usable soil file exists, use this function instead of creating a dataframe from Brook90 data
    :param organic_layer_as_layer: for treating an organic layer like a regular soil layer
    :param soil_file: file name
    """
    soil = pd.read_csv(soil_file, sep=';', decimal=',', engine='python', header=0)
    if organic_layer_as_layer:
        soil.drop(['Depth(m)'], axis=1, inplace=True)
        soil.rename(columns={'Depth(negativ)': 'Depth(m)'}, inplace=True)
    return soil


# SURFACE
def surface_df(source, irrigation=False, profile=False, drop=False):
    drop = drop if drop else ['Nr. (fortl.)', 'Irrigation', 'Profile']
    df = read_data(source=source, irrigation=irrigation, profile=profile, drop=drop)
    return df


# ------------------------------------------- Evaluation -------------------------------------------
def evaluation_water_df(source=Path('input/MIT_Evaluation.csv'), irrigation=False, profile=False, drop=False):
    """
    Creates DataFrame from file, reduces it to the important rows and columns, and calculates the cumulated sum of
    water (as opposed to the water amount in intervals of different time steps)
    :param source: path of evaluation file
    :param irrigation: number(s) of irrigation which should be considered
    :param profile: number(s) of soil profile which should be considered
    :param drop: columns which should be excluded from evaluation file
    :return:
    """
    df = read_data(source, irrigation=irrigation, profile=profile, drop=drop)
    df['amount_measured_absolute'] = df['amount [ml]'] / 1000  # this refers to the L per lysimeter
    df['amount_measured_l_per_m2'] = df['amount_measured_absolute'] / 0.2  # since area of lysimeter is 0.2 m2
    df['measured_flux_l_per_m2_per_min'] = df['amount_measured_l_per_m2'] / df['duration [min]']
    df['measured_flux_l_per_m2_per_day'] = df['measured_flux_l_per_m2_per_min'] * 60 * 24
    df['measured_flux_m3_per_m2_per_day'] = df['measured_flux_l_per_m2_per_day'] * 1e-3
    # Important: this still refers to the flux per m2
    return df


def evaluation_phosphorus_df(evaluation):
    """
    Reading evaluation file and add states (=total amounts) of DIP, DOP and PP for lysimeter samples.
    All concentrations are in micrograms per L, and all states are in micrograms!
    Careful: Results from CMF are AMOUNT/m3, so they need to be converted
    """
    evaluation.rename(columns={'DIP [mcg/l]': 'dip_measured_mcg_per_L', 'DOP [mcg/l]': 'dop_measured_mcg_per_L',
                               'PP [mcg/l]': 'pp_measured_mcg_per_L'}, inplace=True)

    # concentration refers to mcg/L. To calculate mcg/m3, it needs to be multiplied with 1000 L/m3
    evaluation['dip_measured_mcg_per_m3'] = evaluation['dip_measured_mcg_per_L'] * 1e3
    evaluation['dop_measured_mcg_per_m3'] = evaluation['dop_measured_mcg_per_L'] * 1e3
    evaluation['pp_measured_mcg_per_m3'] = evaluation['pp_measured_mcg_per_L'] * 1e3

    # the state is the total amount of TRANSPORTED P per m2
    evaluation['dip_measured_state_per_m2'] = evaluation['dip_measured_mcg_per_L'] * evaluation[
        'amount_measured_l_per_m2']
    evaluation['dop_measured_state_per_m2'] = evaluation['dop_measured_mcg_per_L'] * evaluation[
        'amount_measured_l_per_m2']
    evaluation['pp_measured_state_per_m2'] = evaluation['pp_measured_mcg_per_L'] * evaluation[
        'amount_measured_l_per_m2']
    return evaluation


# ------------------------------------------- FORMAT SIMULATION RESULTS -------------------------------------------
def depth_and_layers(approach):
    if approach.profile == 1:
        depth = [0.12, 0.34, 0.68]
    elif approach.profile == 2:
        depth = [0.18, 0.47, 0.65]
    elif approach.profile == 3:
        depth = [0.26]
    elif approach.profile == 4:
        depth = [0.13, 0.35, 0.70]
    elif approach.profile == 5:
        depth = [0.14, 0.37, 0.74]
    elif approach.profile == 6:
        depth = [0.18, 0.25]  # 0.16 oder 0.18?
    else:
        depth = [0.1, 0.4, 0.8]

    # save number of soil layers corresponding to lysimeter depths
    number = 0
    layers = []  # [7, 18, 28]
    for i in approach.project.c.layers:
        up = i.upper_boundary
        low = i.lower_boundary
        for j in depth:
            if up < j <= low:
                layers.append(number)
        number += 1

    return depth, layers


def timespan(begin, row):
    # time0 and time1 are start and end point of sample -> needed for choosing from simulation results
    t0 = begin + cmf.min * (int(row['time [min]']) - int(row['duration [min]']))  # start time of sample
    t1 = begin + cmf.min * (int(row['time [min]']))  # end time of sample

    return t0, t1


def format_water_results(approach, water_results):
    """
    TEXT
    :param water_results:
    :param approach:
    :return:
    """
    # save depth of lysimeters and number of corresponding soil layers:
    depth, layers = depth_and_layers(approach)

    # begin of the irrigation experiment
    begin = cmf.Time(12, 6, 2018, 10, 1)

    simulation_results = {'simulated_flux_l_per_m2_per_day': [], 'amount_simulated_l_per_m2': []}
    for index, row in approach.evaluation_df.iterrows():
        # i is the index from simulation results: depth.index chooses the index of depth (0,1,2) according to depth of
        # lysimeter; layers[depth.index] uses this index to find the right index of water_results (7,18,28)
        i = layers[depth.index(row['depth [m]'])]

        time0, time1 = timespan(begin, row)

        # mean flux in l per m2 per sec over time period t0:t1:
        mean_for_period = statistics.mean([value[i] for key, value in  # value[i] is the flux in the i-th soil layer
                                           water_results['simulated_flux_l_per_m2_per_day'].items() if
                                           time0 <= cmf.Time(datetime.strptime(key, '%d.%m.%Y %H:%M')) < time1])
        simulation_results['simulated_flux_l_per_m2_per_day'].append(mean_for_period)

        # L over time period t0:t1:
        amount_for_period = (mean_for_period * int(row['duration [min]'])) / (24 * 60)
        simulation_results['amount_simulated_l_per_m2'].append(amount_for_period)

    return simulation_results


def mean_concentration_for_period(phosphorus, water, i, s, time0, time1):
    conc = [value[i] for key, value in phosphorus[s.Name + '_simulated_mcg_per_m3_mx+mp'].items()
            if time0 <= cmf.Time(datetime.strptime(key, '%d.%m.%Y %H:%M')) < time1]
    flux = [value[i] for key, value in water['simulated_flux_l_per_m2_per_day'].items() if
            time0 <= cmf.Time(datetime.strptime(key, '%d.%m.%Y %H:%M')) < time1]

    return sum([(a * b) for a, b in zip(conc, flux)]) / sum(flux)


def format_phosphorus_results(approach, phosphorus_results, water_results):
    """
    TEXT
    :param phosphorus_results:
    :param water_results:
    :param approach:
    :return:
    """
    # save depth of lysimeters and number of corresponding soil layers:
    depth, layers = depth_and_layers(approach)
    # begin of the irrigation experiment
    begin = cmf.Time(12, 6, 2018, 10, 1)
    simulation_results = {'dip_simulated_mcg_per_m3_mx+mp': [], 'dop_simulated_mcg_per_m3_mx+mp': [],
                          'pp_simulated_mcg_per_m3_mx+mp': [], 'dip_simulated_state_per_m2_mx+mp': [],
                          'dop_simulated_state_per_m2_mx+mp': [], 'pp_simulated_state_per_m2_mx+mp': []}

    for index, row in approach.evaluation_df.iterrows():
        i = layers[depth.index(row['depth [m]'])]
        time0, time1 = timespan(begin, row)
        for s in approach.project.solutes:
            # Concentration over time: this is not simply the mean of all time steps, since it needs to be weighted by
            # the water amount, similar to total_concentration_mcg_per_m3() in input_and_output.py
            mcg_per_m3_mean = mean_concentration_for_period(phosphorus_results, water_results, i, s, time0, time1)
            simulation_results[s.Name + '_simulated_mcg_per_m3_mx+mp'].append(mcg_per_m3_mean)

            # since the state is always per m2 for a whole day, it needs to be transformed to the minute
            sum_for_period = sum([value[i] / (24 * 60) for key, value in  # value[i] is the flux in the i-th soil layer
                                  phosphorus_results[s.Name + '_simulated_state_per_m2_mx+mp'].items()
                                  if time0 <= cmf.Time(datetime.strptime(key, '%d.%m.%Y %H:%M')) < time1])

            # maybe for testing: concentration * water_amount should be the same
            simulation_results[s.Name + '_simulated_state_per_m2_mx+mp'].append(sum_for_period)

    return simulation_results


# ------------------------------------------- ERROR FILE -------------------------------------------
def write_error_file(spotpy, name='error.csv'):
    param_list = ['reason', 'time_step']
    for i in spotpy:
        param_list.append(i.name)

    with open(name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(param_list)


def append_error_file(spotpy, name='error.csv', mode='a', error='IndistinctError', time_step='-'):
    param_list = [error, time_step]
    for i in list(vars(spotpy).values())[1]:
        param_list.append(i[0])

    with open(name, mode=mode, newline='') as file:
        writer = csv.writer(file)
        writer.writerow(param_list)
