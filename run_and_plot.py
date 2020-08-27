# This file will hold functions to run the script and to plot the results


# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 16:39:50 2019

@author: pferdmenges-j
"""

import cmf
import numpy as np
import pandas as pd
import time
from build_the_model import CmfModel, MacroporeFastFlow, BypassFastFlow
from operator import add


def create_water_result_df():
    """
    Creates empty dataframe to fill later with results of water fluxes.

    :return: dataset to save water results
    """

    water = {'wetness_mx': {}, 'wetness_mp': {}, 'water_volume_l_per_m2_mx': {}, 'water_volume_l_per_m2_mp': {},
             'percolation_l_per_m2_per_day_mx': {}, 'percolation_l_per_m2_per_day_mp': {},
             'gw_recharge_wb_l_per_m2_per_day': {}, 'gw_recharge_flux_l_per_m2_per_day': {},
             'surface_runoff_l_per_m2_per_day': {}, 'infiltration_l_per_m2_per_day': {},
             'ex_l_per_m2_per_day_mp_mx': {}, 'simulated_flux_l_per_m2_per_day': {}}

    return water


def create_phosphorus_result_df(model: CmfModel):
    """
    Creates empty dataframe to fill later with results of solute concentration and fluxes.

    :return: dataset to save solute results
    """

    phosphorus = {'concentration_gw_recharge_mcg_per_m3': {}}

    for s in model.solutes:
        phosphorus.update({'concentration_mcg_per_m3_mx_' + s.Name: {}, 'concentration_mcg_per_m3_mp_' + s.Name: {},
                           'simulated_state_mcg_per_m2_per_layer_mx_' + s.Name: {},
                           'simulated_state_mcg_per_m2_per_layer_mp_' + s.Name: {},
                           'concentration_flux_mcg_per_m3_mx_' + s.Name: {},
                           'concentration_flux_mcg_per_m3_mp_' + s.Name: {}})
    return phosphorus


def amount_per_m3_to_amount_per_l(amount_per_m3):
    """
    Since volumes are always in m3 by default, but evaluation data is in L minute,
    this function re-calculates the fluxes in L/min

    :param amount_per_m3: concentration in amount per m3
    :return: concentration of solute in amount per L
    """
    return amount_per_m3 * 1e-3


def fill_water_result_df(model: CmfModel, df, i, t):
    """
    Function which is called each time step of solute_results to add values to df

    :param model: CmfModel
    :param df: xarray Dataset for solute_results results of water fluxes
    :param i: value increasing each time step +1 for selecting the right location in dataset
    :param t: current time
    :return: filled df xarray
    """
    tstr = str(t)
    df['wetness_mx'][tstr] = model.c.layers.wetness.tolist()  # Volume of water per Volume of pores
    df['water_volume_l_per_m2_mx'][tstr] = model.c.layers.volume.tolist()
    df['percolation_l_per_m2_per_day_mx'][tstr] = model.c.layers.get_percolation(t).tolist()
    df['gw_recharge_wb_l_per_m2_per_day'][tstr] = model.gw.waterbalance(t)
    df['gw_recharge_flux_l_per_m2_per_day'][tstr] = model.c.layers[-1].flux_to(model.gw, t)
    df['infiltration_l_per_m2_per_day'][tstr] = model.c.surfacewater.flux_to(model.c.layers[0], t)
    if model.surface_runoff:
        df['surface_runoff_l_per_m2_per_day'][tstr] = model.surface_runoff.waterbalance(t)

    if type(model.flow_approach) == MacroporeFastFlow:
        df['wetness_mp'][tstr] = model.flow_approach.macropores.wetness.tolist()
        df['water_volume_l_per_m2_mp'][tstr] = model.flow_approach.macropores.volume.tolist()
        df['percolation_l_per_m2_per_day_mp'][tstr] = model.flow_approach.macropores.percolation(t).tolist()
        df['ex_l_per_m2_per_day_mp_mx'][tstr] = [mp.flux_to(mp.layer, t) for mp in model.flow_approach.macropores]
        df['gw_recharge_flux_l_per_m2_per_day'][tstr] = (model.c.layers[-1].flux_to(model.gw, t) +
                                                         model.flow_approach.macropores[-1].flux_to(model.gw, t))
        df['infiltration_l_per_m2_per_day'][tstr] = (model.c.surfacewater.flux_to(model.c.layers[0], t) +
                                                     model.c.surfacewater.flux_to(model.flow_approach.macropores[0], t))

    elif type(model.flow_approach) == BypassFastFlow:
        df['percolation_l_per_m2_per_day_mp'][tstr] = [model.flow_approach.bypass[i].q(
            model.flow_approach.bypass[i].right_node(), t) for i in range(len(model.flow_approach.bypass))]

    if type(model.flow_approach) == MacroporeFastFlow or type(model.flow_approach) == BypassFastFlow:
        df['simulated_flux_l_per_m2_per_day'][tstr] = np.add(model.c.layers.get_percolation(t),
                                                             model.flow_approach.macropores.percolation(t)).tolist()
    else:
        df['simulated_flux_l_per_m2_per_day'][tstr] = model.c.layers.get_percolation(t).tolist()


def fill_phosphorus_result_df(model: CmfModel, x_solutes, i, t):
    """
    Function which is called each time step of solute_results to add values to df
    :param model: CmfModel
    :param x_solutes: xarray Dataset for solute_results results of solute transport
    :param i: value increasing each time step +1 for selecting the right location in dataset
    :param t: current time
    :return: filled x_solute xarray
    """
    tstr = str(t)
    for s in model.solutes:
        x_solutes['concentration_mcg_per_m3_mx_' + s.Name][tstr] = [layer.conc(s) for layer in model.c.layers]
        x_solutes['concentration_flux_mcg_per_m3_mx_' + s.Name][tstr] = [model.mx_infiltration.conc(t, s)] + [
            flux.conc(t, s) for flux in model.mx_percolation]
        x_solutes['simulated_state_mcg_per_m2_per_layer_mx_' + s.Name][tstr] = [layer.Solute(s).state * 1e-3 for layer
                                                                                in model.c.layers]
        if type(model.flow_approach) == MacroporeFastFlow:
            x_solutes['concentration_mcg_per_m3_mp_' + s.Name][tstr] = [mp.conc(s) for mp in
                                                                        model.flow_approach.macropores]
            x_solutes['concentration_flux_mcg_per_m3_mp_' + s.Name][tstr] = [model.flow_approach.mp_infiltration.conc(
                t, s)] + [flux.conc(t, s) for flux in model.flow_approach.mp_percolation]
            x_solutes['simulated_state_mcg_per_m2_per_layer_mp_' + s.Name][tstr] = [mp.Solute(s).state * 1e-3 for mp in
                                                                                    model.flow_approach.macropores]
        elif type(model.flow_approach) == BypassFastFlow:
            x_solutes['concentration_mcg_per_m3_mp_' + s.Name][tstr] = [bp.conc(t, s) for bp in
                                                                        model.flow_approach.bypass]
    x_solutes['concentration_gw_recharge_mcg_per_m3'][tstr] = [model.gw.conc(t, T) for T in model.solutes]


def create_solver(model: CmfModel):
    """
    Creates a solver for the model run.

    :param model: CmfModel
    :return: cmf.CVodeKrylov
    """
    return cmf.CVodeKrylov(model, 1e-9)


def run(model: CmfModel, print_time=False):
    """
    Runs the model and saves results in previously generated xarray Datasets
    :param model: CmfModel
    :param print_time: boolean; if True current time of each step is printed
    :param spotpy_parameter: Spotpy parameters
    :return: xarray.Datasets, storing solute_results results of water fluxes and solute transport
    """

    start_timestamp = time.time()
    solver = create_solver(model)

    water_results = create_water_result_df()
    if model.solutes:
        phosphorus_results = create_phosphorus_result_df(model)
    else:
        phosphorus_results = False

    i = 0
    for t in solver.run(model.begin, model.tend, model.dt):
        if (time.time() - start_timestamp) >= 15 * 60:  # 10 minutes (in seconds)
            print('Timeout Error')
            return False, False

        if print_time:
            print(t)

        fill_water_result_df(model, water_results, i, t)
        if model.solutes:
            fill_phosphorus_result_df(model, phosphorus_results, i, t)
        i += 1

    end_timestamp = time.time()
    print('Run time: ', (end_timestamp - start_timestamp) / 60, ' min')

    return water_results, phosphorus_results
