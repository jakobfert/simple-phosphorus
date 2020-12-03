# -*- coding: utf-8 -*-
"""
Created in Aug 2020

This file will hold functions to run the script and to plot the results

@author: pferdmenges-j
"""

import cmf
import numpy as np
import time
from build_the_model import CmfModel, MacroporeFastFlow, BypassFastFlow
import input_and_output as iao
import matplotlib.pyplot as plt
from spotpy.objectivefunctions import rrmse


def create_water_result_df():
    """
    Creates empty dataframe to fill later with results of water fluxes.

    :return: dataset to save water results
    """

    water = {'wetness_mx': {}, 'wetness_mp': {}, 'water_volume_l_per_m2_mx': {}, 'water_volume_l_per_m2_mp': {},
             'percolation_l_per_m2_per_day_mx': {}, 'percolation_l_per_m2_per_day_mp': {},
             'gw_recharge_wb_l_per_m2_per_day': {}, 'gw_recharge_flux_l_per_m2_per_day': {},
             'surface_runoff_l_per_m2_per_day': {}, 'infiltration_l_per_m2_per_day': {},
             'ex_l_per_m2_per_day_mp_mx': {}, 'simulated_flux_l_per_m2_per_day': {},
             }

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
                           'concentration_flux_mcg_per_m3_mp_' + s.Name: {},
                           s.Name + '_simulated_mcg_per_m3_mx+mp': {},
                           s.Name + '_simulated_state_per_m2_mx+mp': {}})
    return phosphorus


def fill_water_result_df(model: CmfModel, df, t):
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
        df['simulated_flux_l_per_m2_per_day'][tstr] = np.add(df['percolation_l_per_m2_per_day_mx'][tstr],
                                                             df['percolation_l_per_m2_per_day_mp'][tstr])
    else:
        df['simulated_flux_l_per_m2_per_day'][tstr] = df['percolation_l_per_m2_per_day_mx'][tstr]


def total_concentration_mcg_per_m3(model, p_df, w_df, solute, t_str):
    # Note: here, concentration (mcg / m3) is multiplied with l/(m2*day). This is correct, since l/m2 = m3/1000m2.
    # Moreover, it is divided by percolating water, so it actually doesn't matter.
    if type(model.flow_approach) == MacroporeFastFlow or type(model.flow_approach) == BypassFastFlow:
        conc = [(a * b + c * d) / (b + d) if (b + d) else 0 for a, b, c, d in
                zip(p_df['concentration_flux_mcg_per_m3_mx_' + solute.Name][t_str],
                    w_df['percolation_l_per_m2_per_day_mx'][t_str],
                    p_df['concentration_flux_mcg_per_m3_mp_' + solute.Name][t_str],
                    w_df['percolation_l_per_m2_per_day_mp'][t_str])]

    else:
        conc = p_df['concentration_flux_mcg_per_m3_mx_' + solute.Name][t_str]

    return conc


def total_amount_mcg_per_m2(model, p_df, w_df, solute, t_str):
    if type(model.flow_approach) == MacroporeFastFlow or type(model.flow_approach) == BypassFastFlow:
        amount = [a * (b + c) / model.c.area for a, b, c in zip(
            total_concentration_mcg_per_m3(model, p_df, w_df, solute, t_str),
            w_df['percolation_l_per_m2_per_day_mx'][t_str], w_df['percolation_l_per_m2_per_day_mp'][t_str])]
    else:
        amount = [a * b / model.c.area for a, b in zip(total_concentration_mcg_per_m3(model, p_df, w_df, solute, t_str),
                                                       w_df['percolation_l_per_m2_per_day_mx'][t_str])]
    return amount


def fill_phosphorus_result_df(model: CmfModel, df, water_df, t):
    """
    Function which is called each time step of solute_results to add values to df
    :param model: CmfModel
    :param df: xarray Dataset for solute_results results of solute transport
    :param i: value increasing each time step +1 for selecting the right location in dataset
    :param t: current time
    :return: filled x_solute xarray
    """
    tstr = str(t)
    for s in model.solutes:
        df['concentration_mcg_per_m3_mx_' + s.Name][tstr] = [layer.conc(s) for layer in model.c.layers]
        df['concentration_flux_mcg_per_m3_mx_' + s.Name][tstr] = [model.mx_infiltration.conc(t, s)] + [
            flux.conc(t, s) for flux in model.mx_percolation]
        df['simulated_state_mcg_per_m2_per_layer_mx_' + s.Name][tstr] = [layer.Solute(s).state / model.c.area for layer
                                                                         in model.c.layers]
        if type(model.flow_approach) == MacroporeFastFlow:
            df['concentration_mcg_per_m3_mp_' + s.Name][tstr] = [mp.conc(s) for mp in
                                                                 model.flow_approach.macropores]
            df['concentration_flux_mcg_per_m3_mp_' + s.Name][tstr] = [model.flow_approach.mp_infiltration.conc(
                t, s)] + [flux.conc(t, s) for flux in model.flow_approach.mp_percolation]
            df['simulated_state_mcg_per_m2_per_layer_mp_' + s.Name][tstr] = [mp.Solute(s).state / model.c.area for mp in
                                                                             model.flow_approach.macropores]
        elif type(model.flow_approach) == BypassFastFlow:
            df['concentration_flux_mcg_per_m3_mp_' + s.Name][tstr] = [bp.conc(t, s) for bp in
                                                                      model.flow_approach.bypass]

        df[s.Name + '_simulated_mcg_per_m3_mx+mp'][tstr] = total_concentration_mcg_per_m3(model, df, water_df, s, tstr)
        df[s.Name + '_simulated_state_per_m2_mx+mp'][tstr] = total_amount_mcg_per_m2(model, df, water_df, s, tstr)

    df['concentration_gw_recharge_mcg_per_m3'][tstr] = [model.gw.conc(t, T) for T in model.solutes]


def create_solver(model: CmfModel):
    """
    Creates a solver for the model run.

    :param model: CmfModel
    :return: cmf.CVodeKrylov
    """
    return cmf.CVodeKrylov(model, 1e-9)


def save_to_csv(df, name):
    """
    This function is just for testing. It allows to save specific data to file for checking...
    """
    import csv
    with open(name + '.csv', mode='w', newline='') as file:
        w = csv.writer(file)
        w.writerow(df.keys())
        w.writerows(zip(*df.values()))

    import math
    min_val = math.inf
    max_val = -math.inf
    for key in df:
        min_key = min(df[key])
        max_key = max(df[key])
        min_val = min_key if min_key < min_val else min_val
        max_val = max_key if max_key > max_val else max_val

    print('Maximum Value for ' + name + ': ', max_val)
    print('Minimum Value for ' + name + ': ', min_val)


def result_evaluation(model: CmfModel, df):
    for s in model.solutes:
        save_to_csv(df['concentration_mcg_per_m3_mx_' + s.Name], name='concentration_mcg_per_m3_mx_' + s.Name)
        save_to_csv(df['concentration_flux_mcg_per_m3_mx_' + s.Name], name='concentration_flux_mcg_per_m3_mx_' + s.Name)
        save_to_csv(df['simulated_state_mcg_per_m2_per_layer_mx_' + s.Name],
                    name='simulated_state_mcg_per_m2_per_layer_mx_' + s.Name)
        save_to_csv(df['concentration_mcg_per_m3_mp_' + s.Name], name='concentration_mcg_per_m3_mp_' + s.Name)
        save_to_csv(df['concentration_flux_mcg_per_m3_mp_' + s.Name], name='concentration_flux_mcg_per_m3_mp_' + s.Name)
        save_to_csv(df['simulated_state_mcg_per_m2_per_layer_mp_' + s.Name],
                    name='simulated_state_mcg_per_m2_per_layer_mp_' + s.Name)
        save_to_csv(df[s.Name + '_simulated_mcg_per_m3_mx+mp'], name=s.Name + '_simulated_mcg_per_m3_mx+mp')
        save_to_csv(df[s.Name + '_simulated_state_per_m2_mx+mp'], name=s.Name + '_simulated_state_per_m2_mx+mp')


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

        fill_water_result_df(model, water_results, t)
        if model.solutes:
            fill_phosphorus_result_df(model, phosphorus_results, water_results, t)
        i += 1

    end_timestamp = time.time()
    if print_time:
        print('Run time: ', (end_timestamp - start_timestamp) / 60, ' min')

    result_evaluation(model, phosphorus_results)

    return water_results, phosphorus_results


def plotting(model, results):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))

    if model.mode == 'phosphorus':
        simulation = iao.format_phosphorus_results(model, results.phosphorus_results, results.water_results)
        sim1 = (simulation['dp_simulated_mcg_per_m3_mx+mp'] + simulation['pp_simulated_mcg_per_m3_mx+mp'])
        sim2 = (simulation['dp_simulated_state_per_m2_mx+mp'] + simulation['pp_simulated_state_per_m2_mx+mp'])
        sim_total = (simulation['dp_simulated_mcg_per_m3_mx+mp'] + simulation['pp_simulated_mcg_per_m3_mx+mp'] +
                     simulation['dp_simulated_state_per_m2_mx+mp'] + simulation['pp_simulated_state_per_m2_mx+mp'])
        eval1 = (list(model.evaluation_df['dp_measured_mcg_per_m3']) +
                 list(model.evaluation_df['pp_measured_mcg_per_m3']))
        eval2 = (list(model.evaluation_df['dp_measured_state_per_m2']) +
                 list(model.evaluation_df['pp_measured_state_per_m2']))

        eval_total = (list(model.evaluation_df['dp_measured_mcg_per_m3']) +
                      list(model.evaluation_df['pp_measured_mcg_per_m3']) +
                      list(model.evaluation_df['dp_measured_state_per_m2']) +
                      list(model.evaluation_df['pp_measured_state_per_m2']))

        # sim1 = (simulation['dip_simulated_mcg_per_m3_mx+mp'] + simulation['dop_simulated_mcg_per_m3_mx+mp'] +
        #         simulation['pp_simulated_mcg_per_m3_mx+mp'])
        # sim2 = (simulation['dip_simulated_state_per_m2_mx+mp'] + simulation['dop_simulated_state_per_m2_mx+mp'] +
        #         simulation['pp_simulated_state_per_m2_mx+mp'])

        # sim_total = (simulation['dip_simulated_mcg_per_m3_mx+mp'] + simulation['dop_simulated_mcg_per_m3_mx+mp'] +
        #              simulation['pp_simulated_mcg_per_m3_mx+mp'] + simulation['dip_simulated_state_per_m2_mx+mp'] +
        #              simulation['dop_simulated_state_per_m2_mx+mp'] + simulation['pp_simulated_state_per_m2_mx+mp'])

        # eval1 = (list(model.evaluation_df['dip_measured_mcg_per_m3']) +
        #          list(model.evaluation_df['dop_measured_mcg_per_m3']) +
        #          list(model.evaluation_df['pp_measured_mcg_per_m3']))
        # eval2 = (list(model.evaluation_df['dip_measured_state_per_m2']) +
        #          list(model.evaluation_df['dop_measured_state_per_m2']) +
        #          list(model.evaluation_df['pp_measured_state_per_m2']))
        #
        # eval_total = (list(model.evaluation_df['dip_measured_mcg_per_m3']) +
        #               list(model.evaluation_df['dop_measured_mcg_per_m3']) +
        #               list(model.evaluation_df['pp_measured_mcg_per_m3']) +
        #               list(model.evaluation_df['dip_measured_state_per_m2']) +
        #               list(model.evaluation_df['dop_measured_state_per_m2']) +
        #               list(model.evaluation_df['pp_measured_state_per_m2']))

        ax[0].set_ylabel('phosphorus flux [mcg per m3 water]')
        ax[1].set_ylabel('phosphorus state [mcg per m2 soil]')

        print('RRMSE mcg per m3: ', rrmse(eval1, sim1))
        print('RRMSE P state: ', rrmse(eval2, sim2))
        print('RRMSE total: ', rrmse(eval_total, sim_total))
    else:
        simulation = iao.format_water_results(model, results.water_results)
        sim1 = simulation['simulated_flux_l_per_m2_per_day']
        sim2 = simulation['amount_simulated_l_per_m2']

        sim_total = simulation['simulated_flux_l_per_m2_per_day'] + simulation['amount_simulated_l_per_m2']

        eval1 = list(model.evaluation_df['measured_flux_l_per_m2_per_day'])
        eval2 = list(model.evaluation_df['amount_measured_l_per_m2'])

        eval_total = (list(model.evaluation_df['measured_flux_l_per_m2_per_day']) +
                      list(model.evaluation_df['amount_measured_l_per_m2']))

        ax[0].set_ylabel('water flux [L m-2 s-1]')
        ax[1].set_ylabel('water amount [L m-2]')

        print('RRMSE flux [L m-2 day-1]: ', rrmse(eval1, sim1))
        print('RRMSE amount [L m-2]: ', rrmse(eval2, sim2))
        print('RRMSE total: ', rrmse(eval_total, sim_total))

    ax[0].plot(sim1, linestyle='-', color='green', label='simulated')
    ax[0].plot(eval1, linestyle='--', color='red', label='measured')

    ax[1].plot(sim2, linestyle='-', color='green', label='simulated')
    ax[1].plot(eval2, linestyle='--', color='red', label='measured')

    plt.legend(loc='upper right')
    ax[0].set_xlabel('time steps')
    ax[1].set_xlabel('time steps')

    plt.show()
