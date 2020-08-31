# -*- coding: utf-8 -*-
"""
Created in Aug 2020

The aim of this script is to run the model in different ways:
a) single run with water (DONE)
b) single run with phosphorus (DONE)
c) spotpy with water (DONE)
d) spotpy with phosphorus (TODO)
e) validation of water and phosphorus routines (TODO)

TODO: Ich habe gerade das Problem, dass ich die Parameter aus einem File übernehmen will (siehe spotpy_run.py von
 ptrans). Hier herrscht etwas Chaos, wann ich welche Parameter übergebe: water_params und phosphorus_params werden in
 ModelInterface erstellt. Für Spotpy werden diese nicht nochmal überschrieben (oder?), also müsste ich eigentlich an
 dieser Stelle die Sets aus dem File übergeben. Für SingleRun hingegen werden diese (teilweise) überschrieben. Hierfür
 will ich mir die Möglichkeiten offen halten, die Parameter entweder direkt per Wert festzulegen, oder über ein File.
 Also muss ich rausfinden, an welcher Stelle und wie ich das am besten mache.
 Und zur Frage "wie": ist es sinnvoll, eine eigene Klasse dafür zu erstellen (wie bisher angefangen) um so ein bisschen
 mehr Ordnung zu halten? Da fehlt mir noch die Idee, wie ich das unkompliziert hinkriege. Alternativ könnte ich die
 Funktionen auch direkt in ModelInterface packen. Das würde die Anwendung vielleicht ein wenig vereinfachen, den Code
 aber vielleicht unübersichtlicher machen.

TODO: Wenn ich das geschafft habe würde ich mich gerne noch auf Fehlersuche begeben: Es werden immer noch häufig Fehler
 produziert. Dazu würde ich gerne einige der entsprechenden Datensätze direkt einlesen (über ein File?) und im
 SingleRun-Mode austesten, ob ich den/die Fehler finde.

TODO: Außerdem werden die letzten Läufe immer extrem langsam. Hierzu habe ich noch zwei Optionen dieses Problem zu
 untersuchen: zum einen kann ich mich belesen und vielleicht irgendwie austesten, ob der Speicher mit der Zeit vollläuft
 und wenn ja, wie ich ihn leer kriege. Und zum anderen kann ich doch nochmal schauen, ob ich eine effektivere Zeitsperre
 in run() reinkriege, sodass WIRKLICH nach z.B. 12 Minuten abgebrochen wird (ohne, dass auf den nächsten Schritt
 gewartet wird)

@author: pferdmenges-j
"""


import os
import parallel_processing
import cmf
import spotpy
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from build_the_model import CmfModel
import input_and_output as iao
import run_and_plot as rap


def parallel():
    """
    Returns 'mp_infiltration', if this code runs with MPI, else returns 'seq'
    :return:
    """
    return 'mpi' if 'OMPI_COMM_WORLD_SIZE' in os.environ else 'seq'


def u(name, low, high, default=None, doc=None):
    """
    Creates a uniform distributed parameter

    :param name: name of variable
    :param low: Minimum value
    :param high: Maximum value
    :param default: Default value
    :param doc: Documentation of the parameter (use reStructuredText)
    :return: The parameter object
    """
    if default is None:
        default = 0.5 * (low + high)
    return spotpy.parameter.Uniform(name, low, high, optguess=default, doc=doc)


def create_spotpy_water_parameters(vgm_via_spotpy=True, system=1):
    """
    Here, spotpy parameters for simulation of phosphorus are created.
    All states are in micrograms! Filter: 1: solute can cross barrier completely; 0: no solute is crossing
    """
    param_list = [u(name='saturated_depth', low=1.0, high=10.0, default=4.5, doc='saturated depth at beginning'),
                  u(name='puddle_depth', low=0.0, high=0.01, default=0.002,
                    doc='water depth at which runoff starts [m]'),
                  u(name='porosity_mx_ah', low=0.001, high=0.9, default=0.1,
                    doc='porosity of matrix [m3 Pores / m3 Soil]'),
                  u(name='porosity_mx_bv1', low=0.001, high=0.9, default=0.1,
                    doc='porosity of matrix [m3 Pores / m3 Soil]'),
                  u(name='porosity_mx_bv2', low=0.001, high=0.9, default=0.1,
                    doc='porosity of matrix [m3 Pores / m3 Soil]'),
                  u(name='w0_ah', low=0.8, high=1.0, default=0.99,
                    doc='VanGenuchtenMualem parameter w0, first layer'),
                  u(name='w0_bv1', low=0.8, high=1.0, default=0.99,
                    doc='VanGenuchtenMualem parameter w0, deepest layer'),
                  u(name='w0_bv2', low=0.8, high=1.0, default=0.99,
                    doc='VanGenuchtenMualem parameter w0, deepest layer')]

    if vgm_via_spotpy:
        # value ranges for n and alpha follow http://www.international-agrophysics.org/Relationship-between-van-
        # Genuchten-s-parameters-of-the-retention-curve-equation-nand,106597,0,2.html
        param_list.extend([u(name='ksat_mx_ah', low=0.1, high=20, default=5,
                             doc='saturated conductivity of ah layer in matrix [m/day]'),
                           u(name='ksat_mx_bv1', low=0.1, high=20, default=5,
                             doc='saturated conductivity of bv1 layer in matrix [m/day]'),
                           u(name='ksat_mx_bv2', low=0.1, high=20, default=5,
                             doc='saturated conductivity of bv2 layer in matrix [m/day]'),
                           u(name='n_ah', low=1.0, high=3.6, default=1.211,
                             doc='VanGenuchtenMualem parameter n of ah layer in matrix [m/day]'),
                           u(name='n_bv1', low=1.0, high=3.6, default=1.211,
                             doc='VanGenuchtenMualem parameter n of bv1 layer in matrix [m/day]'),
                           u(name='n_bv2', low=1.0, high=3.6, default=1.211,
                             doc='VanGenuchtenMualem parameter n of bv2 layer in matrix [m/day]'),
                           u(name='alpha_ah', low=0, high=1, default=0.2178,
                             doc='VanGenuchtenMualem parameter alpha of ah layer in matrix [m/day]'),
                           u(name='alpha_bv1', low=0, high=1, default=0.2178,
                             doc='VanGenuchtenMualem parameter alpha of bv1 layer in matrix [m/day]'),
                           u(name='alpha_bv2', low=0, high=1, default=0.2178,
                             doc='VanGenuchtenMualem parameter alpha of bv2 layer in matrix [m/day]')
                           ])

    if system == 2:
        param_list.append(
            u(name='ksat_mp', low=1, high=240, default=10, doc='saturated conductivity of macropores [m/day]'))
    elif system == 3:
        param_list.extend(
            [u(name='ksat_mp', low=1, high=240, default=10, doc='saturated conductivity of macropores [m/day]'),
             u(name='porefraction_mp', low=0.1, high=1.0, default=0.2, doc='macropore fraction [m3/m3]'),
             u(name='density_mp', low=0.001, high=1.0, default=0.05,
               doc='mean distance between the macropores [m]'),
             u(name='k_shape', low=0.0, high=1.0, default=0.0,
               doc='the shape of the conductivity in relation to the matric '
                   'potential of the micropore flow_approach')])

    return param_list


def create_spotpy_phosphorus_parameters(system=1):
    param_list = [u(name='dip_state', low=0, high=1000, default=10, doc='state of DIP'),
                  u(name='dop_state', low=0, high=1000, default=10, doc='state of DOP'),
                  u(name='pp_state', low=0, high=1000, default=10, doc='state of PP'),
                  u(name='mx_filter_dp', low=0, high=1, default=1, doc='Filter for DIP + DOP in matrix'),
                  u(name='mx_filter_pp', low=0, high=1, default=0.1, doc='Filter for PP in matrix')]
    if system == 2 or system == 3:
        param_list.extend([u(name='mp_filter_dp', low=0, high=1, default=1,
                             doc='Filter for DIP + DOP in macropores/bypass'),
                           u(name='mp_filter_pp', low=0, high=1, default=0.8,
                             doc='Filter for PP in macropores/bypass')])
    if system == 3:
        param_list.extend([u(name='exch_filter_dp', low=0, high=1, default=1,
                             doc='exchange of DIP and DOP between matrix and macropores'),
                           u(name='exch_filter_pp', low=0, high=1, default=0.8,
                             doc='exchange of PP between matrix and macropores')])
    return param_list


def set_parameters():
    if water_or_phosphorus == 'water':
        phosphorus = None
        if use_spotpy:
            water = create_spotpy_water_parameters(vgm_via_spotpy=vgm_params_via_spotpy, system=fastflow)
        else:
            if w_params_from_file:
                save_file = Path('results/SELECTION_water_FF' + str(fastflow) + '_P' + str(prof) + '.csv')
                if save_file.exists():
                    database = pd.read_csv(save_file, sep=',', decimal='.', engine='python', header=0)
                else:
                    database = ParameterFromFile(name=dbname + '.csv', method='percentage', value=0.01,
                                                 save_under=save_file)
                    database = database.df

                index_w = int(sys.argv[5]) if len(sys.argv) == 6 else 0
                water = WaterParameters(spotpy_set=database.iloc[[index_w]],
                                           spotpy_soil_params=vgm_params_via_spotpy, system=fastflow)
            else:
                water = WaterParameters(spotpy_soil_params=vgm_params_via_spotpy, system=fastflow)

    else:  # should be 'phosphorus'
        if use_spotpy:
            phosphorus = create_spotpy_phosphorus_parameters(system=fastflow)
            if w_params_from_file:
                save_file = Path(
                    'results/SELECTION_water_FF' + str(fastflow) + '_P' + str(prof) + '.csv')
                if save_file.exists():
                    database = pd.read_csv(save_file, sep=',', decimal='.', engine='python', header=0)
                else:
                    database = ParameterFromFile(name='results/LHS_water_FF' + str(fastflow) + '_P' + str(prof) +
                                                      '.csv', method='percentage', value=0.01,
                                                 save_under=save_file)
                    database = database.df

                index_w = int(sys.argv[5]) if len(sys.argv) == 6 else 0
                water = WaterParameters(spotpy_set=database.iloc[[index_w]],
                                           spotpy_soil_params=vgm_params_via_spotpy, system=fastflow)
            else:
                water = WaterParameters(spotpy_soil_params=vgm_params_via_spotpy, system=fastflow)
        else:
            if w_params_from_file and p_params_from_file:
                save_w_file = Path(
                    'results/SELECTION_water_FF' + str(fastflow) + '_P' + str(prof) + '.csv')
                save_p_file = Path(
                    'results/SELECTION_phosphorus_FF' + str(fastflow) + '_P' + str(prof) + '.csv')
                if save_w_file.exists():
                    database_w = pd.read_csv(save_w_file, sep=',', decimal='.', engine='python', header=0)
                else:
                    database_w = ParameterFromFile(name='results/LHS_water_FF' + str(fastflow) + '_P' + str(prof) +
                                                        '.csv', method='percentage', value=0.01,
                                                   save_under=save_w_file)
                    database_w = database_w.df

                if save_p_file.exists():
                    database_p = pd.read_csv(save_p_file, sep=',', decimal='.', engine='python', header=0)
                else:
                    database_p = ParameterFromFile(name=dbname + '.csv', method='percentage', value=0.01,
                                                   save_under=save_w_file)
                    database_p = database_p.df

                index_w = int(sys.argv[5]) if len(sys.argv) == 7 else 0
                index_p = int(sys.argv[6]) if len(sys.argv) == 7 else 0

                water = WaterParameters(spotpy_set=database_w.iloc[[index_w]],
                                           spotpy_soil_params=vgm_params_via_spotpy, system=fastflow)
                phosphorus = PhosphorusParameters(spotpy_set=database_w.iloc[[index_w]], system=fastflow)
            elif not w_params_from_file and not p_params_from_file:
                water = WaterParameters(spotpy_soil_params=vgm_params_via_spotpy, system=fastflow)
                phosphorus = PhosphorusParameters(system=fastflow)
            elif w_params_from_file and not p_params_from_file:
                save_w_file = Path(
                    'results/SELECTION_water_FF' + str(fastflow) + '_P' + str(prof) + '.csv')
                if save_w_file.exists():
                    database_w = pd.read_csv(save_w_file, sep=',', decimal='.', engine='python', header=0)
                else:
                    database_w = ParameterFromFile(name='results/LHS_water_FF' + str(fastflow) + '_P' + str(prof) +
                                                        '.csv', method='percentage', value=0.01,
                                                   save_under=save_w_file)
                    database_w = database_w.df

                index_w = int(sys.argv[6]) if len(sys.argv) == 7 else 0

                water = WaterParameters(spotpy_set=database_w.iloc[[index_w]],
                                           spotpy_soil_params=vgm_params_via_spotpy, system=fastflow)
                phosphorus = PhosphorusParameters(system=fastflow)
            else:
                save_p_file = Path(
                    'results/SELECTION_phosphorus_FF' + str(fastflow) + '_P' + str(prof) + '.csv')
                if save_p_file.exists():
                    database_p = pd.read_csv(save_p_file, sep=',', decimal='.', engine='python', header=0)
                else:
                    database_p = ParameterFromFile(name=dbname + '.csv', method='percentage', value=0.01,
                                                   save_under=save_p_file)
                    database_p = database_p.df

                index_p = int(sys.argv[6]) if len(sys.argv) == 7 else 0

                phosphorus = PhosphorusParameters(spotpy_set=database_p.iloc[[index_p]], system=fastflow)
                water = WaterParameters(spotpy_soil_params=vgm_params_via_spotpy, system=fastflow)

    return water, phosphorus


class ModelInterface:
    """
    Class to create a CmfModel and run it via Spotpy for calibration
    """

    def __init__(self, water_params, phosphorus_params, spotpy_soil_params=True,
                 irrigation=1, profile=1, flow_approach=3, mode='water'):
        self.project = None
        self.mode = mode
        self.flow_approach = flow_approach
        self.water_params = water_params
        self.phosphorus_params = phosphorus_params

        self.irrigation = irrigation
        self.profile = profile
        self.spotpy_soil_params = spotpy_soil_params

        self.begin = cmf.Time(12, 6, 2018, 9, 00)  # starting time of solute_results
        self.dt = cmf.min  # time steps (cmf.sec, cmf.min, cmf.h, cmf.day, cmf.week, cmf.month, cmf.year)

        self.evaluation_df = iao.evaluation_water_df(source=Path('input/MIT_Evaluation.csv'),
                                                     irrigation=self.irrigation, profile=self.profile)
        self.error_file = Path('results/LHS_FF' + str(self.flow_approach) + '_P' + str(self.profile) + '_errors.csv')

        if self.mode == 'phosphorus':
            iao.write_error_file(spotpy=self.phosphorus_params, name=self.error_file)
            self.evaluation_df = iao.evaluation_phosphorus_df(self.evaluation_df)
            self.tracer = 'dip dop pp'
        else:
            iao.write_error_file(spotpy=self.water_params, name=self.error_file)
            self.tracer = ''

        self.sim_stop = max(self.evaluation_df['time [min]'])  # time when last measurement was taken (min after start)

    def parameters(self):
        if self.mode == 'phosphorus':
            return spotpy.parameter.generate(self.phosphorus_params)
        else:
            return spotpy.parameter.generate(self.water_params)

    def set_parameters(self, vector):  # Probably it is not possible to have 2 spotpy sets?
        if self.mode == 'phosphorus':
            phosphorus_params = vector
            water_params = self.water_params
        else:
            phosphorus_params = None
            water_params = vector

        return phosphorus_params, water_params

    def simulation(self, vector):
        """
        Creates the model via calling the constructor and initiates the model run.

        :param vector: spotpy parameters
        :return: list with solute_results results, adjusted to match the evaluation list
        """
        empty_list = [np.nan] * len(self.evaluation_df['amount_measured_absolute'].tolist())
        empty_list = [list(empty_list), list(empty_list)]

        try:
            phosphorus_params, water_params = self.set_parameters(vector)
        except:
            print('Parameter Error')
            iao.append_error_file(spotpy=vector, name=self.error_file, error='ParameterError')
            return empty_list

        try:
            self.project = CmfModel(water_params=water_params,
                                    phosphorus_params=phosphorus_params,
                                    spotpy_soil_params=self.spotpy_soil_params,
                                    irrigation=self.irrigation,
                                    profile=self.profile,
                                    fast_component=self.flow_approach,
                                    tracer=self.tracer,
                                    begin=self.begin,
                                    cell=(0, 0, 0, 1000, True),  # IMPORTANT: now the area is 1000 m2
                                    surface_runoff=True)
        except:
            print('Setup Error')
            iao.append_error_file(spotpy=vector, name=self.error_file, error='Setup')
            return empty_list

        try:
            water_results, phosphorus_results = rap.run(self.project, print_time=False)

            if self.mode == 'phosphorus':
                # TODO: phosphorus results need to be formatted
                simulation = iao.format_phosphorus_results(self, phosphorus_results)
                return [list(simulation['dip_simulated_mcg_per_m3_mx+mp']) +
                        list(simulation['dop_simulated_mcg_per_m3_mx+mp']) +
                        list(simulation['pp_simulated_mcg_per_m3_mx+mp']),
                        list(simulation['dip_simulated_state_per_m2_mx+mp']) +
                        list(simulation['dop_simulated_state_per_m2_mx+mp']) +
                        list(simulation['pp_simulated_state_per_m2_mx+mp'])]
                # return simulation
            else:
                simulation = iao.format_water_results(self, water_results)
                return [list(simulation['simulated_flux_l_per_m2_per_day']),
                        # flux rate is independent from actual water amount
                        list(simulation['amount_simulated_l_per_m2'])]
        except:  # RuntimeError:
            print('CVode Error')
            iao.append_error_file(spotpy=vector, name=self.error_file, error='CVodeError')
            return empty_list

    def evaluation(self):
        """
        :return: the evaluation data
        """
        if self.mode == 'phosphorus':
            observations = [list(self.evaluation_df['dip_measured_mcg_per_m3']) +
                            list(self.evaluation_df['dop_measured_mcg_per_m3']) +
                            list(self.evaluation_df['pp_measured_mcg_per_m3']),
                            list(self.evaluation_df['dip_measured_state_per_m2']) +
                            list(self.evaluation_df['dop_measured_state_per_m2']) +
                            list(self.evaluation_df['pp_measured_state_per_m2'])]
            return observations
        else:
            return [list(self.evaluation_df['measured_flux_l_per_m2_per_day']),  # flow rate
                    list(self.evaluation_df['amount_measured_l_per_m2'])]  # absolute water amount (per m2)

    def objectivefunction(self, simulation, evaluation):
        """
        Calculates the goodness of the solute_results.

        :param simulation: list of solute_results data
        :param evaluation: list of evaluation data
        :return: objective function
        """
        if self.mode == 'phosphorus':
            obj1 = -spotpy.objectivefunctions.rrmse(evaluation[0], simulation[0])  # mcg per m3 water
            obj2 = -spotpy.objectivefunctions.rrmse(evaluation[1], simulation[1])  # total amount per m2 soil
            # return obj1  # alternative: return [obj1, obj2]
            return [obj1, obj2]
        else:
            obj1 = -spotpy.objectivefunctions.rrmse(evaluation[0], simulation[0])  # Flux data [m3 per day]
            obj2 = -spotpy.objectivefunctions.rrmse(evaluation[1], simulation[1])  # Total water amount data [l per m2]
            # return obj1  # alternative: return [obj1, obj2]
            return -spotpy.objectivefunctions.rrmse(evaluation, simulation)


class WaterParameters:
    def __init__(self, spotpy_set=None, spotpy_soil_params=True, system=1):
        """
        Here, spotpy parameters are extracted from a row of spotpy results. These parameters then are used to create a
        new model
        :param spotpy_set: row of spotpy results
        :param system: 1 for matrix flow only, 2 for bypass flow, and 3 for macropores
        """
        self.saturated_depth = spotpy_set.parsaturated_depth if spotpy_set else 4.76
        self.puddle_depth = spotpy_set.parpuddle_depth if spotpy_set else 0.004276
        self.porosity_mx_ah = spotpy_set.parporosity_mx_ah if spotpy_set else 0.8057
        self.porosity_mx_bv1 = spotpy_set.parporosity_mx_bv1 if spotpy_set else 0.0909
        self.porosity_mx_bv2 = spotpy_set.parporosity_mx_bv2 if spotpy_set else 0.7163
        self.w0_ah = spotpy_set.parw0_ah if spotpy_set else 0.9517
        self.w0_bv1 = spotpy_set.parw0_bv1 if spotpy_set else 0.843
        self.w0_bv2 = spotpy_set.parw0_bv2 if spotpy_set else 0.855

        if spotpy_soil_params:
            self.ksat_mx_ah = spotpy_set.parksat_mx_ah if spotpy_set else 14.63
            self.ksat_mx_bv1 = spotpy_set.parksat_mx_bv1 if spotpy_set else 3.541
            self.ksat_mx_bv2 = spotpy_set.parksat_mx_bv2 if spotpy_set else 0.7764
            self.n_ah = spotpy_set.n_ah if spotpy_set else 1.211
            self.n_bv1 = spotpy_set.n_bv1 if spotpy_set else 1.211
            self.n_bv2 = spotpy_set.n_bv2 if spotpy_set else 1.211
            self.alpha_ah = spotpy_set.alpha_ah if spotpy_set else 0.2178
            self.alpha_bv1 = spotpy_set.alpha_bv1 if spotpy_set else 0.2178
            self.alpha_bv2 = spotpy_set.alpha_bv2 if spotpy_set else 0.2178

        if system == 2:
            self.ksat_mp = spotpy_set.parksat_mp if spotpy_set else 10
        elif system == 3:
            self.ksat_mp = spotpy_set.parksat_mp if spotpy_set else 62.7772
            self.porefraction_mp = spotpy_set.parporefraction_mp if spotpy_set else 0.284378
            self.density_mp = spotpy_set.pardensity_mp if spotpy_set else 0.96332
            self.k_shape = spotpy_set.park_shape if spotpy_set else 0.01


class PhosphorusParameters:
    def __init__(self, spotpy_set=None, system=1):
        """
        Here, spotpy parameters are extracted from a row of spotpy results. These parameters then are used to create a
        new model
        :param spotpy_set: row of spotpy results
        :param system: 1 for matrix flow only, 2 for bypass flow, and 3 for macropores
        """
        self.dip_state = spotpy_set.pardip_state if spotpy_set else 10
        self.dop_state = spotpy_set.pardop_state if spotpy_set else 10
        self.pp_state = spotpy_set.parpp_state if spotpy_set else 10
        self.mx_filter_dp = spotpy_set.parmx_filter_dp if spotpy_set else 1
        self.mx_filter_pp = spotpy_set.parmx_filter_pp if spotpy_set else 0.1

        if system == 2 or system == 3:
            self.mp_filter_dp = spotpy_set.parmp_filter_dp if spotpy_set else 1
            self.mp_filter_pp = spotpy_set.parmp_filter_pp if spotpy_set else 0.8
        if system == 3:
            self.exch_filter_dp = spotpy_set.parexch_filter_dp if spotpy_set else 1
            self.exch_filter_pp = spotpy_set.parexch_filter_pp if spotpy_set else 0.8


class ParameterFromFile:
    def __init__(self, name, method, value, save_under):
        """
        First reading the csv file containg results from spotpy and sorting it by the objective function;
        then choosing one of the two rejection approaches

        :param name: name of data file, which will be read and sorted
        :param method: either 'percentage' or 'threshold' (or 'testing')
        :param value: when method=='percentage' this should be the proportion; otherwise this should be a threshold
        """

        self.df = pd.read_csv(name, sep=',', decimal='.', engine='python', header=0)
        self.df = self.df.sort_values(by=['like1'], ascending=False).reset_index(drop=True)

        if method == 'threshold':
            self.df = self.model_rejection_by_threshold(self.df, value)
        elif method == 'percentage':
            self.df = self.model_rejection_by_percentage(self.df, value)
        elif method == 'number':
            self.df = self.model_rejection_by_number(self.df, value)

        self.df.to_csv(save_under)

    def model_rejection_by_percentage(self, df, value=0.01):
        """
        Option 1: when for example the best 1 % of the spotpy runs should be used for validation, this function
        needs to be used.

        :param df: sorted data frame (from read_db())
        :param value: Proportion of to be selected parameter sets
        """
        n = int(len(df.index) * value)
        df = df.truncate(after=n - 1)  # -1 since we need the index, which starts with 0
        return df

    def model_rejection_by_threshold(self, df, value=-0.5):
        """
        Option 2: when all parameter sets better than a specific threshold should be used for validation, this function
        needs to be used.

        :param df: sorted data frame (from read_db())
        :param value: Threshold of the objective value (depending of used objective function)
        """
        df = df[df['like1'] > value]
        return df

    def model_rejection_by_number(self, df, value=10):
        """
        Option 3: This function chooses an absolute number of parameter sets. This can be used for testing of the
        functionality

        :param df: sorted data frame (from read_db())
        :param value: number of to be chosen parameter sets
        """
        df = df.truncate(after=value - 1)  # -1 since we need the index, which starts with 0
        return df


class SingleRun:
    def __init__(self, init):
        init.project = CmfModel(water_params=init.water_params,
                                phosphorus_params=init.phosphorus_params,
                                spotpy_soil_params=init.spotpy_soil_params,
                                irrigation=init.irrigation,
                                profile=init.profile,
                                fast_component=init.flow_approach,
                                tracer=init.tracer,
                                begin=init.begin,
                                cell=(0, 0, 0, 1000, True),  # IMPORTANT: now the area is 1000 m2
                                surface_runoff=True)

        self.water_results, self.phosphorus_results = rap.run(init.project, print_time=True)


if __name__ == '__main__':
    if len(sys.argv) != 5:
        irr = 1
        prof = 1
        fastflow = 2
        runs = 20
    else:
        irr = int(sys.argv[1])
        prof = int(sys.argv[2])
        fastflow = int(sys.argv[3])
        runs = int(sys.argv[4])

    water_or_phosphorus = 'water'  # 'water' or 'phosphorus'
    dbname = 'results/LHS_' + water_or_phosphorus + '_FF' + str(fastflow) + '_P' + str(prof)
    vgm_params_via_spotpy = True
    use_spotpy = False
    w_params_from_file = False
    p_params_from_file = False

    w_params, p_params = set_parameters()

    setup = ModelInterface(water_params=w_params, phosphorus_params=p_params,
                           spotpy_soil_params=vgm_params_via_spotpy, irrigation=irr, profile=prof,
                           flow_approach=fastflow, mode=water_or_phosphorus)

    if use_spotpy:
        sampler = spotpy.algorithms.lhs(setup, parallel=parallel(), dbname=dbname, dbformat='csv')
        sampler.sample(runs)
    else:
        single_run = SingleRun(setup)
