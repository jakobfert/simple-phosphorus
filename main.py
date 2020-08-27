# -*- coding: utf-8 -*-
"""
Created in Aug 2020

The aim of this script is to run the model in different ways:
a) single run with water (DONE)
b) single run with phosphorus (DONE)
c) spotpy with water (DONE)
d) spotpy with phosphorus (TODO)
e) validation of water and phosphorus routines (TODO)

@author: pferdmenges-j
"""


import os
import parallel_processing
import cmf
import spotpy
import sys
from pathlib import Path
import numpy as np
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


class SpotpyInterface:
    """
    Class to create a CmfModel and run it via Spotpy for calibration
    """

    def __init__(self, spotpy_set=None, spotpy_soil_params=True,
                 irrigation=1, profile=1, flow_approach=3, mode='water'):
        self.project = None
        self.mode = mode
        self.flow_approach = flow_approach
        self.spotpy_set = spotpy_set

        self.irrigation = irrigation
        self.profile = profile
        self.spotpy_soil_params = spotpy_soil_params

        self.begin = cmf.Time(12, 6, 2018, 9, 00)  # starting time of solute_results
        self.dt = cmf.min  # time steps (cmf.sec, cmf.min, cmf.h, cmf.day, cmf.week, cmf.month, cmf.year)

        self.evaluation_df = iao.evaluation_water_df(source=Path('input/MIT_Evaluation.csv'),
                                                     irrigation=self.irrigation, profile=self.profile)
        self.error_file = Path('results/LHS_FF' + str(self.flow_approach) + '_P' + str(self.profile) + '_errors.csv')

        if self.mode == 'phosphorus':
            self.phosphorus_params = self.create_phosphorus_parameters()
            self.water_params = WaterParameters(spotpy_set=self.spotpy_set,
                                                spotpy_soil_params=self.spotpy_soil_params,
                                                system=self.flow_approach)
            iao.write_error_file(spotpy=self.phosphorus_params, name=self.error_file)
            self.evaluation_df = iao.evaluation_phosphorus_df(self.evaluation_df)
            self.tracer = 'dip dop pp'
        else:
            self.water_params = self.create_water_parameters()
            self.phosphorus_params = None
            iao.write_error_file(spotpy=self.water_params, name=self.error_file)
            self.tracer = ''

        self.sim_stop = max(self.evaluation_df['time [min]'])  # time when last measurement was taken (min after start)

    def create_water_parameters(self):
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

        if self.spotpy_soil_params:
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

        if self.flow_approach == 2:
            param_list.append(
                u(name='ksat_mp', low=1, high=240, default=10, doc='saturated conductivity of macropores [m/day]'))
        elif self.flow_approach == 3:
            param_list.extend(
                [u(name='ksat_mp', low=1, high=240, default=10, doc='saturated conductivity of macropores [m/day]'),
                 u(name='porefraction_mp', low=0.1, high=1.0, default=0.2, doc='macropore fraction [m3/m3]'),
                 u(name='density_mp', low=0.001, high=1.0, default=0.05,
                   doc='mean distance between the macropores [m]'),
                 u(name='k_shape', low=0.0, high=1.0, default=0.0,
                   doc='the shape of the conductivity in relation to the matric '
                       'potential of the micropore flow_approach')])

        return param_list

    def create_phosphorus_parameters(self):
        param_list = [spotpy.parameter.Uniform('dip_state', 0, 1000, optguess=10, doc='state of DIP'),
                      spotpy.parameter.Uniform('dop_state', 0, 1000, optguess=10, doc='state of DOP'),
                      spotpy.parameter.Uniform('pp_state', 0, 1000, optguess=10, doc='state of PP'),
                      spotpy.parameter.Uniform('mx_filter_dp', 0, 1, optguess=1, doc='Filter for DIP + DOP in matrix'),
                      spotpy.parameter.Uniform('mx_filter_pp', 0, 1, optguess=0.1, doc='Filter for PP in matrix')]
        if self.flow_approach == 2 or self.flow_approach == 3:
            param_list.extend([spotpy.parameter.Uniform('mp_filter_dp', 0, 1, optguess=1,
                                                        doc='Filter for DIP + DOP in macropores/bypass'),
                               spotpy.parameter.Uniform('mp_filter_pp', 0, 1, optguess=0.8,
                                                        doc='Filter for PP in macropores/bypass')])
        if self.flow_approach == 3:
            param_list.extend([spotpy.parameter.Uniform('exch_filter_dp', 0, 1, optguess=1,
                                                        doc='exchange of DIP and DOP between matrix and macropores'),
                               spotpy.parameter.Uniform('exch_filter_pp', 0, 1, optguess=0.8,
                                                        doc='exchange of PP between matrix and macropores')])
        return param_list

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


class SingleRun:
    def __init__(self, init):
        if init.mode == 'phosphorus':
            init.phosphorus_params = PhosphorusParameters(spotpy_set=init.spotpy_set, system=init.flow_approach)
        else:
            init.water_params = WaterParameters(spotpy_set=init.spotpy_set, spotpy_soil_params=init.spotpy_soil_params,
                                                system=init.flow_approach)

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
        fastflow = 3
        runs = 20
    else:
        irr = int(sys.argv[1])
        prof = int(sys.argv[2])
        fastflow = int(sys.argv[3])
        runs = int(sys.argv[4])

    dbname = 'results/LHS'
    water_or_phosphorus = 'water'  # 'water' or 'phosphorus'
    use_spotpy = True

    setup = SpotpyInterface(spotpy_set=None, spotpy_soil_params=True,
                            irrigation=irr, profile=prof, flow_approach=fastflow, mode=water_or_phosphorus)

    if use_spotpy:
        sampler = spotpy.algorithms.lhs(setup, parallel=parallel(), dbname=dbname, dbformat='csv')
        sampler.sample(runs)
    else:
        single_run = SingleRun(setup)
