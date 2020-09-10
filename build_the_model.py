# -*- coding: utf-8 -*-
"""
Created in August 2020

This script will be used to build the model. It will provide functions for the different setup options and so on.
In main.py, the model will be created by choosing from the options given here.

@author: pferdmenges-j
"""

import cmf
import numpy as np
from math import sqrt
from pathlib import Path
import input_and_output as iao


# ------------------------------------------- FLOW OPTION 2: BYPASS -------------------------------------------
class BypassFastFlow:
    def __init__(self, model, ksat_mp):
        """
        Creates a list of fast bypass components, channeling water (and solutes) directly
        from the surface to every soil layer.

        :param model: the CmfModel object.
        :param ksat_mp: the saturated conductivity of the bypass components in m/day.
        """
        self.bypass = []
        for i in model.c.layers:
            self.bypass.append(cmf.LayerBypass(model.c.surfacewater, i, Kmax=ksat_mp))
        self.macropores = None


# ------------------------------------------- MACROPORE CLASS -------------------------------------------
class MacroPoreList(list):
    @property
    def wetness(self):
        return np.array([mp.filled_fraction for mp in self])

    @property
    def potential(self):
        return np.array([mp.potential for mp in self])

    @property
    def capacity(self):
        return np.array([mp.get_capacity() for mp in self])

    @property
    def volume(self):
        return np.array([mp.volume for mp in self])

    def percolation(self, t):
        mpups = [self[0].layer.cell.surfacewater] + self[:-1]
        return np.array([up.flux_to(down, t) for up, down in zip(mpups, self)])

    def aggregate_infiltration(self, t):
        return np.array([mp.flux_to(mp.layer, t) for mp in self])


# ------------------------------------------- FLOW OPTION 3: MACROPORES -------------------------------------------
class MacroporeFastFlow:
    def __init__(self, model, porefraction_mp, ksat_mp, density_mp, k_shape):

        """
        Creation of macropores, which are connected to every layer
        (mp_exchange between layer and corresponding macropore).

        :param model: the CmfModel object.
        :param porefraction_mp: the fraction of the macropores in m3/m3. This adds to the porosity of the layer.
        :param ksat_mp: the saturated conductivity of the macropores in m/day.
        :param density_mp: The mean distance between the macropores in m.
        :param infiltration: type of macropore infiltration, e.g. cmf.JarvisMacroFlow or cmf.KinematicMacroFlow
        """
        self.macropores = MacroPoreList(
            cmf.MacroPore.create(layer, porefraction=porefraction_mp, Ksat=ksat_mp, density=density_mp, K_shape=k_shape)
            for layer in model.c.layers)

        # Infiltration: either cmf.KinematicMacroFlow of cmf.JarvisMacroFlow
        self.mp_infiltration = cmf.KinematicMacroFlow(model.c.surfacewater, self.macropores[0])

        # Percolation: either cmf.KinematicMacroFlow, cmf.JarvisMacroFlow, or cmf.GradientMacroFlow)
        self.mp_percolation = []
        for left, right in zip(self.macropores[:-1], self.macropores[1:]):
            self.mp_percolation.append(cmf.KinematicMacroFlow(left, right))

        # Macropore-Matrix Exchange: cmf.DiffusiveMacroMicroExchange(left, right, omega=omega),
        # cmf.GradientMacroMicroExchange, or cmf.MACROlikeMacroMicroExchange
        self.mp_mx_exchange = []
        for left, right in zip(self.macropores, model.c.layers):
            self.mp_mx_exchange.append(cmf.GradientMacroMicroExchange(right, left))

        # Percolation to groundwater:
        cmf.KinematicMacroFlow(self.macropores[-1], model.gw)
        self.bypass = None


# ------------------------------------------- CLASS FOR THE MODEL -------------------------------------------
class CmfModel(cmf.project):
    """
    A model for studying the influence of different fast flow components (none, bypass, direct routing, and macropores)
    for water fluxes and phosphorus components (colloidal P and dissolved P)
    """
    # TODO: Check adsorption: use LinearAdsorption? Or should I create a equilibrium between DIP, DOP and PP?

    def __init__(self, water_params=None,
                 phosphorus_params=None,
                 spotpy_soil_params=False,
                 irrigation=1,
                 profile=1,
                 fast_component=3,
                 tracer='',
                 begin=cmf.Time(1, 1, 2019, 0, 00),
                 cell=(0, 0, 0, 1000, True),  # IMPORTANT: now the area is 1000 m2
                 surface_runoff=True, **kwargs):
        """
        Creates the basic structure of the model.

        :param water_params: spotpy (or determined) parameter for water
        :param phosphorus_params: spotpy (or determined) parameters for phosphorus
        :param irrigation: for which irrigation the model is created (important only for setting phosphorus conc)
        :param profile: the profile for which the model is created (important for soil layers)
        :param fast_component: either None, MacroporeFastFlow, or BypassFastFlow
        :param tracer: one or more possible tracer that is/are transported via water, adsorbed and filtered
        :param begin: cmf.Time object, starting point of solute_results
        :param cell: tuple describing the simulated cell (x, y, z, area, with surfacewater=true)
        :param kwargs: further parameters possible
        """
        self.begin = begin
        self.dt = cmf.min

        rain = iao.irrigation_experiment(self.begin, self.dt, fade_out=2 * 60)
        self.duration = len(rain)
        self.tend = self.begin + self.dt * self.duration

        super().__init__(tracer)

        self.c = self.NewCell(*cell)

        soil = iao.real_soil_from_csv(soil_file=Path('input/MIT' + str(profile) + '_soil.csv'))
        self.layer_boundary = self.create_layers(soil, water_params, based_on_spotpy=spotpy_soil_params)
        self.mx_infiltration, self.mx_percolation = self.connect_matrix()

        self.gw = self.create_groundwater()
        if fast_component == 1:
            self.flow_approach = None
        elif fast_component == 2:
            self.flow_approach = BypassFastFlow(self, water_params.ksat_mp)
        elif fast_component == 3:
            self.flow_approach = MacroporeFastFlow(self, porefraction_mp=water_params.porefraction_mp,
                                                   ksat_mp=water_params.ksat_mp, density_mp=water_params.density_mp,
                                                   k_shape=water_params.k_shape)

        self.c.surfacewater.puddledepth = water_params.puddle_depth
        self.c.saturated_depth = water_params.saturated_depth

        if surface_runoff:
            self.surface_runoff = self.create_surface_runoff()
        else:
            self.surface_runoff = False

        self.rain_station = self.create_rainfall_station(rain)

        if phosphorus_params:
            self.dip, self.dop, self.pp = self.solutes
            self.matrix_filter(phosphorus_params)
            self.rainstation_concentration(irrigation, profile)
            self.layer_concentration(phosphorus_params)
            if type(self.flow_approach) == BypassFastFlow:
                self.bypass_filter(phosphorus_params)
            elif type(self.flow_approach) == MacroporeFastFlow:
                self.macropore_filter(phosphorus_params)

    # -------------------------------------- CREATING AND CONNECTION SOIL LAYERS --------------------------------------
    def create_layers(self, soil_horizons, params, based_on_spotpy=False):
        """
        Creates a list of layers with attributes taken from a csv file (created via Brook90).

        :param params: class containing all parameters
        :param based_on_spotpy:
        :param soil_horizons: data frame including Van-Genuchten-Mualem soil parameters
        :return: A list of the lower boundaries of all created layer
        """
        layer_boundary = []

        for i, row in soil_horizons.iterrows():
            depth = -soil_horizons['Depth(m)'][i]  # for positive values
            if depth > 0:
                layer_boundary.append(depth)

                if row['HorId'] == 'Ah':
                    phi = params.porosity_mx_ah
                    w0 = params.w0_ah
                elif row['HorId'] == 'Bv1':
                    phi = params.porosity_mx_bv1
                    w0 = params.w0_bv1
                elif row['HorId'] == 'Bv2':
                    phi = params.porosity_mx_bv2
                    w0 = params.w0_bv2
                else:
                    phi = (params.porosity_mx_ah + params.porosity_mx_bv1 + params.porosity_mx_bv2) / 3
                    w0 = (params.w0_ah + params.w0_bv1 + params.w0_bv2) / 3

                if based_on_spotpy:
                    if row['HorId'] == 'Ah':
                        k = params.ksat_mx_ah
                        alpha = params.alpha_ah
                        n = params.n_ah
                        m = params.m_ah
                    elif row['HorId'] == 'Bv1':
                        k = params.ksat_mx_bv1
                        alpha = params.alpha_bv1
                        n = params.n_bv1
                        m = params.m_bv1
                    elif row['HorId'] == 'Bv2':
                        k = params.ksat_mx_bv2
                        alpha = params.alpha_bv2
                        n = params.n_bv2
                        m = params.m_bv2
                    else:
                        k = (params.ksat_mx_ah + params.ksat_mx_bv1 + params.ksat_mx_bv2) / 3
                        alpha = (params.alpha_ah + params.alpha_bv1 + params.alpha_bv2) / 3
                        n = (params.n_ah + params.n_bv1 + params.n_bv2) / 3
                        m = (params.m_ah + params.m_bv1 + params.m_bv2) / 3
                else:
                    k = row['KS/KF[m/d]']
                    alpha = row['Alfa/PsiF[m-1]/[kPa]'] / 100
                    n = row['n/b']
                    m = 1 - 1 / n

                vgm = cmf.VanGenuchtenMualem(Ksat=k, phi=phi, alpha=alpha, n=n, m=m, w0=w0)
                self.c.add_layer(depth, vgm)

        return layer_boundary

    def connect_matrix(self):
        """
        Creates matrix percolation (MatrixInfiltration is created automatically)
        """
        self.c.surfacewater.remove_connection(self.c.layers[0])
        mx_infiltration = cmf.MatrixInfiltration(self.c.layers[0], self.c.surfacewater)

        mx_percolation = []
        for left, right in zip(self.c.layers[:-1], self.c.layers[1:]):
            mx_percolation.append(cmf.Richards(left, right))
        return mx_infiltration, mx_percolation

    # -------------------------------------- INFLOW AND OUTFLOW OF WATER --------------------------------------

    def create_groundwater(self):
        """
        Creates the groundwater as a DirichletBoundary and connects the deepest SoilLayer with it.

        :return: outlet in form of a groundwater body
        """
        gw = self.NewOutlet('groundwater', self.c.x, self.c.y, self.c.z - self.c.layers[-1].lower_boundary)

        cmf.FreeDrainagePercolation(self.c.layers[-1], gw)
        return gw

    def create_surface_runoff(self):
        """
        If a topographic slope exists surface water can runoff to another outlet.
        https://philippkraft.github.io/cmf/cmf_tut_kinematic_wave.html
        """
        surface_outlet = self.NewOutlet('surface', self.c.x - 0.5, self.c.y - 0.5, self.c.z - 0.5)

        cmf.KinematicSurfaceRunoff(self.c.surfacewater, surface_outlet, flowwidth=sqrt(self.c.area))

        # I deleted the PowerLawEquation, since this needed so many unknown parameters...

        return surface_outlet

    def create_rainfall_station(self, rain):
        """
        Creates a cmf.RainfallStation from a list of rainfall data and sets concentration of Pd and Pc
        according to mean concentration in surfacewater at MIT (this might need editing!)
        :param rain: cmf.timeseries, containing rain data
        :return: a cmf.RainfallStation
        """
        rain_station = self.rainfall_stations.add(Name='pot', Data=rain, Position=(0, 0, 0))
        rain_station.use_for_cell(self.c)
        return rain_station

    # -------------------------------------- CONCENTRATION OF PHOSPHORUS AT T0 --------------------------------------
    def rainstation_concentration(self, irrigation, profile):
        surface = iao.surface_df(source=Path('input/MIT_Surface.csv'), irrigation=irrigation, profile=profile)

        self.rain_station.concentration[self.dip] = cmf.timeseries.from_scalar(
            amount_per_l_to_amount_per_m3(surface[surface['depth [m]'] == 'BLANK']['DIP [mcg/l]'].mean()))
        self.rain_station.concentration[self.dop] = cmf.timeseries.from_scalar(
            amount_per_l_to_amount_per_m3(surface[surface['depth [m]'] == 'BLANK']['DOP [mcg/l]'].mean()))
        self.rain_station.concentration[self.pp] = cmf.timeseries.from_scalar(
            amount_per_l_to_amount_per_m3(surface[surface['depth [m]'] == 'BLANK']['PP [mcg/l]'].mean()))

    def layer_concentration(self, phosphorus_params):
        for layer in self.c.layers:
            # layer.Solute(self.dip).set_adsorption(adsorption)
            layer.Solute(self.dip).state = phosphorus_params.dip_state
            layer.Solute(self.dop).state = phosphorus_params.dop_state
            layer.Solute(self.pp).state = phosphorus_params.pp_state
        if type(self.flow_approach) == MacroporeFastFlow:
            for mp in self.flow_approach.macropores:
                mp.Solute(self.dip).state = 0  # I think "0" for macropores is correct: MPs are empty at the beginning
                mp.Solute(self.dop).state = 0
                mp.Solute(self.pp).state = 0

    # -------------------------------------- FILTER FOR PHOSPHORUS --------------------------------------
    def matrix_filter(self, phosphorus_params):
        """
        1.0 is no filter and 0.0 means no solute is crossing this connection
        """
        self.mx_infiltration.set_tracer_filter(self.dip, phosphorus_params.mx_filter_dp)
        self.mx_infiltration.set_tracer_filter(self.dop, phosphorus_params.mx_filter_dp)
        self.mx_infiltration.set_tracer_filter(self.pp, phosphorus_params.mx_filter_pp)
        for layer in self.mx_percolation:
            layer.set_tracer_filter(self.dip, phosphorus_params.mx_filter_dp)
            layer.set_tracer_filter(self.dop, phosphorus_params.mx_filter_dp)
            layer.set_tracer_filter(self.pp, phosphorus_params.mx_filter_pp)

    def bypass_filter(self, phosphorus_params):
        """
        1.0 is no filter and 0.0 means no solute is crossing this connection
        """
        for bp in self.flow_approach.bypass:
            bp.set_tracer_filter(self.dip, phosphorus_params.mp_filter_dp)
            bp.set_tracer_filter(self.dop, phosphorus_params.mp_filter_dp)
            bp.set_tracer_filter(self.pp, phosphorus_params.mp_filter_pp)

    def macropore_filter(self, phosphorus_params):
        """
        1.0 is no filter and 0.0 means no solute is crossing this connection
        """
        self.flow_approach.mp_infiltration.set_tracer_filter(self.dip, phosphorus_params.mp_filter_dp)
        self.flow_approach.mp_infiltration.set_tracer_filter(self.dop, phosphorus_params.mp_filter_dp)
        self.flow_approach.mp_infiltration.set_tracer_filter(self.pp, phosphorus_params.mp_filter_pp)

        # Do I nead a filter for dissolved P in the macropores? I think not?
        for mp in self.flow_approach.mp_percolation:
            mp.set_tracer_filter(self.dip, phosphorus_params.mp_filter_dp)
            mp.set_tracer_filter(self.dop, phosphorus_params.mp_filter_dp)
            mp.set_tracer_filter(self.pp, phosphorus_params.mp_filter_pp)

        for mp in self.flow_approach.mp_mx_exchange:
            mp.set_tracer_filter(self.dip, phosphorus_params.exch_filter_dp)
            mp.set_tracer_filter(self.dop, phosphorus_params.exch_filter_dp)
            mp.set_tracer_filter(self.pp, phosphorus_params.exch_filter_pp)


# -----------------------------------------------------------------------------------------------------
# ------------------------------------------- CHANGING UNITS ------------------------------------------


def amount_per_l_to_amount_per_m3(mcg_per_l):
    """
    CMF always calculates concentrations in AMOUNT per m3. Since input units are in L, this function re-calculates it
    to AMOUNT/m3 [AMOUNT is usually mcg].

    :param mcg_per_l: concentration in mcg per l
    :return: concentration of solute in mcg per m3
    """
    return mcg_per_l * 1e3

