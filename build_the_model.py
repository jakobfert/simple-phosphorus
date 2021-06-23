# -*- coding: utf-8 -*-
"""
Created in August 2020

This script will be used to build the model. It will provide functions for the different setup options and so on.
In main.py, the model will be created by choosing from the options given here.

@author: pferdmenges-j
"""

import cmf
import numpy as np


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

        rain_times = (60 * [0.]  # 60 minutes dry
                      + 40 * ([2. * 10 * 24] + 5 * [0.])  # 40 times 1 minute of rain and 5 without = 4h
                      + 19 * 60 * [0.])  # 19 hours without rain = a total of 24 h of simulation time
        rain = cmf.timeseries.from_array(self.begin, self.dt, data=rain_times)
        self.duration = len(rain)
        self.tend = self.begin + self.dt * self.duration

        super().__init__(tracer)

        self.c = self.NewCell(*cell)

        horizon_depths = [0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28]  # , 2.56, 5.12, 10.24]
        self.create_layers(horizon_depths, water_params)
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

        self.rain_station = self.create_rainfall_station(rain)

        self.dp, self.pp = self.solutes
        # for s in self.solutes:
        #     for layer in self.c.layers:
        #         layer.Solute(s).set_abs_errtol(10)

        self.matrix_filter(phosphorus_params)
        self.rainstation_concentration()
        self.layer_concentration(horizon_depths, phosphorus_params)
        # self.layer_decay(phosphorus_params)

        if type(self.flow_approach) == BypassFastFlow:
            self.bypass_filter(phosphorus_params)
        elif type(self.flow_approach) == MacroporeFastFlow:
            self.macropore_filter(phosphorus_params)

    # -------------------------------------- CREATING AND CONNECTION SOIL LAYERS --------------------------------------
    def create_layers(self, soil_horizons, params):
        """
        Creates a list of layers with attributes taken from a csv file (created via Brook90).

        :param params: class containing all parameters
        :param soil_horizons: data frame including Van-Genuchten-Mualem soil parameters
        :return: A list of the lower boundaries of all created layer
        """

        for depth in soil_horizons:
            if depth <= 0.1:
                phi = params.porosity_mx_ah
                w0 = params.w0_ah
                k = params.ksat_mx_ah
                alpha = params.alpha_ah
                n = params.n_ah
                m = params.m_ah
            elif depth <= 0.5:
                phi = params.porosity_mx_bv1
                w0 = params.w0_bv1
                k = params.ksat_mx_bv1
                alpha = params.alpha_bv1
                n = params.n_bv1
                m = params.m_bv1
            else:
                phi = params.porosity_mx_bv2
                w0 = params.w0_bv2
                k = params.ksat_mx_bv2
                alpha = params.alpha_bv2
                n = params.n_bv2
                m = params.m_bv2

            vgm = cmf.VanGenuchtenMualem(Ksat=k, phi=phi, alpha=alpha, n=n, m=m, w0=w0)
            self.c.add_layer(depth, vgm)

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
    def rainstation_concentration(self):
        self.rain_station.concentration[self.dp] = cmf.timeseries.from_scalar(1000)

        self.rain_station.concentration[self.pp] = cmf.timeseries.from_scalar(1000)

    def layer_concentration(self, horizon_depths, phosphorus_params):
        """
        state of solute in a WaterStorage is always the total state, thus the derivated concentration is depending on
        the cell area (or, more precisely, the total water amount in the WaterStorage)
        """
        i = 0
        for layer in self.c.layers:
            if horizon_depths[i] <= 0.1:
                layer.Solute(self.dp).state = phosphorus_params.dp_state_ah * self.c.area
                layer.Solute(self.pp).state = phosphorus_params.pp_state_ah * self.c.area
            elif horizon_depths[i] <= 0.5:
                layer.Solute(self.dp).state = phosphorus_params.dp_state_bv1 * self.c.area
                layer.Solute(self.pp).state = phosphorus_params.pp_state_bv1 * self.c.area
            else:
                layer.Solute(self.dp).state = phosphorus_params.dp_state_bv2 * self.c.area
                layer.Solute(self.pp).state = phosphorus_params.pp_state_bv2 * self.c.area
            i += 1

        if type(self.flow_approach) == MacroporeFastFlow:
            for mp in self.flow_approach.macropores:
                mp.Solute(self.dp).state = 0
                mp.Solute(self.pp).state = 0

    # def layer_decay(self, phosphorus_params):
    #     """
    #     Folgende Übergänge existieren:
    #     DIP <-> DOP [param: dip_to_dop]
    #     DIP <-> PP [param: dip_to_pp]
    #     DOP <-> PP [param: dop_to_pp]
    #
    #     You give decay rates (positive). Thus, a positive value means DECREASE, a negative value means INCREASE.
    #     Thus: for dop_decay it is minus because direction of dip_to_dop is TOWARDS DOP.
    #     for pp_decay, both parameters are negative, because both directions are TOWARDS PP.
    #
    #     Since water does not stay for long in macropores or bypass, conversions between P forms are only present in the
    #     soil layers
    #     """
    #     # dip_decay = phosphorus_params.dip_to_dop + phosphorus_params.dip_to_pp
    #     # dop_decay = phosphorus_params.dop_to_pp - phosphorus_params.dip_to_dop
    #     # pp_decay = -phosphorus_params.dip_to_pp - phosphorus_params.dop_to_pp
    #
    #     for layer in self.c.layers:
    #         layer.Solute(self.dp).decay = phosphorus_params.dp_to_pp
    #         layer.Solute(self.pp).decay = -phosphorus_params.dp_to_pp
    #
    #         # layer.Solute(self.dip).decay = dip_decay
    #         # layer.Solute(self.dop).decay = dop_decay
    #         # layer.Solute(self.pp).decay = pp_decay

    # -------------------------------------- FILTER FOR PHOSPHORUS --------------------------------------
    def matrix_filter(self, phosphorus_params):
        """
        1.0 is no filter and 0.0 means no solute is crossing this connection
        """
        self.mx_infiltration.set_tracer_filter(self.dp, phosphorus_params.mx_filter_dp)
        self.mx_infiltration.set_tracer_filter(self.pp, phosphorus_params.mx_filter_pp)
        for layer in self.mx_percolation:
            layer.set_tracer_filter(self.dp, phosphorus_params.mx_filter_dp)
            layer.set_tracer_filter(self.pp, phosphorus_params.mx_filter_pp)

    def bypass_filter(self, phosphorus_params):
        """
        1.0 is no filter and 0.0 means no solute is crossing this connection
        """
        for bp in self.flow_approach.bypass:
            bp.set_tracer_filter(self.dp, phosphorus_params.mp_filter_dp)
            bp.set_tracer_filter(self.pp, phosphorus_params.mp_filter_pp)

    def macropore_filter(self, phosphorus_params):
        """
        1.0 is no filter and 0.0 means no solute is crossing this connection
        """
        self.flow_approach.mp_infiltration.set_tracer_filter(self.dp, phosphorus_params.mp_filter_dp)
        self.flow_approach.mp_infiltration.set_tracer_filter(self.pp, phosphorus_params.mp_filter_pp)

        # Do I nead a filter for dissolved P in the macropores? I think not?
        for mp in self.flow_approach.mp_percolation:
            mp.set_tracer_filter(self.dp, phosphorus_params.mp_filter_dp)
            mp.set_tracer_filter(self.pp, phosphorus_params.mp_filter_pp)

        for mp in self.flow_approach.mp_mx_exchange:
            mp.set_tracer_filter(self.dp, phosphorus_params.exch_filter_dp)
            mp.set_tracer_filter(self.pp, phosphorus_params.exch_filter_pp)
