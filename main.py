# -*- coding: utf-8 -*-
"""
Created in Aug 2020

The aim of this script is to run the model in different ways:
a) single run with water (DONE)
b) single run with phosphorus (DONE)
c) spotpy with water (DONE)
d) spotpy with phosphorus (DONE)
e) validation of water and phosphorus routines (TODO)

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

import cmf
from build_the_model import CmfModel
import run_and_plot as rap


# ------------------------------------------- SETUP OF MODEL AND SPOTPY -------------------------------------------
class ModelInterface:
    """
    Class to create a CmfModel and run it via Spotpy for calibration
    """

    def __init__(self, flow_approach=3):
        self.flow_approach = flow_approach

        self.water_params = WaterParameters(system=self.flow_approach)
        self.phosphorus_params = PhosphorusParameters(system=self.flow_approach)

        self.begin = cmf.Time(12, 6, 2018, 0, 00)  # starting time of solute_results
        self.dt = cmf.min  # time steps (cmf.sec, cmf.min, cmf.h, cmf.day, cmf.week, cmf.month, cmf.year)

        self.tracer = 'dp pp'
        # self.tracer = 'dip dop pp'

        self.project = CmfModel(water_params=self.water_params,
                                phosphorus_params=self.phosphorus_params,
                                fast_component=self.flow_approach,
                                tracer=self.tracer,
                                begin=self.begin,
                                cell=(0, 0, 0, 1000, True),  # IMPORTANT: now the area is 1000 m2
                                surface_runoff=True)


# --------------------------------- CLASSES FOR WATER AND PHOSPHORUS PARAMETER SETS ---------------------------------
class WaterParameters:
    def __init__(self, system=1):
        """
        Here, spotpy parameters are extracted from a row of spotpy results. These parameters then are used to create a
        new model
        :param system: 1 for matrix flow only, 2 for bypass flow, and 3 for macropores
        """

        self.saturated_depth = 6.137  # 1.7  # 4.76
        self.puddle_depth = 0.003645  # 0.002348  # 0.004276
        self.porosity_mx_ah = 0.4119  # 0.4639  # 0.8057
        self.porosity_mx_bv1 = 0.006695  # 0.0354  # 0.0909
        self.porosity_mx_bv2 = 0.01784  # 0.34299999999999997  # 0.7163
        self.w0_ah = 0.866  # 0.9863  # 0.9517
        self.w0_bv1 = 0.8438  # 0.9565  # 0.843
        self.w0_bv2 = 0.9414  # 0.9946  # 0.855

        self.ksat_mx_ah = 0.39  # 0.268  # 14.445  # 14.63
        self.ksat_mx_bv1 = 3.432  # 3.432  #2.6210000000000004  # 3.541
        self.ksat_mx_bv2 = 14.37  # 1.777  # 0.7764
        self.n_ah = 3.312  # 1.0590000000000002  # 1.211
        self.n_bv1 = 3.18  # 1.088  # 1.211
        self.n_bv2 = 3.168  # 2.102  # 1.211
        self.alpha_ah = 0.03058  # 0.0912  # 0.2178
        self.alpha_bv1 = 0.3984  # 0.97  # 0.2178
        self.alpha_bv2 = 0.707  # 0.6416  # 0.2178
        self.m_ah = 0.2502  # -1  # negative: 1 - 1/n
        self.m_bv1 = 0.9893  # -1  # negative: 1 - 1/n
        self.m_bv2 = 0.557  # -1  # negative: 1 - 1/n

        if system == 2:
            self.ksat_mp = 10
        elif system == 3:
            self.ksat_mp = 240  # 144  #62.7772
            self.porefraction_mp = 0.91  # 0.284378
            self.density_mp = 0.0381  # 0.0381  #0.96332
            self.k_shape = 0.73  # 0.946  #0.01


class PhosphorusParameters:
    def __init__(self, system=1):
        """
        Here, spotpy parameters are extracted from a row of spotpy results. These parameters then are used to create a
        new model
        :param system: 1 for matrix flow only, 2 for bypass flow, and 3 for macropores
        """
        self.pp_state_ah = 646.5  # 10
        self.pp_state_bv1 = 156.5  # 10
        self.pp_state_bv2 = 170.5  # 10

        self.dp_state_ah = 885.5  # 10
        self.dp_state_bv1 = 334  # 10
        self.dp_state_bv2 = 334.8  # 10

        self.mx_filter_dp = 0.9165  # 0.0815611  # 1
        self.mx_filter_pp = 0.2285  # 0.0984676  # 0.1

        if system == 2 or system == 3:
            self.mp_filter_dp = 0.2141
            self.mp_filter_pp = 0.000997
        if system == 3:
            self.exch_filter_dp = 0.873
            self.exch_filter_pp = 0.886


if __name__ == '__main__':
    fastflow = 3  # 1: matrix, 2: bypass, 3: macropores

    setup = ModelInterface(flow_approach=fastflow)

    water_results, phosphorus_results = rap.run(setup.project, print_time=True)

    # rap.plotting(model=setup)
