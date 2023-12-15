'''Initialization file for the module containing necessary data used for the runner modules.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import modules
from .utility_functions import check_save_file_name
# import specific typed dictionaries
from .solver_type_data import ModeInfoOrPolyDict, ModeSelectionDict, ModeComputationInfoDict, ComputationDict, SavingDict, PolyModelDict, AngularIntegrationDict, CouplingCoefficientProfileSelectionDict, CouplingCoefficientProfilePlotDict
from .plotter_analyzer_type_data import SaveInfo, ModeInfo, PlotInfo, LoadInfo, AnalyzeInfo


# define version, author and auto-import packages
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'
__all__ = ['check_save_file_name', 'ModeInfoOrPolyDict', 'ModeSelectionDict', 'ModeComputationInfoDict', 'ComputationDict', 'SavingDict', 'PolyModelDict', 'AngularIntegrationDict', 'CouplingCoefficientProfileSelectionDict', 'CouplingCoefficientProfilePlotDict', 'SaveInfo', 'ModeInfo', 'PlotInfo', 'LoadInfo', 'AnalyzeInfo']
