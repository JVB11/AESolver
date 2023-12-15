'''Contains typing classes used in the plotter and analyzer run modules.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import statements
from typing import TypedDict


class SaveInfo(TypedDict):
    save_base_dir: str
    save_dir: str
    extra_specifier_save: str
    save_formats: list[str]
    sharpness_save_name: str
    sharpness_subdir: str


class ModeInfo(TypedDict):
    rot_pct: float
    base_dir: str
    alternative_directory_path: str
    alternative_file_name: str
    verbose: bool
    debug: bool


class PlotInfo(TypedDict):
    plot_overview: bool
    verbose: bool
    plot_coefficient_profile: bool
    plot_resonance_sharpness: bool
    fig_size: tuple[float, float] | list[float]
    fig_dpi: int
    ax_label_size: int
    tick_label_size: int
    tick_size: int
    rad_nr: int
    show_plots: bool
    cmap: str
    tick_label_pad: int


class LoadInfo(TypedDict):
    base_file_names: list[str]
    specification_names: list[str]
    masses: list[float]
    Xcs: list[float]
    triple_pairs: list[list[int]]
    meridional_degrees: list[int]
    azimuthal_orders: list[int]
    file_suffix: str
    nr_files: int
    output_dir: str
    base_dir: str
    use_standard_format: bool


class AnalyzeInfo(TypedDict):
    numbers_categories: bool
    q_Q1_condition: bool
    isolated_mode_properties: bool
    isolated_stab_val_estimators: bool
    isolated_triad_observables: bool
