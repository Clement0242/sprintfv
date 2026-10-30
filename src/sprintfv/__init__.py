"""sprintfv — sprint acceleration and force-velocity profiling in Python.

Radar or laser guns, timing gates and GPS/LPS tracking data in; mono-exponential
sprint model, Samozino force-velocity-power profile, optimal profile and
in-situ acceleration-speed profile out.
"""

from .datasets import load_jb_morin, load_lps_session, load_radar_gun, load_split_times
from .fit import SprintFit, fit_radar, fit_splits
from .fvp import (ForceVelocityProfile, OptimalProfile, air_resistance_coefficient,
                  force_velocity_profile, optimal_profile)
from .insitu import METHODS as IN_SITU_METHODS
from .insitu import InSituProfile, fit_in_situ
from .kalman import Kinematics, KalmanParams, kalman_smooth, positions_to_kinematics
from .signal import acceleration_end, clean_radar

__version__ = "0.1.0"

__all__ = [
    "SprintFit", "fit_radar", "fit_splits",
    "ForceVelocityProfile", "OptimalProfile", "air_resistance_coefficient",
    "force_velocity_profile", "optimal_profile",
    "InSituProfile", "fit_in_situ", "IN_SITU_METHODS",
    "KalmanParams", "Kinematics", "kalman_smooth", "positions_to_kinematics",
    "clean_radar", "acceleration_end",
    "load_jb_morin", "load_lps_session", "load_radar_gun", "load_split_times",
]
