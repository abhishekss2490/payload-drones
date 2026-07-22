"""Vendored from mj-adapt-drones at commit 1d52e89 on 2026-07-23.

This is a vendored copy for payload-drones independence; do not edit casually.
Only the reachable articulation configuration is retained.  The corresponding
quad XML is payload-drones/assets/quad_with_hook.xml, which has the same four
motor sites and actuator names as the source asset plus the local hook site.
"""

from mjlab.actuator import XmlActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import EntityArticulationInfoCfg

QUADROTOR_ACTUATOR_CFGS = XmlActuatorCfg(
    target_names_expr=("motor_.*",),
    transmission_type=TransmissionType.SITE,
    delay_min_lag=0,
    delay_max_lag=1,
    delay_hold_prob=0.1,
    delay_update_period=2,
)

QUADROTOR_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(QUADROTOR_ACTUATOR_CFGS,),
)
