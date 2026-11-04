"""Articulation settings for the quadrotor assets used in payload transport."""

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
