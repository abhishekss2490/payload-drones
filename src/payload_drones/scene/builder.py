"""Assembly of the three-quadrotor spatial-tendon payload scene."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import mujoco
import numpy as np
from mjlab.entity import EntityCfg
from mjlab.scene import SceneCfg

from payload_drones.assets.entities import get_hook_quadrotor_cfg, get_payload_cfg

AIR_DENSITY: Final = 1.2
AIR_VISCOSITY: Final = 1.8e-5
N_CABLES: Final = 3
DEFAULT_CABLE_LENGTH: Final = 0.60
DEFAULT_CABLE_STIFFNESS: Final = 2_000.0
DEFAULT_CABLE_DAMPING: Final = 80.0
_QUAD_POSITIONS: Final = (
    (0.15, 0.0, 0.76),
    (-0.075, 0.129904, 0.76),
    (-0.075, -0.129904, 0.76),
)


def _set_air_drag_and_cables(spec: mujoco.MjSpec) -> None:
    """Configure root-scene air drag and add cross-entity spatial tendons."""
    spec.option.density = AIR_DENSITY
    spec.option.viscosity = AIR_VISCOSITY
    for index in range(N_CABLES):
        hook = f"quad_{index}/hook"
        attach = f"payload/attach_{index}"
        if spec.site(hook) is None or spec.site(attach) is None:
            raise RuntimeError(f"Expected attached sites {hook!r} and {attach!r}.")
        tendon = spec.add_tendon(name=f"cable_{index}")
        tendon.wrap_site(hook)
        tendon.wrap_site(attach)
        tendon.limited = True
        tendon.range = [0.0, DEFAULT_CABLE_LENGTH]
        # The MjSpec Python binding represents both parameters as 3-vectors.
        tendon.stiffness = np.full(3, DEFAULT_CABLE_STIFFNESS)
        tendon.damping = np.full(3, DEFAULT_CABLE_DAMPING)


def make_payload_scene_cfg(
    quadrotors: Sequence[EntityCfg] | None = None,
    payload: EntityCfg | None = None,
    *,
    num_envs: int = 1,
    env_spacing: float = 4.0,
) -> SceneCfg:
    """Build a three-quadrotor free-payload scene with spatial cable tendons."""
    if quadrotors is None:
        quadrotors = tuple(get_hook_quadrotor_cfg(_QUAD_POSITIONS[i]) for i in range(N_CABLES))
    if len(quadrotors) != N_CABLES:
        raise ValueError(f"Expected exactly {N_CABLES} quadrotors, got {len(quadrotors)}.")
    if payload is None:
        payload = get_payload_cfg()
    entities = {f"quad_{index}": quad for index, quad in enumerate(quadrotors)}
    entities["payload"] = payload
    return SceneCfg(entities=entities, num_envs=num_envs, env_spacing=env_spacing, spec_fn=_set_air_drag_and_cables)


if __name__ == "__main__":
    from mjlab.scene import Scene

    scene = Scene(make_payload_scene_cfg(), device="cpu")
    model = scene.compile()
    assert model.ntendon == N_CABLES
    print([scene.spec.site(f"quad_{i}/hook").name for i in range(N_CABLES)])
    print(f"Compiled {model.ntendon} spatial tendons: {list(model.tendon_range)}")
