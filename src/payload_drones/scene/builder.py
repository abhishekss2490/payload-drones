"""Assembly of the three-quadrotor spatial-tendon payload scene."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import mujoco
import numpy as np
from mjlab.entity import EntityCfg
from mjlab.scene import SceneCfg
from mjlab.terrains import TerrainEntityCfg

from payload_drones.assets.entities import get_hook_quadrotor_cfg, get_payload_cfg

AIR_DENSITY: Final = 1.2
AIR_VISCOSITY: Final = 1.8e-5
N_CABLES: Final = 3
DEFAULT_CABLE_LENGTH: Final = 0.60
# A cable is an inextensible unilateral constraint, not a stiff Euler-integrated
# spring.  The limit solver uses this reference time to resolve its upper bound.
CABLE_LIMIT_SOLREF: Final = (0.01, 1.0)
# Must match payload.xml's attach-site triangle exactly: each hook sits directly
# above its own attach point, so the cables are vertical.
_QUAD_POSITIONS: Final = (
    (0.3, 0.0, 0.76),
    (-0.15, 0.259808, 0.76),
    (-0.15, -0.259808, 0.76),
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
        # A real cable is inextensible and unilateral.  The upper tendon limit
        # is therefore the load-bearing mechanism; it pulls only once the
        # geometric length reaches L and never pushes while slack.  Make the
        # flag explicit instead of relying on MuJoCo's auto-limit inference.
        tendon.limited = True
        tendon.springlength = [0.0, DEFAULT_CABLE_LENGTH]
        tendon.range = [0.0, DEFAULT_CABLE_LENGTH]
        # Do not add an explicit spring/damper: its force is integrated by
        # Euler and becomes either unstable (stiff) or a bungee (soft) for the
        # randomized 20--800 g quadrotors. Limit forces are solver constraints.
        tendon.stiffness = np.zeros(3)
        tendon.damping = np.zeros(3)
        tendon.solref_limit = CABLE_LIMIT_SOLREF


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
    return SceneCfg(
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities=entities,
        num_envs=num_envs,
        env_spacing=env_spacing,
        spec_fn=_set_air_drag_and_cables,
    )


if __name__ == "__main__":
    from mjlab.scene import Scene

    scene = Scene(make_payload_scene_cfg(), device="cpu")
    model = scene.compile()
    assert model.ntendon == N_CABLES
    print([scene.spec.site(f"quad_{i}/hook").name for i in range(N_CABLES)])
    print(f"Compiled {model.ntendon} spatial tendons: {list(model.tendon_range)}")
