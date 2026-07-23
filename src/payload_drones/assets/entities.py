"""Entity factories for the local payload and hook-equipped quadrotor assets."""

from __future__ import annotations

from pathlib import Path

import mujoco
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from payload_drones.assets.quadrotor_constants import QUADROTOR_ARTICULATION

_ASSET_DIR = Path(__file__).parent


def _spec_from(path: Path) -> mujoco.MjSpec:
    """Load an XML asset as an editable MuJoCo specification."""
    return mujoco.MjSpec.from_file(str(path))


def get_hook_quadrotor_cfg(
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> EntityCfg:
    """Return the base quadrotor dynamics with the local cable hook site."""
    def get_spec() -> mujoco.MjSpec:
        spec = _spec_from(_ASSET_DIR / "quad_with_hook.xml")
        spec.body("quad").pos = position
        return spec

    return EntityCfg(
        spec_fn=get_spec,
        articulation=QUADROTOR_ARTICULATION,
    )


def get_payload_cfg() -> EntityCfg:
    """Return the free, rigid triangular-attachment payload."""
    return EntityCfg(spec_fn=lambda: _spec_from(_ASSET_DIR / "payload.xml"))


if __name__ == "__main__":
    print(get_hook_quadrotor_cfg().build().spec)
    print(get_payload_cfg().build().spec)
