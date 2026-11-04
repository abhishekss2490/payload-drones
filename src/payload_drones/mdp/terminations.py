"""Failure conditions for the cable-coupled team."""

from __future__ import annotations

import torch
import warp as wp

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.terminations import nan_detection, time_out
from mjlab.managers.termination_manager import TerminationTermCfg

from payload_drones.mdp.observations import _payload, _ref_pos
from payload_drones.mdp.team_scaling import N_AGENTS


def _active_contacts(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Return geom pairs and world ids for actual penetrating/touching contacts.

    MuJoCo-Warp stores contacts for all worlds in one compact prefix of a
    capacity-sized buffer.  Reading only that prefix is essential: stale values
    outside ``nacon`` survive resets and look like real collisions otherwise.
    """
    data = env.sim.wp_data
    geom = wp.to_torch(data.contact.geom)
    world = wp.to_torch(data.contact.worldid).long()
    dist = wp.to_torch(data.contact.dist)
    nacon = wp.to_torch(data.nacon)[0]
    row = torch.arange(geom.shape[0], device=env.device)
    valid = (row < nacon) & (dist <= 0.0)
    return geom[valid], world[valid]


def _contacts_between(
    env: ManagerBasedRlEnv,
    first_geom_ids: torch.Tensor,
    second_geom_ids: torch.Tensor,
) -> torch.Tensor:
    geom, world = _active_contacts(env)
    out = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if geom.numel() == 0:
        return out
    first = first_geom_ids.to(device=env.device, dtype=geom.dtype)
    second = second_geom_ids.to(device=env.device, dtype=geom.dtype)
    hit = (
        torch.isin(geom[:, 0], first) & torch.isin(geom[:, 1], second)
    ) | (
        torch.isin(geom[:, 1], first) & torch.isin(geom[:, 0], second)
    )
    out[world[hit]] = True
    return out


def _terrain_geom_ids(env: ManagerBasedRlEnv) -> torch.Tensor:
    if env.scene.terrain is None:
        raise RuntimeError("contact-based ground termination requires a terrain entity")
    return env.scene.terrain.indexing.geom_ids


def payload_ground_contact(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Terminate on actual payload-geometry contact with the terrain."""
    return _contacts_between(
        env, env.scene["payload"].indexing.geom_ids, _terrain_geom_ids(env)
    )


def any_quad_ground_contact(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Terminate when any quad geometry actually contacts the terrain."""
    terrain = _terrain_geom_ids(env)
    result = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for i in range(N_AGENTS):
        result |= _contacts_between(env, env.scene[f"quad_{i}"].indexing.geom_ids, terrain)
    return result


def payload_far_from_goal(env: ManagerBasedRlEnv, command_name: str = "payload_pose", max_dist: float = 3.0) -> torch.Tensor:
    return (_payload(env).data.root_link_pos_w - _ref_pos(env, command_name)).norm(dim=-1) > max_dist


def any_quad_tumbling(env: ManagerBasedRlEnv, max_ang_vel: float = 50.0) -> torch.Tensor:
    rates = torch.stack([env.scene[f"quad_{i}"].data.root_link_ang_vel_b.norm(dim=-1) for i in range(N_AGENTS)], -1)
    active = env.team_state.active_mask if hasattr(env, "team_state") else torch.ones_like(rates, dtype=torch.bool)
    return ((rates > max_ang_vel) & active).any(-1)


def quad_collision(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Terminate on the same inter-quad contacts resolved by the simulator.

    This intentionally replaces the arm-length centre-distance proxy.  The
    current collision geoms are nominal-size and rotation-dependent, so only
    the solver's contact list exactly matches the collision forces applied to
    the simulated vehicles.
    """
    result = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for i in range(N_AGENTS):
        for j in range(i + 1, N_AGENTS):
            result |= _contacts_between(
                env,
                env.scene[f"quad_{i}"].indexing.geom_ids,
                env.scene[f"quad_{j}"].indexing.geom_ids,
            )
    return result


def cable_overextension(env: ManagerBasedRlEnv, factor: float = 1.5) -> torch.Tensor:
    if not hasattr(env, "team_state"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    p = _payload(env)
    lengths = []
    for i in range(N_AGENTS):
        q = env.scene[f"quad_{i}"]
        hook = q.data.site_pos_w[:, q.find_sites(("hook",))[0][0]]
        attach = p.data.site_pos_w[:, p.find_sites((f"attach_{i}",))[0][0]]
        lengths.append((attach - hook).norm(dim=-1))
    return (torch.stack(lengths, -1) > factor * env.team_state.cable_length).any(-1)


termination_terms = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "payload_ground_contact": TerminationTermCfg(func=payload_ground_contact),
    "any_quad_ground_contact": TerminationTermCfg(func=any_quad_ground_contact),
    "payload_far_from_goal": TerminationTermCfg(func=payload_far_from_goal),
    "any_quad_tumbling": TerminationTermCfg(func=any_quad_tumbling),
    "quad_collision": TerminationTermCfg(func=quad_collision),
    "cable_overextension": TerminationTermCfg(func=cable_overextension),
    "nan_detection": TerminationTermCfg(func=nan_detection),
}
