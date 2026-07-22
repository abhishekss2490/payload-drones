"""Throughput of the full training loop: env step + policy forward + backward.

`bench_env.py` measures env stepping only. That is an upper bound on training
speed, because it excludes the actor/critic forward during rollout collection and
the PPO update. This measures the whole loop so training-time estimates are
honest.

Run: ``uv run --no-sync python scripts/bench_train_step.py``
"""

from __future__ import annotations

import time

import torch

from mjlab.envs import ManagerBasedRlEnv
from payload_drones.payload_env_cfg import make_payload_env_cfg
from payload_drones.rl.models import MARLTeacherModel

WARMUP = 20
MEASURE = 100


def build(num_envs: int):
    cfg = make_payload_env_cfg(num_envs=num_envs)
    env = ManagerBasedRlEnv(cfg, device="cuda")
    obs, _ = env.reset()
    obs_groups = {
        "policy": ["policy", "payload_trajectory", "agent_dynamics"],
        "critic": ["critic"],
    }
    actor = MARLTeacherModel(
        obs, obs_groups, "policy",
        output_dim=env.action_manager.total_action_dim,
        distribution_cfg={"class_name": "GaussianDistribution"},
    ).to("cuda")
    return env, actor, obs


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device; this benchmark is meaningless on CPU")

    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"{'num_envs':>9} | {'env only':>11} | {'env+policy':>11} | "
          f"{'+backward':>11} | {'GPU (GiB)':>10}")
    print("-" * 66)

    for n in (4096, 8192):
        env, actor, obs = build(n)
        actions = torch.zeros(n, env.action_manager.total_action_dim, device="cuda")

        # --- env only ---
        for _ in range(WARMUP):
            env.step(actions.uniform_(-1, 1))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(MEASURE):
            env.step(actions.uniform_(-1, 1))
        torch.cuda.synchronize()
        env_only = MEASURE * n / (time.perf_counter() - t0)

        # --- env + policy forward (rollout collection) ---
        # Each phase needs its own warmup: kernel compilation and autotuning
        # happen on first use of a new code path, and without this the later
        # phases absorb setup cost from the earlier ones (or benefit from
        # already-warm kernels), producing impossible orderings.
        for _ in range(WARMUP):
            with torch.no_grad():
                a = actor(obs, stochastic_output=True)
            obs, *_ = env.step(a)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(MEASURE):
            with torch.no_grad():
                a = actor(obs, stochastic_output=True)
            obs, *_ = env.step(a)
        torch.cuda.synchronize()
        with_policy = MEASURE * n / (time.perf_counter() - t0)

        # --- + backward (approximates the PPO update cost per step) ---
        # `stochastic_output=True` returns distribution.sample(), which is not
        # differentiable, so differentiate the deterministic output instead. It
        # exercises the same forward and backward graph, which is what we are
        # timing — this is a cost proxy, not a real PPO objective.
        opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
        for _ in range(WARMUP):
            a = actor(obs, stochastic_output=False)
            loss = a.square().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                obs, *_ = env.step(a.detach())
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(MEASURE):
            a = actor(obs, stochastic_output=False)
            loss = a.square().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                obs, *_ = env.step(a.detach())
        torch.cuda.synchronize()
        with_backward = MEASURE * n / (time.perf_counter() - t0)

        _, total = torch.cuda.mem_get_info()
        free_after, _ = torch.cuda.mem_get_info()
        used = (total - free_after) / 2**30

        print(f"{n:>9} | {env_only:>11,.0f} | {with_policy:>11,.0f} | "
              f"{with_backward:>11,.0f} | {used:>10.2f}")

        env.close()
        del env, actor, opt
        torch.cuda.empty_cache()

    print()
    print("Training-time estimate uses the '+backward' column, which is the")
    print("closest single-number proxy for sustained PPO throughput.")


if __name__ == "__main__":
    main()
