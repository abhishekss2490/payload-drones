"""End-to-end environment throughput on GPU.

`SCAFFOLD_FINDINGS.md` reports raw *physics* throughput. That number is optimistic
for training-time planning because it excludes observation assembly, reward and
termination evaluation, and the event/command managers. This measures the full
`env.step()` path, which is what actually bounds a PPO run.

Run: ``uv run --no-sync python scripts/bench_env.py``
"""

from __future__ import annotations

import time

import torch

from payload_drones.payload_env_cfg import make_payload_env_cfg

from mjlab.envs import ManagerBasedRlEnv

WARMUP = 50
MEASURE = 200


def bench(num_envs: int) -> tuple[float, float]:
    """Return (env-steps/s, peak GPU memory in GiB) for a given batch size."""
    cfg = make_payload_env_cfg(num_envs=num_envs)
    env = ManagerBasedRlEnv(cfg, device="cuda")
    env.reset()

    action_dim = env.action_manager.total_action_dim
    actions = torch.zeros(num_envs, action_dim, device="cuda")

    for _ in range(WARMUP):
        actions.uniform_(-1.0, 1.0)
        env.step(actions)
    torch.cuda.synchronize()

    free_before, total = torch.cuda.mem_get_info()
    start = time.perf_counter()
    for _ in range(MEASURE):
        actions.uniform_(-1.0, 1.0)
        env.step(actions)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # mujoco-warp allocates outside torch's caching allocator, so
    # torch.cuda.max_memory_allocated() under-reports badly. mem_get_info sees
    # the true device-level usage.
    free_after, _ = torch.cuda.mem_get_info()
    used_gib = (total - free_after) / 2**30

    sps = MEASURE * num_envs / elapsed
    env.close()
    del env
    torch.cuda.empty_cache()
    return sps, used_gib


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device; this benchmark is meaningless on CPU")

    print(f"device: {torch.cuda.get_device_name(0)}")
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"total memory: {total:.1f} GiB")
    print()
    print(f"{'num_envs':>9} | {'env-steps/s':>12} | {'GPU used (GiB)':>15}")
    print("-" * 43)

    for n in (1024, 2048, 4096, 8192):
        try:
            sps, mem = bench(n)
            print(f"{n:>9} | {sps:>12,.0f} | {mem:>15.2f}")
        except torch.cuda.OutOfMemoryError:
            print(f"{n:>9} | {'OOM':>12} | {'-':>15}")
            torch.cuda.empty_cache()
            break


if __name__ == "__main__":
    main()
