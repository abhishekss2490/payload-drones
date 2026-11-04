# payload-drones

Cooperative aerial payload-transport training and evaluation on MJLab. This
package is installed editable in its own uv project environment.

```bash
uv sync --locked
uv run --locked python -m payload_drones.train Mjlab-PayloadTransport-RMA
uv run --locked python -m payload_drones.evaluate Mjlab-PayloadTransport-RMA --run-name <wandb-name>
uv run --locked python -m payload_drones.play Mjlab-PayloadTransport-RMA-Phase1
uv run --locked python -m payload_drones.play_stats Mjlab-PayloadTransport-RMA-Phase1
```

This project selects Python 3.12.11 with `.python-version` and installs
`robotics-common` from the Git commit pinned in `pyproject.toml` and
`uv.lock`. A sibling checkout is optional. This project has its own
`.venv`. Update the common commit when this project is ready for a shared
change.

Submit a training run from this directory with:

```bash
scripts/slurm-train PIPELINE_NAME [train args...]
```

Training and evaluation require a clean nested git repository. Training records
the commit and branch in W&B and tags the commit as
`runs/<pipeline>/<run-name>`. Evaluation runs one fresh episode for both phases.

## Archived current training run

This branch preserves the Ubuntu training source at `a1954c2b8d77e6e0e2131e999c41f93a90cc0147`, with no policy, reward, curriculum or optimizer changes. The existing `main` branch is preserved separately. This is an archive of the current setup; no encoder alternative has been implemented or trained here.

- `models/mappo_v1-20261027T003522Z/phases/01-teacher/`: all 46 locally available checkpoints, iterations 0 through 4500.
- `results/training/`: saved environment/agent parameters, run manifest, TensorBoard metrics and console log. Training seed: **4551**.
- `results/evaluation/`: numerical evaluation outputs; original job and seed names are retained.
- `results/diagnostics/`: the y-only paired, failure and height tests from server jobs 73168, 73184 and 73185.
- `results/local_replay/`: local validation recordings, separately labelled from server results.
- `results/derived/`: offline numerical summaries and plots.
- `results/artifact_manifest.json`: file provenance, hashes, seeds and archive limitations.

The downloaded run manifest still says `running`; that is a historical record, not a live job status. The latest available checkpoint is 4500; no completed distillation model is included. Half-speed raw seeds 3 and 4 were not available locally. Evaluation seeds 0–4 are retained for future comparisons, with the smaller diagnostic seed subsets recorded in the manifest. No new training was launched during archiving.

Verify the archived artifacts from the repository root:

```bash
sha256sum --check results/SHA256SUMS
```

Local notes, dossiers, agent memory and generated videos are excluded by `.gitignore`. Only the curated `models/` and `results/` archives are committed; normal live run outputs remain ignored.
