# Experiment Matrix Runner

## Run the smoke matrix

```bash
python -m fractal_swin_unet.exp.run --matrix configs/matrices/smoke_matrix.yaml
```

## Outputs

- `reports/<matrix_name>_summary.csv`
- `reports/<matrix_name>_summary.md`
- `reports/<matrix_name>_runs.json`
- `reports/<matrix_name>_qual/` (PNG panels)

## Add a new experiment

Add an entry under `experiments` with a base config and overrides:

```yaml
- name: "fractal_topology_loss"
  config: "configs/smoke_losses_topology.yaml"
  overrides:
    model.name: "fractal_prior"
    loss.cldice.enabled: true
    loss.cldice.weight: 0.2
```

Overrides are dot-notation keys applied on top of the loaded config.
