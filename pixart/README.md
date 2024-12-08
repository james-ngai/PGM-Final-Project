# Consistency Alignment

## Training

```bash
scripts/run.sh
```

### Config

Update configs `configs/eca.yaml`.

```yaml
network_kwargs:
    class_name: networks.pixart_alpha.PixArt_alpha_DDIM
    ckpt_path: /path/to/pixart # Checkpoint dir

training_args:
    run_dir: ect-run

```

