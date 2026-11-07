# MCD

Code and supplementary examples for the MCD project on reusable motor motifs.
This repository packages the three experiments supplied in `code_submit`.
Internal implementation names such as `spedersac` are retained from that package.

## Repository layout

| Directory | Contents |
| --- | --- |
| `gridworld/` | Multi-task gridworld experiment, training and evaluation scripts, notebooks, and bundled small artifacts. |
| `labyrinth/` | Animal navigation experiment, training and visualization scripts, notebooks, and bundled small artifacts. |
| `free-moving/` | Free-moving animal behavior experiment, representation learning and evaluation code, notebooks, and bundled checkpoints. |
| `media/` | Supplementary videos and an animal-behavior rollout GIF. |
| `environment.yaml` | Original exported Conda environment (`mixer`). |
| `data_manifest.json` | Original paths, byte sizes, and SHA-256 hashes of separately hosted large files. |
| `download_data.py` | Downloader that restores those files to their expected paths and verifies their hashes. |

This release contains the supplied supplementary package, not every experiment
or camera-ready analysis in the development workspace. In particular, it does
not include the later macaque-drawing analysis.

## Environment

The supplied environment targets Linux, Python 3.8, PyTorch 1.12.0, and CUDA 11.6.
It is an exported research environment, not a minimal portable dependency list.

```bash
conda env create -f environment.yaml
conda activate mixer
```

Some scripts use `cuda:0` directly. Check CUDA availability and the configuration
inside the selected script before running it. The package also includes code for
alternative agents; installing the environment and importing every optional agent
have not been validated as part of this repository upload.

## Large data files

Files larger than 10 MiB are separated from Git history and available in the
public dataset repository
[`GeertWang/MCD`](https://huggingface.co/datasets/GeertWang/MCD).
The four files preserve their original paths, and their remote byte sizes and
SHA-256 checksums are checked against `data_manifest.json`.

| File | Size (MiB) |
| --- | ---: |
| `free-moving/kms/test_data_all.pth` | 71.1 |
| `gridworld/model/RandomGridWorld-v0/spedersac/replay_buffer_VI_onehotsa.pkl` | 29.3 |
| `labyrinth/model/labyrinth/spedersac/replay_buffer_labyrinth.pkl` | 241.1 |
| `labyrinth/model/labyrinth/spedersac/replay_buffer_random_labyrinth.pkl` | 183.1 |

Run from the repository root to restore and verify these files:

```bash
pip install huggingface_hub
python download_data.py
```

For a private dataset, first authenticate using `hf auth login` with an account
that has access. You can override the repository with `--repo-id USER/DATASET`.
To check files already restored locally without downloading:

```bash
python download_data.py --verify-only
```

The downloader preserves the paths expected by the scripts. It verifies file
integrity but does not establish that pickle or PyTorch serialization is safe:
only deserialize files obtained from a trusted source.

## Running the supplied experiments

Run each experiment from its own directory because the scripts use relative paths.
The following commands describe the supplied launchers; they are not a claim of
end-to-end reproducibility verification.

### Gridworld

```bash
cd gridworld
bash run.sh
```

The launcher enables training. To evaluate a matching checkpoint, disable the
training command and enable the test command in `run.sh`, as described in the
original `gridworld/README.md`.

### Labyrinth

```bash
cd labyrinth
bash run.sh
```

The supplied launcher starts training. Visualization examples are in
`simulation.ipynb` and `appendix.ipynb`.

### Free-moving animal behavior

Inspect `free-moving/run.sh`, `main.py`, and `visualize.py` before running this
experiment. The supplied launcher currently enables visualization, not training.
Its active command requests a 64-dimensional MCD checkpoint, whereas the bundled
MCD checkpoint directory is named `A_f16_task78_ctrl_critic_nohidden`. Select a
checkpoint and feature dimension that agree before running visualization.

The supplied package includes `kms/test_data_all.pth` as a separate download, but
does **not** include `kms/train_data_all.pth`, which `main.py` expects for training
with the `kms` dataset. Restoring the four large files is therefore not enough
to train every experiment from scratch. Additional artifacts referenced by
individual analysis routines may also need to be generated or provided.

## Checks performed for this upload

- Python syntax parsing passed for all 168 supplied Python source files.
- The three experiment launchers passed `bash -n` shell syntax checking.
- A scan for common credential formats found no matches; this is not a guarantee
  that every possible secret format has been detected.
- Nested Git metadata, Python bytecode caches, and macOS metadata are excluded.
- Experiment source files are preserved; full training, notebook execution,
  environment installation, and quantitative reproduction were not rerun.

No new license is selected by this packaging step. Existing third-party notices
are retained where present; consult the authors regarding reuse permissions.
