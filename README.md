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


## Environment

The supplied environment targets Linux, Python 3.8, PyTorch 1.12.0, and CUDA 11.6.

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
The five files preserve their original paths, and their remote byte sizes and
SHA-256 checksums are checked against `data_manifest.json`.

| File | Size (MiB) |
| --- | ---: |
| `free-moving/kms/train_data_all.pth` | 283.9 |
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

The experiment uses separate training and test replay buffers, with
16-dimensional states and 16-dimensional displacement actions:

| Path within `free-moving/` | Role | Transitions |
| --- | --- | ---: |
| `kms/train_data_all.pth` | Representation learning through `main.py` | 515,120 |
| `kms/test_data_all.pth` | Held-out analysis through `visualize.py` | 128,781 |

Both buffers are available in the public Hugging Face dataset linked above.
Run `python download_data.py` from the repository root to restore both files to
their expected paths and verify their checksums before training or analysis.

To select training, comment out the entire `python visualize.py` block in
`free-moving/run.sh` and uncomment the `python main.py --alg spedersac` block
above it, including all continuation lines. Then run:

```bash
cd free-moving
bash run.sh
```

**Training-call note for this snapshot:** the loop in `main.py` currently passes
`seq_len=kwargs['seq_len']` for every algorithm, but the MCD branch does not
define this option and its training method does not accept it. For MCD, the call
must instead be `agent.train(expert_buffer, batch_size=args.batch_size)`.
Sequence-based baseline methods retain their own `seq_len` argument. This
documentation update does not modify the experiment source or start training.

To select visualization instead, enable the `python visualize.py` block and
disable the training block. Match `--dir`, `--feature_dim`, `--actor_type`, and
the requested checkpoint step to the model being loaded. The bundled MCD
checkpoint has 16 features, whereas the default visualization command requests
64; the training and visualization examples also specify different actor types.
Adjust these settings to match the checkpoint before visualization. Additional
artifacts used by individual analysis routines may need to be generated first.

## License

Unless otherwise noted, this repository is licensed under the
[Creative Commons Attribution 4.0 International License (CC BY 4.0)](LICENSE).
Third-party components retain their respective licenses and attribution notices.
