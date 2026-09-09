<!-- Copyright (c) 2026 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# SIPS: Stochastic Interpolant Prior for Speech

[![arXiv](https://img.shields.io/badge/arXiv-2605.06189-b31b1b.svg)](https://arxiv.org/abs/2605.06189)

This repository contains the reference code for **SIPS: Stochastic Interpolant Prior for Speech**. It provides training and inference code for speech enhancement and speech separation with SIPS, along with bundled third-party predictor/model components used by the research pipeline.

## Citation

If you use this repository, please cite:

Julius Richter, Yoshiki Masuyama, Christoph Boeddeker, Takahiro Edo, Gordon Wichern, and Jonathan Le Roux. "Predictive-Generative Drift Decomposition for Speech Enhancement and Separation." arXiv preprint arXiv:2605.06189 (2026).

```bibtex
@article{richter2026predictive,
  title   = {Predictive-Generative Drift Decomposition for Speech Enhancement and Separation},
  author  = {Richter, Julius and Masuyama, Yoshiki and Boeddeker, Christoph and Edo, Takahiro and Wichern, Gordon and {Le Roux}, Jonathan},
  journal = {arXiv preprint arXiv:2605.06189},
  year    = {2026}
}
```

## Table of Contents

- [Citation](#citation)
- [Repository Layout](#repository-layout)
- [Installation](#installation)
  - [Base Installation with Git LFS](#base-installation-with-git-lfs)
  - [Base Installation without Git LFS](#base-installation-without-git-lfs)
  - [Conda Base Installation Example](#conda-base-installation-example)
  - [SEMamba Installation](#semamba-installation)
- [Data Layout](#data-layout)
- [Checkpoints](#checkpoints)
- [Training](#training)
- [Inference](#inference)
  - [Predictor-only Inference](#predictor-only-inference)
- [Contributing](#contributing)
- [Copyright and License](#copyright-and-license)

## Repository Layout

- `sips_speech/train.py`: command-line entry point for training.
- `sips_speech/infer.py`: command-line entry point for inference.
- `sips_speech/training/`: dataset loading, training loop, loss, EMA, checkpointing, logging, and schedules.
- `sips_speech/inference/`: sampling and WAV enhancement pipeline.
- `sips_speech/models/`: SIPS model and spectrogram encoder.
- `sips_speech/predictors/`: predictor modules used to generate predictor outputs for SIPS.
- `sips_speech/third_party/`: bundled third-party model code used by predictors and the SIPS backbone.
- `checkpoints/`: expected location for model and predictor checkpoints.

## Installation

SIPS provides two installation paths:

1. [Base installation](#base-installation-with-git-lfs), which supports training, inference with precomputed predictor outputs via `--proc_dir`, and predictors that do not require the SEMamba CUDA extension, such as `--predictor ncsnpp`.
2. [SEMamba installation](#semamba-installation), which adds support for the online SEMamba predictor and requires a compatible CUDA-enabled PyTorch environment and the compiled `mamba-ssm` selective-scan extension.

We recommend using a fresh virtual environment for either installation path, although the choice of environment manager is up to the user.

### Base Installation with Git LFS

The recommended base installation uses Git LFS so that released checkpoints are downloaded as complete files rather than Git LFS pointer files:

```bash
git lfs install
git clone https://github.com/merlresearch/sips-speech.git
cd sips-speech

python -m pip install --upgrade pip setuptools wheel

# Optional: Install PyTorch manually first if the default build does not match
# your CUDA or hardware setup. See:
# https://pytorch.org/get-started/locally/

python -m pip install -e .
```

### Base Installation without Git LFS

If Git LFS is not available, clone the repository without downloading LFS objects and download the released checkpoint files directly from GitHub:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/merlresearch/sips-speech.git
cd sips-speech

mkdir -p checkpoints
for file in SIPS.pt Conv-TasNet.pt flexio_avg.ckpt; do
  curl -L "https://github.com/merlresearch/sips-speech/raw/main/checkpoints/${file}" \
    -o "checkpoints/${file}"
done

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

### Conda Base Installation Example

The following example uses Conda to create and manage an isolated environment for the base installation:

```bash
conda create -n sips-speech python git-lfs -c conda-forge -y
conda activate sips-speech

git lfs install
git clone https://github.com/merlresearch/sips-speech.git
cd sips-speech

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

### SEMamba Installation

The online SEMamba predictor requires the [Mamba selective-scan CUDA extension](https://pypi.org/project/mamba-ssm/). Unlike the base SIPS installation, this extension depends on a compiled CUDA/C++ wheel and is therefore sensitive to the Python version, PyTorch version, CUDA version, PyTorch C++ ABI, and GPU architecture.

The setup below is a known-good configuration using Python 3.10, PyTorch 2.6.0, CUDA 12.6 wheels, and a matching prebuilt `mamba-ssm` wheel. Other combinations may work if a matching `mamba-ssm` wheel is available, but they should be tested separately.

If installation fails or your environment resolver selects different package versions, compare against `requirements-pinned.txt`. It records the exact package versions used for the tested SEMamba environment, including the PyTorch CUDA wheels and the `mamba-ssm` wheel URL.

```bash
conda create -n sips-mamba python=3.10 -y
conda activate sips-mamba

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
```

The `mamba-ssm` wheel must match the C++ ABI used by the installed PyTorch build. The following command detects the ABI automatically and installs the corresponding wheel:

```bash
ABI=$(python -c "import sys, torch; print(torch.__version__, torch.version.cuda, torch._C._GLIBCXX_USE_CXX11_ABI, file=sys.stderr); print(str(torch._C._GLIBCXX_USE_CXX11_ABI).upper())")

case "$ABI" in
  TRUE|FALSE)
    python -m pip install --no-deps \
      "https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1%2Bcu12torch2.6cxx11abi${ABI}-cp310-cp310-linux_x86_64.whl"
    ;;
  *)
    echo "Failed to determine the PyTorch C++ ABI. Got: '$ABI'" >&2
    exit 1
    ;;
esac
```

To inspect the installed PyTorch, CUDA, and ABI versions manually, run:

```bash
python -c "import sys, torch; print(torch.__version__, torch.version.cuda, torch._C._GLIBCXX_USE_CXX11_ABI, file=sys.stderr); print(str(torch._C._GLIBCXX_USE_CXX11_ABI).upper())"
```

> **Note:** Installing the wheel with `--no-deps` may leave `pip` dependency warnings for optional `mamba-ssm` package requirements. These warnings can be ignored for SIPS, since SEMamba only uses the compiled `selective_scan_cuda` extension from the wheel.

Then install SIPS:

```bash
python -m pip install -e .
```

Smoke-test the SEMamba extension:

```bash
python -c "import torch; import selective_scan_cuda; print(\"selective_scan_cuda ok\")"
python -c "from sips_speech.third_party.semamba import SEMamba; print(\"SEMamba import ok\")"
sips-infer --help
```

> **Note:** The online SEMamba predictor requires a GPU architecture supported by the installed `mamba-ssm` CUDA extension. In practice, the prebuilt wheel may require a newer NVIDIA compute capability than older Pascal-generation GPUs such as NVIDIA TITAN Xp / compute capability 6.1.
> If this happens, use a different predictor (e.g., `--predictor ncsnpp`) or run SIPS with precomputed predictor outputs via `--proc_dir`, both of which avoid the online SEMamba CUDA extension.

## Data Layout

Training expects 16 kHz mono WAV files. Set `--data CLEAN_TRAIN_DIR` to the clean training WAV directory. The dataset loader will recursively discover and read all `*.wav` files under that path.

For validation during training, pass `--valid_dir` for the noisy validation WAV directory and `--valid_pred_dir` for the matching precomputed predictor WAV directory.

## Checkpoints

The released binary checkpoints are tracked with Git LFS:

```text
checkpoints/SIPS.pt
checkpoints/Conv-TasNet.pt
checkpoints/flexio_avg.ckpt
```

The SIPS model config is included as a regular Git file at `checkpoints/SIPS_config.json`. Its `dataset_kwargs.path` value is intentionally unset in the release because dataset locations are machine-specific; inference only reads the model settings from this file.

For Git LFS setup and the manual checkpoint download fallback for machines without Git LFS, see [Base Installation without Git LFS](#base-installation-without-git-lfs).

Some predictor checkpoints are downloaded automatically by `sips_speech/predictors/registry.py` when a predictor is selected.

## Training

Set `--nproc_per_node=NUM_GPUS` to the number of GPUs to use on the node for distributed training. For example, use `--nproc_per_node=2` to train on 2 GPUs. For single-GPU training, use `--nproc_per_node=1`, or call the script directly with `python -m sips_speech.train` instead of `torchrun`.

The training option `--batch` is the total batch size across all distributed workers, not the per-GPU batch size. The training loop splits this total batch across `--nproc_per_node` processes and uses gradient accumulation if `--batch_gpu` is set. For example, `--nproc_per_node=2 ... --batch=16` gives an effective training batch of 16, with 8 examples per GPU (`--nproc_per_node` is a `torchrun` option; `--batch` is passed to the training script). Keeping `--batch` fixed should therefore keep the optimizer batch size and learning-rate schedule fixed when changing the number of GPUs, although the total batch size must be divisible by the number of workers.

The released SIPS checkpoint follows the paper recipe: two NVIDIA A40 GPUs, `--batch=16` total (`2x8` per-GPU), and `--duration=4096K`. The `K` suffix is parsed as 1024, so this corresponds to `4_194_304` training examples, i.e., about 4.2M two-second training samples. The SIPS model itself does not use batch-normalization layers with running statistics, so changing the number of GPUs is not expected to change batch-normalization statistics for this checkpoint recipe; the main practical effects are memory use and wall-clock time.

Example single-node distributed training command:

```bash
torchrun --standalone --nproc_per_node=2 -m sips_speech.train \
  --run_dir runs/sips_vbdmd \
  --data /path/to/VoiceBank-DEMAND_16k/train/clean \
  --valid_dir /path/to/VoiceBank-DEMAND_16k/valid/noisy \
  --valid_pred_dir /path/to/precomputed/valid/predictor \
  --batch 16 \
  --duration 4096K
```

In this command, `--run_dir` is the output directory for logs, snapshots, checkpoints, and the saved JSON training configuration. `--data` points to the clean training WAV directory. `--valid_dir` points to the noisy validation WAV directory, and `--valid_pred_dir` points to matching precomputed predictor WAV files used during validation. `--batch` sets the total distributed training batch size, and `--duration` sets the total number of training examples to process before stopping.

## Inference

Set `--nproc_per_node=NUM_GPUS` to the number of GPUs to use for distributed inference. For single-GPU inference, use `--nproc_per_node=1`, or call the script directly with `python -m sips_speech.infer` instead of `torchrun`.

SIPS inference reads WAV inputs as mono audio. If a WAV file has multiple channels, only the first channel is used and a warning is emitted once per process.

Example inference with an online predictor:

```bash
torchrun --standalone --nproc_per_node=1 -m sips_speech.infer \
  --in_dir /path/to/noisy_wavs \
  --out_dir out/sips_ncsnpp/files \
  --model checkpoints/SIPS.pt \
  --predictor ncsnpp \
  --num_steps 15 \
  --kappa 0.0
```

For SIPS inference with FlexIO, pass `--predictor flexio`. FlexIO uses `--num_speakers 2` by default when running with SIPS; set this explicitly if your mixtures contain a different number of speakers.

```bash
torchrun --standalone --nproc_per_node=1 -m sips_speech.infer \
  --in_dir /path/to/mixture_wavs \
  --out_dir out/sips_flexio/files \
  --model checkpoints/SIPS.pt \
  --predictor flexio \
  --num_speakers 2 \
  --num_steps 15 \
  --kappa 0.0
```

Example inference with precomputed predictor outputs:

```bash
torchrun --standalone --nproc_per_node=1 -m sips_speech.infer \
  --in_dir /path/to/noisy_wavs \
  --proc_dir /path/to/predictor_wavs \
  --out_dir out/sips/files \
  --model checkpoints/SIPS.pt \
  --num_steps 15 \
  --kappa 0.0
```

After installation, `sips-infer` can be used instead of `python -m sips_speech.infer`. For example, to run the default SIPS refinement using precomputed predictor outputs:

```bash
sips-infer \
  --in_dir /path/to/noisy_wavs \
  --proc_dir /path/to/predictor_wavs \
  --out_dir out/sips/files
```

### Predictor-only Inference

Predictors can also be run without SIPS to write their direct outputs to disk. The supported module entry points are `ncsnpp`, `semamba`, `convtasnet`, and `flexio`:

```bash
torchrun --standalone --nproc_per_node=1 -m sips_speech.predictors.ncsnpp.infer \
  --in_dir /path/to/noisy_wavs \
  --out_dir out/ncsnpp/files
```

For FlexIO predictor-only inference, separated speakers are written under `s1/`, `s2/`, etc. FlexIO defaults to `--num_speakers 2`:

```bash
torchrun --standalone --nproc_per_node=1 -m sips_speech.predictors.flexio.infer \
  --in_dir /path/to/mixture_wavs \
  --out_dir out/flexio/files \
  --num_speakers 2
```

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for our policy on contributions.

## Copyright and License

Released under `AGPL-3.0-or-later` license, as found in the [LICENSE.md](LICENSE.md) file.

All files, except as noted below:
```text
Copyright (c) 2026 Mitsubishi Electric Research Laboratories (MERL)
SPDX-License-Identifier: AGPL-3.0-or-later
```

Some files under `sips_speech/third_party/` contain or adapt third-party code under separate licenses:

- `convtasnet.py`
  - Source: [Conv-TasNet](https://github.com/kaituoxu/Conv-TasNet)
  - License: [MIT](LICENSES/MIT.md)

- `ncsnpp.py`
  - Sources: [StoRM](https://github.com/sp-uhh/storm) and [score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch)
  - Licenses: [MIT](LICENSES/MIT.md) and [Apache-2.0](LICENSES/Apache-2.0.txt)

- `semamba.py`
  - Sources: [SEMamba](https://github.com/RoyChao19477/SEMamba), [Mamba](https://github.com/state-spaces/mamba), [MP-SENet](https://github.com/yxlu-0102/MP-SENet), and [CMGAN](https://github.com/ruizhecao96/CMGAN)
  - Licenses: [MIT](LICENSES/MIT.md) and [Apache-2.0](LICENSES/Apache-2.0.txt)

- `edm2.py`
  - Source: [EDM2](https://github.com/NVlabs/edm2)
  - License: [CC-BY-NC-SA-4.0](LICENSES/CC-BY-NC-SA-4.0.txt)

Copyright notices and additional attribution details are retained in the respective source files.
