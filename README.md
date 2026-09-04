# ClapperText: A Benchmark for Text Recognition in Low-Resource Archival Documents

[![ICDAR 2025 DALL](https://img.shields.io/badge/ICDAR%202025-DALL-1b3b6f)](https://link.springer.com/chapter/10.1007/978-3-032-09371-4_20)
[![Paper](https://img.shields.io/badge/DOI-10.1007%2F978--3--032--09371--4__20-b31b1b)](https://doi.org/10.1007/978-3-032-09371-4_20)
[![Dataset](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.17366964-1682d4)](https://doi.org/10.5281/zenodo.17366964)
[![Code license: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](LICENSE)

Official implementation and dataset release for the ICDAR 2025 DALL paper **"ClapperText: A Benchmark for Text Recognition in Low-Resource Archival Documents"** by Tingyu Lin, Marco Peer, Florian Kleber, and Robert Sablatnig.

<p align="center">
  <img src="clappertext_preview.png" width="90%" alt="Annotated ClapperText frame and cropped handwritten and printed word samples.">
</p>

## Overview

ClapperText is a benchmark for text detection and recognition in degraded archival video. It contains 9,813 annotated frames from 127 World War II-era video segments and 94,573 word instances. Each instance includes a four-point polygon, transcription, semantic category, handwritten or printed label, and occlusion status. Of all instances, 67.4% are handwritten and 1,566 are partially occluded.

The benchmark uses disjoint video-level splits: 18 training videos, 8 validation videos, and 101 test videos. The small training split is designed to evaluate OCR methods under realistic low-resource conditions.

## Repository layout

```text
dataset/                  MMOCR annotations and video-level split lists
LICENSES/                 License text for adapted upstream material
mmocr_clappertext/        MMOCR v1.0.1 configs and experiment scripts
results/                  Reported benchmark metrics
```

`mmocr_clappertext/` is an overlay for the official [MMOCR v1.0.1](https://github.com/open-mmlab/mmocr/tree/v1.0.1) source tree. It contains only the files added or changed for ClapperText.

## Installation

The training and validation pipeline was smoke-tested with a released CRNN config on Linux using Python 3.8.20, PyTorch 1.10.2, CUDA 11.3, MMCV 2.0.1, MMEngine 0.10.6, MMDetection 3.1.0, and MMOCR 1.0.1.

```bash
conda create -n clappertext python=3.8.20 -y
conda activate clappertext
conda install pytorch==1.10.2 torchvision==0.11.3 cudatoolkit=11.3 -c pytorch
pip install -U openmim
mim install "mmengine==0.10.6"
mim install "mmcv==2.0.1"
mim install "mmdet==3.1.0"

git clone --branch v1.0.1 https://github.com/open-mmlab/mmocr.git
pip install -v -e ./mmocr
pip install shapely scipy thop python-Levenshtein
```

Clone this repository next to MMOCR and copy the overlay into the MMOCR checkout:

```bash
git clone https://github.com/linty5/ClapperText.git
cp -r ClapperText/mmocr_clappertext/. mmocr/
cd mmocr
```

## Data preparation

Download `clappertext.zip` from [Zenodo](https://doi.org/10.5281/zenodo.17366964) and extract it under `mmocr/data/clappertext`. Copy the repository's consolidated MMOCR annotations into the same directory:

```bash
mkdir -p data/clappertext
unzip /path/to/clappertext.zip -d data/clappertext
cp -r ../ClapperText/dataset/textdet_finetune data/clappertext/
cp -r ../ClapperText/dataset/textrecog_finetune data/clappertext/
```

The resulting paths used by the released configs are:

```text
data/clappertext/
├── detection/
│   ├── imgs/{train,val,test,test_keyframes}/
│   └── annos/{train,val,test,test_keyframes}/
├── recognition/
│   ├── imgs/{train,val,test_keyframes}/
│   └── annos/{train,val,test_keyframes}/
├── textdet_finetune/{train,val}.json
└── textrecog_finetune/{train,val,test}.json
```

Use the `test_keyframes` subsets for results comparable with the paper.

## Training

Fine-tuned ClapperText checkpoints are not distributed. The released configs initialize from official MMOCR pretrained checkpoints, which are downloaded on first use, and write new checkpoints and logs under `train_logs/recog` or `train_logs/det`.

Run commands from the MMOCR repository root and pass one or more configs to the batch runners:

```bash
python train_rec_batch.py \
    configs/textrecog/nrtr/nrtr_resnet31-1by16-1by8_6e_st_mj_250702.py

python train_det_batch.py \
    configs/textdet/dbnetpp/dbnetpp_resnet50_fpnc_1200e_icdar2015_250630.py
```

Use `--help` to configure output directories, distributed launchers, mixed precision, and resume behavior. The released fine-tuning annotations preserve the paper's balanced per-video sampling protocol: samples from short clips are repeated so that each selected video contributes 20 frames.

## Evaluation

Text recognition is evaluated with case- and symbol-normalized word recognition accuracy, averaged per video. Text detection is evaluated with polygon Hmean at IoU 0.5, also averaged per video. Training creates one experiment directory per config; the evaluation scripts read the saved config and its `best_*.pth` checkpoint from that directory.

```bash
python infer_rec_dir_calmetric.py \
    --exp-dir train_logs/recog \
    --models nrtr_resnet31-1by16-1by8_6e_st_mj_250702 \
    --modes zeroshot finetune \
    --image-root data/clappertext/recognition/imgs/test_keyframes \
    --gt-dir data/clappertext/recognition/annos/test_keyframes

python infer_det_dir_calmetric.py \
    --exp-dir train_logs/det \
    --models dbnetpp_resnet50_fpnc_1200e_icdar2015_250630 \
    --modes zeroshot finetune \
    --data-dir data/clappertext/detection/imgs/test_keyframes \
    --gt-dir data/clappertext/detection/annos/test_keyframes \
    --thresholds-file ../ClapperText/results/detection/benchmark_results.json
```

`zeroshot` uses the official MMOCR checkpoint referenced by the saved config; `finetune` uses the locally trained checkpoint. The detection results file records the per-model operating points used for the reported metrics. Use `--score-thr` to override them, and add `--measure-deployment` only when runtime and memory measurements are needed.

The recognition configs cover CRNN, MASTER, NRTR, RobustScanner, SAR, and SVTR. The detection configs cover DBNet, DBNet++, FCENet, Mask R-CNN, PANet, PSENet, and TextSnake, including the augmentation ablations reported in the paper.

Precomputed metrics for the reported experiments are available in
`results/detection/benchmark_results.json` and
`results/recognition/benchmark_results.json`.

## Citation

```bibtex
@inproceedings{lin2026clappertext,
  author    = {Lin, Tingyu and Peer, Marco and Kleber, Florian and Sablatnig, Robert},
  title     = {{ClapperText}: A Benchmark for Text Recognition in Low-Resource Archival Documents},
  booktitle = {Document Analysis and Recognition -- ICDAR 2025 Workshops},
  pages     = {329--346},
  publisher = {Springer Nature Switzerland},
  year      = {2026},
  doi       = {10.1007/978-3-032-09371-4_20}
}
```

## License and acknowledgements

Except where otherwise noted, ClapperText code is released under the [MIT License](LICENSE), and the dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Configuration material adapted from [MMOCR v1.0.1](https://github.com/open-mmlab/mmocr/tree/v1.0.1) remains subject to the [Apache License 2.0](LICENSES/Apache-2.0.txt).

This work was supported by the Austrian Science Fund (FWF), doc.funds.connect project DFH 37-N, "Visual Heritage: Visual Analytics and Computer Vision Meet Cultural Heritage." It builds on [MMOCR](https://github.com/open-mmlab/mmocr), the [HISTORIAN dataset](https://doi.org/10.1109/ICIP46576.2022.9897300), and [CVAT](https://www.cvat.ai/).

## Contact

Tingyu Lin, Computer Vision Lab, TU Wien, [tylin@cvl.tuwien.ac.at](mailto:tylin@cvl.tuwien.ac.at)
