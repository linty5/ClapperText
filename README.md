# ClapperText

📽️ **ClapperText** is a benchmark dataset for printed and handwritten text recognition in low-resource, visually degraded archival video.  
It was accepted to **ICDAR 2025 Workshop on Document Analysis of Low-resource Languages**, held September 16–21, 2025, in Wuhan, China 🇨🇳.

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2510.15557-red)](https://arxiv.org/abs/2510.15557)
[![Dataset](https://img.shields.io/badge/Dataset-Zenodo-orange)](https://zenodo.org/records/17366964)

---

## 🗂️ Overview

ClapperText contains over 94,000 annotated word instances from 127 historical video segments captured by WWII military cameramen. Each instance includes bounding polygons, transcription, semantic category, handwritten/printed flag, and occlusion status.

<p align="center">
  <img src="clappertext_preview.png" alt="Annotated frame and cropped samples" width="90%">
</p>

<p align="center">
  <em>Examples of annotated data in ClapperText. Left: word-level annotations with semantic labels overlaid on a keyframe. Right: cropped word images including handwritten (top) and printed (bottom).</em>
</p>

---

## 📦 Contents (Coming Soon)

- 📁 **Zenodo (v1.0.0 release):** [DOI 10.5281/zenodo.17366963](https://zenodo.org/records/17366964)  
- 🧪 Evaluation scripts and baseline models
- 📊 Benchmark results (recognition and detection)
- 🛠️ Annotation format documentation

---

## 📄 Citation

This work was accepted to:

> **ICDAR 2025 Workshop on Document Analysis of Low-resource Languages**  
> September 16–21, 2025, Wuhan, Hubei, China

If you use this dataset or paper, please cite:

```bibtex
@misc{lin2025clappertext,
      title={ClapperText: A Benchmark for Text Recognition in Low-Resource Archival Documents}, 
      author={Tingyu Lin and Marco Peer and Florian Kleber and Robert Sablatnig},
      year={2025},
      eprint={2510.15557},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.15557}, 
}
```

---

## 📜 License

- Dataset (annotations + derived images): **CC BY 4.0**  
- Code (evaluation + tools): **MIT License**

---

## 🔗 Related Links

- 📘 HISTORIAN dataset (source video): [Zenodo link](https://zenodo.org/record/6644516)
- 📄 ICDAR 2025: https://icdar2025.org/

---

> Maintained by [@linty5](https://github.com/linty5) | TU Wien, Computer Vision Lab  
> For questions or issues, please open an issue or contact tylin@cvl.tuwien.ac.at
