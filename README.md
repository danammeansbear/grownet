# 🌱 GreenSight — Computer Vision Plant Analysis Dashboard

> A single-file Python desktop application for real-time greenhouse monitoring, plant health analysis, hydroponics control, and AI-powered crop steering.

---

## Overview
<img width="1906" height="1001" alt="image" src="https://github.com/user-attachments/assets/15fcde75-6a14-4298-9f19-9dade1f9c52b" />
<img width="1903" height="1002" alt="image" src="https://github.com/user-attachments/assets/d3f1add5-040c-4dc3-9ecb-e1ccec8e941d" />
<img width="1905" height="1001" alt="image" src="https://github.com/user-attachments/assets/a0bfd027-74f4-42a1-8757-5d0dd0eb3f48" />
<img width="1881" height="999" alt="image" src="https://github.com/user-attachments/assets/70f988ae-8c74-4be0-9884-c7666a79a83d" />
<img width="1911" height="998" alt="image" src="https://github.com/user-attachments/assets/7d300bf6-d5ae-46cb-bd20-7f8b4ecf2837" />
<img width="1863" height="995" alt="image" src="https://github.com/user-attachments/assets/c5ccc5e1-caf9-4c21-887d-d72a6ce6b2aa" />
<img width="1906" height="1003" alt="image" src="https://github.com/user-attachments/assets/b9842183-11e8-4908-8f67-76795ba326c6" />
<img width="1910" height="973" alt="image" src="https://github.com/user-attachments/assets/77327191-5f41-4fb2-b95f-afb1eeb39b5e" />
<img width="1856" height="1005" alt="image" src="https://github.com/user-attachments/assets/472967f4-0e58-4713-bfe8-843c866a4008" />







GreenSight is a **Tkinter desktop dashboard** built on top of the open-source [PlantCV](https://plantcv.danforthcenter.org/) computer vision library. It provides a unified interface for:

- Multi-camera plant detection and ROI analysis
- Live environmental sensor monitoring (temperature, humidity, CO₂, light, soil moisture)
- Hydroponics control (pH, EC, TDS, DO, ORP, N/P/K) with dosing pump management
- Plant communication panel — per-camera NDVI proxy, health scores, and nutrient deficiency flags
- Plant history with audit log and Holt-Winters growth forecasting
- Automation engine — rule-based triggers (threshold + schedule) that fire pump/alert/analysis actions
- **Leafy** — an AI crop-steering copilot powered by GPT-4o (with offline rule-based fallback)
- 3D Morphology viewer — loads OBJ / STL / PLY files, or reconstructs live from morphology analysis data
- DJI drone / RTMP / RTSP / YouTube stream support

This project is authored and maintained solely by **[@danammeansbear](https://github.com/danammeansbear)**.
It uses PlantCV as a dependency — all PlantCV credit belongs to the [Danforth Center PlantCV team](https://github.com/danforthcenter/plantcv).

---

## Features

| Tab | Description |
|---|---|
| 🌿 Plant Detection | Up to 20 camera panels, auto-analysis, section/role pairing |
| 📊 Environmental Data | Live Arduino serial sensor feed + historical charts |
| 🔧 Sensor Config | Per-sensor Arduino + RabbitMQ AMQP configuration |
| 🗣 Plant Communication | Per-camera health dashboard with NDVI, color metrics, deficiency flags |
| 📋 Plant History | Audit log, filtering, and Holt-Winters forecast charts |
| 💧 Hydroponics | 9-channel hydro monitor, 8 dosing pumps, deficiency analysis, MQTT/Serial/Manual |
| ⚡ Automations | Visual rule editor + background engine (threshold + schedule triggers) |
| 🫸 3D Morphology | OBJ/STL/PLY viewer + live skeleton reconstruction from analysis data |
| 🌱 Ask Leafy | AI crop-steering copilot (GPT-4o or offline rule-based) |

---

## Requirements

```
Python 3.10+
plantcv
opencv-python
Pillow
matplotlib
matplotlib (mpl_toolkits.mplot3d)
numpy
pandas
pyserial
pika          # RabbitMQ (optional)
paho-mqtt     # MQTT hydroponics input (optional)
openai        # Leafy AI copilot (optional)
```

Install via conda (recommended):

```bash
conda create -n plantcv python=3.10
conda activate plantcv
pip install plantcv opencv-python pillow matplotlib numpy pandas pyserial pika paho-mqtt openai
```

---

## Running

```bash
conda activate plantcv
python app.py
```

---

## PlantCV

This project uses [PlantCV](https://github.com/danforthcenter/plantcv) as a library for plant image analysis pipelines.
PlantCV is developed and maintained by the Donald Danforth Plant Science Center and its contributors.
Please cite PlantCV appropriately if you use this dashboard in research:

> Fahlgren et al. (2015). A versatile phenotyping system and analytics platform reveals diverse patterns of the induction kinetics of large dynamic phosphoproteome changes in Arabidopsis.
> *Frontiers in Plant Science*. [https://doi.org/10.3389/fpls.2015.00619](https://doi.org/10.3389/fpls.2015.00619)

---

## License

GreenSight dashboard code (`app.py`, `wordpress-mona-lisa-saas.html`) is released under the MIT License.
The PlantCV library included in this repository retains its original [Mozilla Public License 2.0](LICENSE).

---

## Author

**[@danammeansbear](https://github.com/danammeansbear)**  
All dashboard code written from scratch. PlantCV used as a library dependency only.
[![DeepSource](https://app.deepsource.com/gh/danforthcenter/plantcv.svg/?label=code+coverage&show_trend=true&token=og8rSyKxywOCGkIk8UNiF7B_)](https://app.deepsource.com/gh/danforthcenter/plantcv/)
[![DeepSource](https://app.deepsource.com/gh/danforthcenter/plantcv.svg/?label=active+issues&show_trend=true&token=og8rSyKxywOCGkIk8UNiF7B_)](https://app.deepsource.com/gh/danforthcenter/plantcv/)
[![Documentation Status](https://readthedocs.org/projects/plantcv/badge/?version=stable)](https://docs.plantcv.org/en/stable/?badge=stable)
[![GitHub release](https://img.shields.io/github/release/danforthcenter/plantcv.svg)](https://github.com/danforthcenter/plantcv/releases)
[![PyPI version](https://badge.fury.io/py/plantcv.svg)](https://badge.fury.io/py/plantcv)
![Conda](https://img.shields.io/conda/v/conda-forge/plantcv)
[![license](https://img.shields.io/github/license/danforthcenter/plantcv.svg)](https://github.com/danforthcenter/plantcv/blob/main/LICENSE)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](docs/CODE_OF_CONDUCT.md)
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-71-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

# PlantCV: Plant phenotyping using computer vision

Please use, cite, and [contribute to](https://docs.plantcv.org/CONTRIBUTING/) PlantCV!
If you have questions, please submit them via the
[GitHub issues page](https://github.com/danforthcenter/plantcv/issues).
Follow us on twitter [@plantcv](https://twitter.com/plantcv).

***

## Introduction to PlantCV

PlantCV is an open-source image analysis software package targeted for plant phenotyping. PlantCV provides a common
programming and documentation interface to a collection of image analysis techniques that are integrated from a variety
of source packages and algorithms. PlantCV utilizes a modular architecture that enables flexibility in the design of
analysis workflows and rapid assimilation and integration of new methods. For more information about the project,
links to recorded presentations, and publications using PlantCV, please visit our homepage: 
<https://plantcv.danforthcenter.org/>.

### Quick Links

* [Documentation](https://docs.plantcv.org/)
* [Tutorial Gallery](https://plantcv.org/tutorials)
* [Installation Instructions](https://docs.plantcv.org/installation/)
* [Updating/Changelog](https://docs.plantcv.org/updating/)
* [Public Image Datasets](https://datasci.danforthcenter.org/data/)
* [Contribution Guide](https://docs.plantcv.org/CONTRIBUTING/)
* [Code of Conduct](https://docs.plantcv.org/CODE_OF_CONDUCT/)
* Downloads
  * [GitHub](https://github.com/danforthcenter/plantcv)
  * [PyPI](https://pypi.org/project/plantcv/)
  * [Conda-forge](https://anaconda.org/conda-forge/plantcv)
  * [Zenodo](https://doi.org/10.5281/zenodo.595522)

### Citing PlantCV

If you use PlantCV, please cite the [PlantCV publications](https://plantcv.danforthcenter.org/#plantcv-publications)
relevant to your work. To see how others have used PlantCV in their research, check out our list of 
[publications using PlantCV](https://plantcv.danforthcenter.org/#publications-using-plantcv).

***

## Issues with PlantCV

Please file any PlantCV suggestions/issues/bugs via our 
[GitHub issues page](https://github.com/danforthcenter/plantcv/issues). Please check to see if any related 
issues have already been filed.

***


