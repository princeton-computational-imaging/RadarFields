<h1 align="center">Radar Fields: Frequency-Space Neural Scene Representations for FMCW Radar</h1>
<p align="center">
   <a href="https://light.princeton.edu/publication/radarfields/">Project Page</a>
    - 
   <a href="https://light.princeton.edu/wp-content/uploads/2024/07/Radar-Fields.pdf">Paper</a>
</p>

<p align="center">
   <img src="assets\teaser.gif" alt="Teaser GIF" width="400" height="auto" style="margin-right: 20px;">
   <img src="assets\teaser2.gif" alt="Teaser GIF 2" width="400" height="auto">
</p>


## News
- [30/8/2024] Radar Fields has been accepted as a spotlight at the ECCV 2024 workshop on [Neural Fields Beyond Conventional Cameras](https://neural-fields-beyond-cams.github.io/)
- [1/8/2024] We presented Radar Fields at [SIGGRAPH 2024](https://dl.acm.org/doi/proceedings/10.1145/3641519?tocHeading=heading44#heading44)

## Installation

```bash
# Set up conda environment & install dependencies
conda env create -f environment.yml
conda activate radarfields

# Install tinycudann
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# Install Radar Fields
pip install -e .
python -c "import radarfields; print(radarfields.__version__)" # Should print "1.0.0"
```

## Running a Model
```bash
# Run a pre-trained demo model
python demo.py --config configs/radarfields.ini --demo --demo_name [DEMO_NAME]

# Run training job
python main.py --config configs/radarfields.ini --name [NAME] --seq [SEQUENCE NAME] --preprocess_file [PATH TO PREPROCESS .JSON]

# More help
python main.py --help
```

## Pre-trained Models
We have several pre-trained models available for [download](https://drive.google.com/drive/folders/1fgfRPabNOHn2uXsKcZn2_owZKuQzj7qc?usp=sharing).

These can be run without downloading any datasets.

## Citation
```bibtex
@inproceedings{radarfields,
author = {Borts, David and Liang, Erich and Broedermann, Tim and Ramazzina, Andrea and Walz, Stefanie and Palladin, Edoardo and Sun, Jipeng and Brueggemann, David and Sakaridis, Christos and Van Gool, Luc and Bijelic, Mario and Heide, Felix},
title = {Radar Fields: Frequency-Space Neural Scene Representations for FMCW Radar},
year = {2024},
isbn = {9798400705250},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3641519.3657510},
doi = {10.1145/3641519.3657510},
booktitle = {ACM SIGGRAPH 2024 Conference Papers},
articleno = {130},
numpages = {10},
keywords = {neural rendering., radar},
location = {Denver, CO, USA},
series = {SIGGRAPH '24}
}
```

## Acknowledgements
The general structure/layout of this codebase was inspired by [LiDAR-NeRF](https://github.com/tangtaogo/lidar-nerf) & [torch-ngp](https://github.com/ashawkey/torch-ngp).

We also rely on [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) for our networks and encodings, and on [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio) for pose optimization.
