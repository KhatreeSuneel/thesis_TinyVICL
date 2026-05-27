# Dataset Information
In the following sections, you will find the information about the datasets used for this master thesis.

## Datasets Used for the model training

### ADE20K
_Scene Parsing through ADE20K Dataset. B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso and A. Torralba. Computer Vision and Pattern Recognition (CVPR), 2017. [PDF] [bib]_
_Semantic Understanding of Scenes through ADE20K Dataset. B. Zhou, H. Zhao, X. Puig, T. Xiao, S. Fidler, A. Barriuso and A. Torralba. International Journal on Computer Vision (IJCV) [PDF][bib]_
- **Task**: Segmentation
- **Labels**: Default
- **Download Link**: http://sceneparsing.csail.mit.edu/

### MVTec AD
_Bergmann, P., Batzner, K., Fauser, M., Sattlegger, D., & Steger, C. (2021). The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection. International Journal of Computer Vision, 129(4), 1038–1059. https://doi.org/10.1007/s11263-020-01400-4_

- **Task**: Inpainting
- **Labels**: Random black rectangular patched overlayed on the images
- **Download Link**: https://www.mvtec.com/research-teaching/datasets/mvtec-ad

### RAISE
_D.-T. Dang-Nguyen, C. Pasquini, V. Conotter, G. Boato, RAISE – A Raw Images Dataset for Digital Image Forensics, ACM Multimedia Systems, Portland, Oregon, March 18-20, 2015_

- **Task**: Denoising
- **Labels**: Gaussian Noise applied on original image
- **Download Link**: https://loki.disi.unitn.it/RAISE/download.html

### RAISE
_D.-T. Dang-Nguyen, C. Pasquini, V. Conotter, G. Boato, RAISE – A Raw Images Dataset for Digital Image Forensics, ACM Multimedia Systems, Portland, Oregon, March 18-20, 2015_

- **Task**: Superresolution
- **Labels**: Original image downscaled
- **Download Link**: https://loki.disi.unitn.it/RAISE/download.html

### BIPED, MDBD, COCO
_Soria, X., Sappa, A., Humanante, P., & Akbarinia, A. (2023). Dense Extreme Inception Network for Edge Detection. Pattern Recognition, 139, 109461. https://doi.org/10.1016/j.patcog.2023.109461_
_Mély, D. A., Kim, J., McGill, M., Guo, Y., & Serre, T. (2016). A systematic comparison between visual cues for boundary detection. Vision Research, 120, 93–107. https://doi.org/10.1016/j.visres.2015.11.007_
_Lin, T.-Y., Maire, M., Belongie, S., Bourdev, L., Girshick, R., Hays, J., Perona, P., Ramanan, D., Zitnick, C. L., & Dollár, P. (2015). Microsoft COCO: Common Objects in Context (arXiv:1405.0312). arXiv. https://doi.org/10.48550/arXiv.1405.0312_

- **Task**: Edge Detection
- **Labels**: Edge map following the Protocol from DexiNed (https://doi.ieeecomputersociety.org/10.1109/WACV45572.2020.9093290)
- **Download Link**: https://opendatalab.com/OpenDataLab/MDBD, https://xavysp.github.io/MBIPED/, https://cocodataset.org/#home

### HRWSI
_Xian, Ke, et al. "Structure-Guided Ranking Loss for Single Image Depth Prediction." IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2020._

- **Task**: Depth Estimation
- **Labels**: Default label
- **Download Link**: https://github.com/KexianHust/Structure-Guided-Ranking-Loss

### Rain13K
_Jiang, K., Wang, Z., Yi, P., Chen, C., Huang, B., Luo, Y., Ma, J., & Jiang, J. (2020). Multi-Scale Progressive Fusion Network for Single Image Deraining (arXiv:2003.10985). arXiv. https://doi.org/10.48550/arXiv.2003.10985_

- **Task**: De-raining
- **Labels**: Default label
- **Download Link**: https://github.com/kuijiang94/MSPFN


## Datasets Used for Downstream Evaluation

### Pascal VOC
- **Task**: Foreground Segmentation
- **Labels**: Segmentation masks
- **Download Link**: https://docs.ultralytics.com/datasets/detect/voc

### DID-MDN
_Zhang, He, and Vishal M. Patel. "Density-aware Single Image De-raining using a Multi-stream Dense Network." IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2018._

- **Task**: Deraining
- **Labels**: Clean images
- **Download Link**: https://github.com/hezhangsprinter/DID-MDN

### BSDS500
_Arbelaez, Pablo, et al. "Contour Detection and Hierarchical Image Segmentation." IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 33, no. 5, pp. 898–916. 2011._

- **Task**: Edge Detection
- **Labels**: Edge maps
- **Download Link**: https://github.com/BIDS/BSDS500

### NYU Depth V2
- **Task**: Depth Estimation
- **Labels**: Depth maps
- **Download Link**: https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html

### ImageNet
- **Task**: Image Colorization
- **Labels**: Color images
- **Download Link**: https://www.image-net.org


## Datasets Preparation for Downstream Evaluation

### Foreground Segmentation
For dataset preparation refer to the [Pascal VOC setup guide](https://docs.ultralytics.com/datasets/detect/voc).

### Other Tasks (colorization, deraining, depth_estimation, edge_detection)
Organize your dataset in the following structure:

```
data/
└── <task>/
    ├── images/
    │   └── img001.jpg ...
    └── labels/
        └── img001.jpg ...
```

Each task requires a CSV file that specifies which images to use as support (few-shot context) and which as the query. The CSV follows this structure — each row is one evaluation sample:

| Column | Description |
|---|---|
| `support_1_image` | Filename of 1st support image |
| `support_1_label` | Filename of 1st support label |
| ... | (repeated for each shot) |
| `support_N_image` | Filename of Nth support image |
| `support_N_label` | Filename of Nth support label |
| `query_image` | Filename of the query image |
| `query_label` | Filename of the query label (ground truth) |

Example row from a 2-shot CSV:

```
support_1_image,support_1_label,support_2_image,support_2_label,query_image,query_label
15011.jpg,15011.jpg,36046.jpg,36046.jpg,100007.jpg,100007.jpg
```

