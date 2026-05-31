
<p style="text-align: center;">
    <img src="https://img.shields.io/badge/python-3.8.0-blue?logo=python" alt="python-version">
    <img src="https://img.shields.io/badge/pytorch-lightning-darkviolet?logo=lightning" alt="pytorch-lightning">
    <img src="https://img.shields.io/badge/tensorboard-gray?logo=tensorflow" alt="tensorboard">
</p>

<p style="text-align: center;">
    <a href="#-about-the-project">About the Project</a> | 
    <a href="#folder-structure">Folder Structure</a> | 
    <a href="#-installation">Installation</a> | 
    <a href="#-how-to-run">How to Run</a> | 
    <a href="#-additional-documentation">Additional Documentation</a> |
    <a href="#-references">References</a>
</p>

# About the Project

This project stress-tests the role of scale in Visual In-Context Learning (VICL) by training a tiny 1-million-parameter model on 70,000 images and comparing it against models that are 7,000× larger. Experiments span three adaptive settings: small distribution shifts, unseen task encodings, and completely new tasks. The results expose key gaps in how adaptive capabilities are currently benchmarked, pointing to a need for more rigorous VICL evaluation.

# Folder Structure 
```
📦TinyVicl (root)
 ┣ 📂assets                                      <-- Contains saved figures, screenshots, ...
 ┣ 📂configs                                     <-- Configuration files for the pipeline
 ┃  ┗ ⚙️config.yaml                              <-- Configuration file for SSH development (default)
 ┣ 📂data                                        <-- Provided data (.csv files, training images, ...)
 ┣ 📂logs                                        <-- Contains logs from the training, e.g. tensorboard logs
 ┣ 📂models                                      <-- Saved models during Development
 ┃  ┣ 📂Generalizer                              <-- Checkpoints of trained Model
 ┣ 📂notebooks                                   <-- Jupyter Notebooks used for experimentations
 ┃  ┣ 📂data                                     <-- EDA Notebooks, Dataloader and Dataset Notebooks
 ┃  ┣ 📂generalizer                              <-- Inference notebook for neuralizer
 ┃  ┣ 📂tasks                                    <-- Task visualization notebook 
 ┣ 📂scripts                                     <-- Standalone scripts    
 ┃  ┣ 📜evaluation.py                            <-- Evaluation protocol 
 ┣ 📂src                                         <-- Source code / modules / classes
 ┃  ┣ 📂data                                     <-- Data related functionalities to collect images and preprocessing them
 ┃  ┣ 📂dataset                                  <-- Contains PyTorch Dataset, DataLoader and custom Sampler
 ┃  ┣ 📂eval                                     <-- Contains logic for evaluation on downstream tasks
 ┃  ┣ 📂neuralizer                               <-- Contains source code from the neuralizer model
 ┃  ┣ 📂tasks                                    <-- Contains implementation of all tasks
 ┃  ┣ 📂train                                    <-- Contains logic for the actual fitting of the models
 ┃  ┣ 📂unet                                     <-- Contains basic UNet
 ┃  ┣ 📂utils                                    <-- Contains all sorts of utility functions
 ┃  ┗ 📂visualizations                           <-- Contains multiple plottings using matplotlib + seaborn
 ┣ 🕹️main.py                                     <-- Entry point of the pipeline
 ┣ 📜README.md                                   <-- The top-level README for developers using this project
 ┗ 📜requirements.txt                            <-- The requirenments file for reproducing the environment
```

# 👨🏽‍💻 Installation

1. Clone the repository 

2. Navigate to the project root directory by running the following command in your terminal:
   ```shell
   cd thesis_TinyVICL
   ```

3. Create a virtual environment and activate it.
   ```shell
   python3 -m venv venv
   source venv/bin/activate
   ```

4. Install the required packages by running the following command in your terminal:
   ```shell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```


# 🚀 How To Run
In the following, you will find the necessary steps to run this pipeline.

## 1. Data Preperation
- First put the data into `data/x` while *x* specifies the data set (e.g. imagenet).
   - If you put the data somewhere else: Specify `search_dir` in configs/config.yaml accordingly.
- The `main.py` module will then create `metadata.csv`, if it does not exist yet.
   - The dataframe will contain essential meta information about all images


## 2. Training
For training the specific model, please refer to the `main.py` module documentation. An example of fitting Neuralizer on GPU with ID 1 and the given config file located in `./config`:
```shell
CUDA_VISIBLE_DEVICES=1 python3 main.py --fit-neuralizer --config "configs/config.yaml"
```
The resulting model is saved in the directory specified in the `training/tensorboard_logger/save_dir` entry in the `config/configs.yaml` file.

## 3. Evaluation
### Evaluation on Multitask Capability of model
Please refer to the module documentation of the stand alone evaluation scripts found in `./scripts`. Example to evaluate models (eval_objects) on GPU 1 on their multitask capabilities:
```shell
CUDA_VISIBLE_DEVICES=1 python3 scripts/evaluation.py --run-eval
```
Results in .csv files in `./data/evaluation-results/` containing the scores.

### Evaluation on Downstream task 
Please refer to [Datasets documentation](docs/datasets.md) for the downstream dataset preparation.
Example to evaluate model on GPU 1 on downstream tasks [colorization, deraining, depth_estimation, edge_detection] :
```shell
CUDA_VISIBLE_DEVICES=1 python3 evaluate_reconstruction.py --csv_dir path/to/dataset.csv --data_dir path/to/images --label_dir path/to/labels --output_dir path/to/output_dir --ckpt_path path/to/checkpoint.ckpt --task "colorization" --num_shots 2
```
Example to evaluate model on GPU 1 on downstream task [foreground segmentation]:
```shell
CUDA_VISIBLE_DEVICES=1 python3 evaluate_segmentation.py --base_dir path/to/pascal_voc --output_dir path/to/output_dir --ckpt_path path/to/checkpoint.ckpt --split 0 --num_shots 2
```

## Trained models

Trained model weights can be found in the [models/ folder](models/).


# 📚 Additional Documentation
- [Datasets documentation](docs/datasets.md)

# 🔗 References and Acknowledgements
This project is based on the following work:
- Czolbe, Steffen, and Adrian V. Dalca. "Neuralizer: General neuroimage analysis without re-training." 
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
- [Implementation of the Neuralizer architecture](https://github.com/SteffenCzolbe/neuralizer)

- Negrini, Alessio, and Reiß, Simon. "Conquering the Retina: Bringing Visual in-Context Learning to OCT." arXiv preprint arXiv:2506.15200. 2025.
- [Training codebase adapted from Negrini et al.](https://github.com/negralessio/thesis-visual-in-context-learning)
 
_(For datasets used within this project, see [Datasets Documentation](docs/datasets.md))_

# Cite this and influential prior work
