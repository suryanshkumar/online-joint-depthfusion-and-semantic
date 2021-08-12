# A Real-Time Online Learning Framework for Joint 3D Reconstruction and Semantic Segmentation for Indoor Scene.

This is the official implementation of the RAL submission [**A Real-Time Online Learning Framework for Joint 3D Reconstruction and Semantic Segmentation for Indoor Scene**](link).

<p align="justify">
<b>Abstract:</b> This paper presents a real-time online vision framework to jointly recover an indoor scene's 3D structure and semantic label. Given noisy depth maps, a camera trajectory, and 2D semantic labels at train time, the proposed neural network learns to fuse the depth over frames with suitable semantic labels in the scene space. Our approach exploits the joint volumetric representation of the depth and semantics in the scene feature space to solve this task. For a compelling online fusion of the semantic labels and geometry in real-time, we introduce an efficient vortex pooling block while dropping the routing network in online depth fusion to preserve high-frequency surface details. We show that the context information provided by the semantics of the scene helps the depth fusion network learn noise-resistant features. Not only that, it helps overcome the shortcomings of the current online depth fusion method in dealing with thin object structures, thickening artifacts, and false surfaces. Experimental evaluation on the Replica dataset shows that our approach can perform depth fusion at 37, 10 frames per second with an average reconstruction F-score of 88%, and 91%, respectively, depending on the depth map resolution. Moreover, our model shows an average IoU score of 0.515 on the ScanNet 3D semantic benchmark leaderboard.
</p>

If you find our code or paper useful, please consider citing
<tr>
<td>
[1] <strong>A Real-Time Online Learning Framework for Joint 3D Reconstruction and Semantic Segmentation of Indoor Scenes </strong><br />
Davide Menini and Suryansh Kumar and Martin R. Oswald and Erik Sandstrom and Cristian Sminchisescu and Luc Van Gool<br />
arXiv 2021<br />
[<a href="https://arxiv.org/abs/2108.05246" target="_blank">pdf</a>]  [<a href="https://github.com/suryanshkumar/online-joint-depthfusion-and-semantic" target="_blank">official code</a>] <br />
</td>
</tr>
<br/>

<b>Bibtex</b><br />
```
@article{menini2021realtime,
title={A Real-Time Online Learning Framework for Joint 3D Reconstruction and Semantic Segmentation of Indoor Scenes},
author={Davide Menini and Suryansh Kumar and Martin R. Oswald and Erik Sandstrom and Cristian Sminchisescu and Luc Van Gool},
year={2021},
eprint={2108.05246},
archivePrefix={arXiv},
primaryClass={cs.CV}
}
```

# Acknowledgment
This work was funded by Focused Research Award from Google.
Authors thank Silvan Weder (CVG, ETH Zurich) for useful discussion.
This project is completed by Mr. Davide Menini for his Master Thesis.

Some parts of the code are modified from the original RoutedFusion [2] implementation.
Prior to using the source code for a commercial application, please contact the authors.


<br/>
<b>Related Work</b>
<tr>
<td>
[2] RoutedFusion: Learning Real-time Depth Map Fusion<br />
Silvan Weder, Johannes L. Schönberger, Marc Pollefeys, Martin R. Oswald<br />
CVPR 2020<br />
[<a href="https://arxiv.org/abs/2001.04388" target="_blank">pdf</a>]  [<a href="https://github.com/weders/RoutedFusion" target="_blank">official code</a>] <br />
</td>
</tr>

## Usage

Below you find instructions on how to use our framework as a 3D reconstruction and semantic segmentation pipeline for training and testing.

### Data Preparation
Our model is trained on dataset generated from Replica and ScanNet. For training, we processed non-watertight mesh to watertight mesh. To get access processed example scene data, visit [project webpage](https://suryanshkumar.github.io/online-joint-depthfusion-and-semantic_project_page/)

Replica dataset [article](https://arxiv.org/abs/1906.05797):
```
@article{replica19arxiv,
  title =   {The {R}eplica Dataset: A Digital Replica of Indoor Spaces},
  author =  {Julian Straub and Thomas Whelan and Lingni Ma and Yufan Chen and Erik Wijmans and Simon Green and Jakob J. Engel and Raul Mur-Artal and Carl Ren and Shobhit Verma and Anton Clarkson and Mingfei Yan and Brian Budge and Yajie Yan and Xiaqing Pan and June Yon and Yuyang Zou and Kimberly Leon and Nigel Carter and Jesus Briales and  Tyler Gillingham and  Elias Mueggler and Luis Pesqueira and Manolis Savva and Dhruv Batra and Hauke M. Strasdat and Renzo De Nardi and Michael Goesele and Steven Lovegrove and Richard Newcombe },
  journal = {arXiv preprint arXiv:1906.05797},
  year =    {2019}
}
```

ScanNet dataset [article](https://arxiv.org/abs/1702.04405)
```
@inproceedings{dai2017scannet,
  title={Scannet: Richly-annotated 3d reconstructions of indoor scenes},
  author={Dai, Angela and Chang, Angel X and Savva, Manolis and Halber, Maciej and Funkhouser, Thomas and Nie{\ss}ner, Matthias},
  booktitle={IEEE, CVPR},
  pages={5828--5839},
  year={2017}
}
```

### Installation

To install our framework, you can use a conda environment with Python 3.7 and PyTorch 1.4.0.

**Clone the repo**

<pre><code>git clone https://github.com/suryanshkumar/online-joint-depthfusion-and-semantic.git
git submodule update --init --recursive
</code></pre>

**Create the Anaconda environment**
<pre><code>conda env create -f environment.yml
conda activate segfusion
</code></pre>

You may have to manually install dependencies in *deps/* by cd-ing into the package with the *setup.py* file and then running:
<pre><code>pip install .
</code></pre>

You can find some prtrained model on the [project webpage](https://suryanshkumar.github.io/online-joint-depthfusion-and-semantic_project_page/). Download and unzip *workspace*, then place it inside the main project folder.

We provide an example scene to run the tests, which again can be found on the [project webpage](https://suryanshkumar.github.io/online-joint-depthfusion-and-semantic_project_page/). In order to use it in the code, assign the path of data root directory (*replica*) to the *root* key in the yaml configuration file (*configs/replica*) and modify the list files in *lists/replica* to only include the downloaded scene (*example.txt* is already available).

### Training
Once the environment is ready, you can first train the segmentation network (2 stages) and then the fusion network (either v1, v2 or v3, depending on the available resources).

**Train Segmentation Network**
<pre><code>python train_segmentation.py --config configs/segmentation/replica_rgb.yaml
python train_segmentation.py --config configs/segmentation/replica_depth.yaml
python train_segmentation.py --config configs/segmentation/replica_multi.yaml
</code></pre>

**Train Fusion Network**
<pre><code>python train_fusion.py --config configs/fusion/replica_accuracy.yaml
</code></pre>

**Change Data Configuration**
For training out model with different hyperparameters, losses or data, you can simply modify the configuration file. 

### Testing
We provide pretrained models for the multimodal AdapNet with RGB and ToF input modalities, and 2 version of the fusion architecture, one optimized for accuracy (Fusion Network v3 with semantics enabled, 256x256 input frames, running across CPU and GPU), and one optimized for speed (Fusion Network v3 without semantics (=v2 without semantics), 128x128 input frames, running entirely on GPU).

**Test Segmentation Network**
<pre><code>python test_segmentation.py --config configs/segmentation/replica_multi.yaml
</code></pre>

**Test Fusion Network**
<pre><code>python test_fusion.py --config configs/fusion/replica_accuracy.yaml
python test_fusion.py --config configs/fusion/replica_speed.yaml
</code></pre>
