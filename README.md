# A Real-Time Online Learning Framework for Joint 3D Reconstruction and Semantic Segmentation for Indoor Scene.

This is the official implementation of the RAL submission [**A Real-Time Online Learning Framework for Joint 3D Reconstruction and Semantic Segmentation for Indoor Scene**](link). 

If you find our code or paper useful, please consider citing

    @InProceedings{
    }

Prior to using the source code in a commercial application, please contact the authors.

# Author. 
Davide Menini (MS, ETH Zurich). This project is completed by Mr. Davide Menini for his Master Thesis.

This work was funded by Focused Research Award from Google.

## Usage

Below you find instructions on how to use our framework as a 3D reconstruction and semantic segmentation pipeline for training and evaluation.

### Data Preparation
The models are trained on our own dataset generated from Replica, from which an example scene is provided, and on ScanNet.
To get our data, please contact the corresponding author (Suryansh Kumar, k.sur46@gmail.com).

### Installation

To install our framework, you can use a conda environment.

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
For training out model with different hyperparameters, losses or data, you can simply modify the config file. 

### Testing
We provide pretrained models for the multimodal AdapNet with RGB and ToF input modalities, and 2 version of the fusion architecture, one optimized for accuracy (Fusion Network v3 with semantics enabled, 256x256 input frames, running across CPU and GPU), and one optimized for speed (Fusion Network v3 without semantics (=v2 without semantics), 128x128 input frames, running entirely on GPU).

**Test Segmentation Network**
<pre><code>python test_segmentation.py --config configs/segmentation/replica_multi.yaml
</code></pre>

**Test Fusion Network**
<pre><code>python test_fusion.py --config configs/fusion/replica_accuracy.yaml
python test_fusion.py --config configs/fusion/replica_speed.yaml
</code></pre>

### Evaluation
TODO
