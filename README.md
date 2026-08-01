# Universal Image Restoration Pipeline 🖼️✨

A comprehensive image restoration pipeline integrating state-of-the-art Deep Learning models: **NAFNet**, **Restormer**, and **MPRNet**. This framework enables high-quality restoration of degraded images, addressing common artifacts such as motion blur, defocus blur, camera noise, low-light degradation, and rain.

---

## 🌟 Features & Supported Models

This project brings together three powerful architectures for image restoration:

| Model | Primary Architecture Type | Restorations Covered in Pipeline |
| :--- | :--- | :--- |
| **NAFNet** *(Non-linear Activation Free Network)* | CNN (Simplified / Activation-Free) | GoPro Motion Deblurring, General Denoising |
| **Restormer** *(Restoration Transformer)* | Transformer (Self-Attention over Channels) | Motion Deblurring, Defocus Deblurring |
| **MPRNet** *(Multi-Stage Progressive Network)* | CNN (Multi-Stage Encoder-Decoder) | Deblurring, Denoising, Deraining |

---

## 📁 Repository Structure

```markdown
├── checkpoints/             # Directory for model checkpoints (*.pth) - Excluded from Git
├── input/                   # Sample degraded images (blur, noise, defocus, etc.)
├── models/                  # Subdirectories containing model architecture definitions
│   ├── mprnet/              # MPRNet codebase
│   ├── nafnet/              # NAFNet codebase
│   └── restormer/           # Restormer codebase
├── output/                  # Restored image outputs - Excluded from Git
├── comparisons/             # Generated comparison matrices & diffmaps - Excluded from Git
├── pipeline.ipynb           # Main Jupyter Notebook executing the full restoration pipeline
├── requirements.txt         # Python dependencies
├── test_nafnet_load.py      # Script to verify model structure and checkpoint compatibility
└── .gitignore               # Configured Git ignore patterns
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Sid-Dev-alt/Image_Restoration.git
cd Image_Restoration
```

### 2. Install Dependencies
Make sure you have [PyTorch](https://pytorch.org/) installed matching your CUDA version. Then install the requirements:
```bash
pip install -r requirements.txt
```

### 3. Download Pretrained Weights
To run the models, download the pre-trained checkpoints and place them in the `checkpoints/` directory. The pipeline expects the following files:
* `checkpoints/nafnet_gopro.pth`
* `checkpoints/restormer_motion.pth`
* `checkpoints/restormer_defocus.pth`
* `checkpoints/mprnet_deblurring.pth`

> [!NOTE]
> Ensure the folder structure is exactly `checkpoints/<filename>.pth` so the pipeline can find and load the weights correctly.

---

## 🚀 How to Run

### Main Pipeline
Open and execute the cells in [`pipeline.ipynb`](file:///c:/Users/sidk4/work_Docs/UniversalImageRestoration/pipeline.ipynb). 

The notebook handles:
1. Path injection for local model directories (`models/nafnet`, `models/restormer`, `models/mprnet`).
2. Initialization of model configurations and weights.
3. **Tiled Inference**: Handles large images efficiently without causing Out-Of-Memory (OOM) errors on your GPU.
4. Saving results to the `output/` directory.
5. Generating visual comparison maps and diffmaps under the `comparisons/` folder.

### Testing Checkpoint Loading
To quickly check if your PyTorch environment is correctly loading the model weights on CPU/GPU, you can run:
```bash
python test_nafnet_load.py
```

---

## 📈 Example Walkthrough
1. Place a degraded image in the `input/` folder (e.g., `input/blurry_motion.jpg`).
2. Run the corresponding model block inside `pipeline.ipynb`.
3. Check the `output/` folder for the restored result, and the `comparisons/` folder to analyze the pixel differences between input and output.