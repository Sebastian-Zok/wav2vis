# wav2vis

**AI Model for Audio to Viseme Translation**

This project implements an AI model for converting audio speech into viseme sequences, enabling realistic lip-sync animation from speech audio for cartoon animations.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage Pipeline](#usage-pipeline)
- [Model Architecture](#model-architecture)
- [Requirements](#requirements)

## 🎯 Project Overview

This project uses a teacher-student distillation approach to create an efficient audio-to-viseme translation model:

- **Teacher Model**: A larger, more accurate phoneme recognition model
- **Student Model**: A smaller, optimized model for real-time inference
- **Dataset**: Mozilla Common Voice corpus with forced alignment for phoneme-to-viseme mapping

## 📁 Project Structure

### Current Structure

```
commonVoiceDataset/
├── 📂 data/                    # Original Mozilla Common Voice data and generated feature files
│   ├── features/
│   ├── labels/
│   ├── processed/
│   └── statistics/
├── 📓 notebooks/               # Jupyter notebooks
├── 🐍 src/                     # Source code modules
│   ├── __init__.py
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── convert_to_mp3.py
│   │   ├── get_gentle_alignments.py
│   │   └── gentle_to_phn.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── teacher_model.py
│   │   └── student_model.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── feature_extraction.py
│   │   ├── audio_utils.py
│   │   └── visualization.py
│   └── config/
│       ├── __init__.py
│       └── config.py
├── requirements.txt
└── README.md
```

## 🛠️ Setup & Installation

### Prerequisites

- Python 3.8+
- Docker (for Gentle forced aligner)
- CUDA-capable GPU (recommended for training)

### Quick Start

```bash
# Create virtual environment
py -3.11 -m venv tf-env# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Gentle Forced Aligner Setup

```bash
# Pull and run Gentle Docker container
docker pull lowerquality/gentle
docker run -p 8765:8765 lowerquality/gentle
```

## 🚀 Usage Pipeline

### 1. 🔊 Audio Alignment

```python
# Extract phoneme alignments using Gentle
python get_gentle_alignments.py
python gentle_to_phn.py
```

### 2. 📥 Data Preparation

```python
# Run DataPreperation.ipynb
# - Removes OOV (Out-of-Vocabulary) sentences
# - Cleans and filters dataset
```

### 3. 🎛️ Feature Extraction

```python
# Run feature_extraction.ipynb
# - Extract MFCC features
# - Generate spectrograms
# - Create feature vectors for training
```

### 4. 📊 Data Analysis & Normalization

```python
# Run Analyze.ipynb
# - Analyze feature distributions
# - Compute normalization statistics
# - Prepare data for training
```

### 5. 🎓 Teacher Model Training

```python
# Run Training.ipynb
# - Train the teacher phoneme recognition model
# - Generate teacher logits for distillation
```

### 6. 🎒 Student Model Distillation

```python
# Run Distillation Script.ipynb
# - Train compressed student model
# - Use teacher knowledge distillation
# - Optimize for real-time inference
```

### 7. 🧪 Model Evaluation

```python
# Run Test.ipynb
# - Evaluate model performance
# - Test on validation set
# - Generate performance metrics
```

## 🏗️ Model Architecture

### Teacher Model

- **Input**: MFCC features (13 coefficients + derivatives)
- **Architecture**: Deep CNN-LSTM network
- **Output**: Phoneme probabilities
- **Purpose**: High-accuracy phoneme recognition

### Student Model

- **Input**: Same MFCC features
- **Architecture**: Lightweight CNN network
- **Output**: Viseme classifications
- **Purpose**: Real-time audio-to-viseme translation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request
