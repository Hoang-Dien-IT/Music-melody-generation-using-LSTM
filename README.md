# Beethoven Melody Generator Using RNN

This project is a deep learning-based melody generation system that uses Recurrent Neural Networks (RNNs) to generate music inspired by Beethoven's compositions. The system processes MIDI files of Beethoven's works, extracts musical features, trains a model, and generates new melodies in MIDI format.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [File Structure](#file-structure)
- [Future Work](#future-work)
- [License](#license)

---

## Project Overview

The goal of this project is to generate new Beethoven-style melodies using deep learning techniques. The system uses MIDI files as input, extracts notes and chords, and trains an RNN-based model to predict the next note in a sequence. The trained model can then generate new melodies by sampling from the learned distribution.

---

## Features

- **MIDI File Processing**: Extracts notes and chords from Beethoven's MIDI files.
- **Data Preprocessing**: Handles rare notes, builds mappings for notes to indices, and prepares sequences for training.
- **Model Training**: Implements multiple RNN architectures (LSTM and CuDNNLSTM) for melody generation.
- **Melody Generation**: Generates new melodies in MIDI format.
- **Visualization**: Plots training loss and accuracy over epochs.
- **Evaluation**: Evaluates the model's performance using metrics like accuracy, precision, recall, and F1-score.

---

## Dataset

The dataset consists of MIDI files of Beethoven's compositions stored in the `data/beeth/` directory. These files include famous works such as:

- "Für Elise"
- "Pathetique Sonata"
- "Waldstein Sonata"
- "Appassionata Sonata"
- "Hammerklavier Sonata"

---

## Model Architecture

The project includes three RNN-based architectures:

1. **Model 1**: A single-layer LSTM with 256 units.
2. **Model 2**: A multi-layer CuDNNLSTM with 512 and 256 units.
3. **Model 3**: A multi-layer LSTM with 512, 256, and 128 units.

All models use dropout for regularization and the Adamax optimizer for training.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- TensorFlow 2.x
- Music21 library for MIDI processing
- Other dependencies listed in `requirements.txt`

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/github_RNN.git
   cd github_RNN
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have a GPU-enabled environment for faster training (optional).

---

## Usage

### 1. Preprocessing MIDI Files
Run the script to process MIDI files and extract notes:
```bash
python create_model/Creat_model.py
```

### 2. Training the Model
Train the model using the preprocessed data:
```bash
python create_model/Creat_model.py
```

### 3. Generating Melodies
Generate new melodies using the trained model:
```bash
python create_model/Test_model.py
```

### 4. Output
Generated melodies are saved as MIDI files in the `create_model/` directory:
- `Beeth_Melody_Generated_model2.mid`
- `Beeth_Melody_Snippet_model2.mid`

---

## Results

### Training Metrics
- **Accuracy**: Achieved up to 90% accuracy on the training set.
- **Loss**: Loss decreased steadily over epochs.

### Generated Melodies
The generated melodies resemble Beethoven's style and are saved as MIDI files for playback.

### Evaluation Metrics
- **Accuracy**: 85-90%
- **Precision**: ~0.88
- **Recall**: ~0.87
- **F1-Score**: ~0.87

---

## File Structure

```
github_RNN/
│
├── data/
│   └── beeth/
│       ├── appass_1.mid
│       ├── appass_2.mid
│       ├── ...
│
├── create_model/
│   ├── Creat_model.py
│   ├── Test_model.py
│   ├── Beeth_Melody_Generated_model2.mid
│   ├── Beeth_Melody_Snippet_model2.mid
│   └── Bao_Cao_RNN_FINAL.pptx
│
├── README.md
└── requirements.txt
```

---

## Future Work

- **Expand Dataset**: Include more compositions from other classical composers.
- **Improve Model**: Experiment with Transformer-based architectures for better results.
- **Interactive Interface**: Develop a web-based interface for real-time melody generation.
- **Music Theory Integration**: Incorporate music theory rules to improve the quality of generated melodies.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
