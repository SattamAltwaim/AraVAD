# AraVAD: Arabic Valence-Arousal-Dominance Sentiment Analysis

## Overview

AraVAD is a Natural Language Processing project developed for the CCAI-413 course at the University of Jeddah. This project focuses on providing continuous sentiment analysis for Arabic text using the Valence-Arousal-Dominance (VAD) model, which offers a more nuanced approach to emotion analysis compared to traditional binary sentiment classification.

## VAD Model

The Valence-Arousal-Dominance model represents emotions in a three-dimensional space:

- **Valence**: The pleasantness of an emotion (positive vs. negative)
- **Arousal**: The intensity or activation level of an emotion (calm vs. excited)
- **Dominance**: The sense of control or power in an emotion (submissive vs. dominant)

This continuous approach allows for more sophisticated emotion analysis that captures the complexity of human emotional states.

## Datasets

The project utilizes the NRC-VAD (National Research Council - Valence-Arousal-Dominance) lexicon datasets:

### Available Datasets
- **NRC-VAD-Lexicon.zip**: Original NRC-VAD lexicon
- **NRC-VAD-Lexicon-v2.1.zip**: Updated version 2.1 of the NRC-VAD lexicon

These datasets are located in the `Datasets/Raw/` directory and contain Arabic word annotations with VAD scores.

### Dataset Credits

The NRC-VAD lexicon datasets are credited to **Dr. Saif M. Mohammad**, Principal Research Scientist at the National Research Council Canada (NRC). 

- **Researcher Profile**: [Dr. Saif M. Mohammad](https://saifmohammad.com)
- **Institution**: National Research Council Canada
- **Research Focus**: Computational linguistics, emotion analysis, and sentiment analysis

## Project Structure

```
AraVAD/
├── README.md
└── Datasets/
    └── Raw/
        ├── NRC-VAD-Lexicon.zip
        └── NRC-VAD-Lexicon-v2.1.zip
```

## Getting Started

### Prerequisites

- Python 3.7+
- Required Python packages (to be specified in requirements.txt)
- Access to Arabic text data for analysis

### Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd AraVAD
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Extract the dataset files from the `Datasets/Raw/` directory

### Usage

[Usage instructions will be added as the project develops]

## Research Applications

This project can be applied to various domains including:

- Social media sentiment analysis
- Customer feedback analysis
- Mental health monitoring
- Content recommendation systems
- Cross-cultural emotion studies

## Contributing

This project is developed as part of academic coursework. Contributions and suggestions are welcome.

## Acknowledgments

- Dr. Saif M. Mohammad and the National Research Council Canada for providing the NRC-VAD lexicon



