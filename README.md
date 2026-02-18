# TON_IoT Network Intrusion Detection System

A machine learning-based network intrusion detection system using the TON_IoT dataset with GAN-based data augmentation and bio-inspired feature selection.

## 📋 Overview

This project implements a complete pipeline for detecting network attacks using the TON_IoT dataset. It addresses class imbalance through GAN-based data augmentation, applies bio-inspired feature selection (Genetic Algorithm + RFE), and evaluates multiple machine learning models including Decision Trees, XGBoost, LightGBM, and Deep Neural Networks.

## ✨ Features

- **Data Preprocessing**: Automatic handling of missing values, one-hot encoding, and normalization
- **GAN-Based Data Augmentation**: Generates synthetic samples for minority attack classes
- **Bio-Inspired Feature Selection**: Two-stage feature selection using Genetic Algorithm and Recursive Feature Elimination (RFE)
- **Multiple ML Models**: Comparison of Decision Tree, XGBoost, LightGBM, and DNN classifiers
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, and training time metrics
- **Visualization**: ROC curves and confusion matrices

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Required Libraries

```bash
pip install pandas numpy scikit-learn tensorflow xgboost lightgbm sklearn-genetic-opt matplotlib seaborn
```

### Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
tensorflow>=2.6.0
xgboost>=1.5.0
lightgbm>=3.3.0
sklearn-genetic-opt>=0.9.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## 📁 Project Structure

```
TON_IoT Datasets/
├── main.py                      # Initial dataset inspection
├── preprocess.py                # Data preprocessing pipeline
├── check_balance.py             # Check class distribution
├── train_gan.py                 # GAN training for data augmentation
├── feature_selection.py         # Bio-inspired feature selection
├── final_evaluation.py          # Model training and evaluation
├── roc_curve.py                 # ROC curve generation
├── visualize_results.py         # Confusion matrix visualization
├── train_test_network.csv       # Original dataset (input)
├── processed_network_data.csv   # Preprocessed data
├── balanced_network_data.csv    # Augmented balanced dataset
├── final_selected_data.csv      # Feature-selected dataset
├── final_model_comparison.csv   # Model evaluation results
└── README.md                    # This file
```

## 🚀 Usage

### Step 1: Dataset Inspection

Inspect the raw dataset to understand its structure:

```bash
python main.py
```

### Step 2: Data Preprocessing

Clean, encode, and normalize the data:

```bash
python preprocess.py
```

This step:
- Removes high-cardinality columns (IPs, ports, URIs)
- Handles missing values
- Performs one-hot encoding for categorical features
- Applies min-max normalization

### Step 3: Check Class Balance

Check the distribution of attack types:

```bash
python check_balance.py
```

### Step 4: GAN-Based Data Augmentation

Generate synthetic samples for minority classes (e.g., MITM attacks):

```bash
python train_gan.py
```

**Configuration in train_gan.py:**
- `TARGET_CLASS`: Minority class to augment (default: 'mitm')
- `TARGET_COUNT`: Desired number of samples (default: 20000)
- `EPOCHS`: Training iterations (default: 2000)
- `LATENT_DIM`: Noise vector dimension (default: 100)

### Step 5: Feature Selection

Apply bio-inspired feature selection:

```bash
python feature_selection.py
```

This uses a two-stage approach:
1. **Genetic Algorithm**: Evolutionary search for optimal feature subsets
2. **Recursive Feature Elimination (RFE)**: Refines the selection

### Step 6: Model Training and Evaluation

Train and compare multiple ML models:

```bash
python final_evaluation.py
```

Models evaluated:
- Decision Tree
- XGBoost
- LightGBM
- Deep Neural Network (DNN)

Results are saved to `final_model_comparison.csv`.

### Step 7: Visualization

Generate ROC curves:

```bash
python roc_curve.py
```

Generate confusion matrix:

```bash
python visualize_results.py
```

## 📊 Dataset

**Source**: TON_IoT Network Dataset

**File**: `train_test_network.csv`

**Attack Types**:
- Normal traffic
- DDoS attacks
- MITM (Man-in-the-Middle) attacks
- Injection attacks
- Scanning attacks
- Password attacks
- Backdoor attacks
- XSS (Cross-Site Scripting) attacks
- Ransomware attacks

**Features**: Network traffic features including connection metadata, protocol information, and statistical features.

## 🎯 Model Performance

After running the complete pipeline, you can view model comparison results in `final_model_comparison.csv`, which includes:
- Accuracy
- Precision
- Recall
- F1-Score
- Training Time

Typical results (after preprocessing, augmentation, and feature selection):
- **XGBoost**: Highest accuracy (~99%+)
- **LightGBM**: Fastest training time with high accuracy
- **Decision Tree**: Fast and interpretable baseline
- **DNN**: Strong performance with more training time

## 🧬 Bio-Inspired Feature Selection

The feature selection module uses evolutionary algorithms to find the optimal feature subset:

**Genetic Algorithm Parameters**:
- Population size: 10
- Generations: 5
- Crossover probability: 0.8
- Mutation probability: 0.1

**RFE Configuration**:
- Based on Decision Tree estimator
- Selects top 20 features by default

## 🔧 Configuration

### GAN Training (`train_gan.py`)

```python
LATENT_DIM = 100       # Size of random noise vector
BATCH_SIZE = 64        # Training batch size
EPOCHS = 2000          # Number of training epochs
TARGET_CLASS = 'mitm'  # Minority class to augment
TARGET_COUNT = 20000   # Desired sample count
```

### Feature Selection (`feature_selection.py`)

```python
SUBSET_SIZE = 10000    # Sample size for faster selection
```

## 📈 Output Files

- `processed_network_data.csv`: Preprocessed and normalized data
- `balanced_network_data.csv`: Data after GAN augmentation
- `final_selected_data.csv`: Data after feature selection
- `final_model_comparison.csv`: Model evaluation metrics
- `confusion_matrix.png`: Confusion matrix visualization
- `roc_curves.png`: ROC curves for all models
- `gan_generator.h5`: Trained GAN generator model
- `gan_loss_history.csv`: GAN training loss history

## 🐛 Troubleshooting

### Issue: "File not found" errors
**Solution**: Ensure you run the scripts in order (preprocess.py → train_gan.py → feature_selection.py → final_evaluation.py)

### Issue: Memory errors during training
**Solution**: Reduce `SUBSET_SIZE` in feature_selection.py or `BATCH_SIZE` in train_gan.py

### Issue: TclError when generating plots
**Solution**: The scripts already use `matplotlib.use('Agg')` to save plots without display

### Issue: GAN not converging
**Solution**: Increase `EPOCHS` in train_gan.py (try 3000-5000 for better quality)

## 📝 Pipeline Summary

```
1. main.py           → Inspect raw data
2. preprocess.py     → Clean and normalize
3. check_balance.py  → Check class distribution
4. train_gan.py      → Generate synthetic samples
5. feature_selection.py → Select optimal features
6. final_evaluation.py → Train and evaluate models
7. roc_curve.py      → Generate ROC visualization
8. visualize_results.py → Generate confusion matrix
```

## 🔬 Methodology

This project implements a research-based approach to network intrusion detection:

1. **Data Preprocessing**: Handle real-world data quality issues
2. **Class Balancing**: Use GANs to address the minority class problem
3. **Feature Engineering**: Apply bio-inspired algorithms for optimal feature selection
4. **Model Comparison**: Evaluate multiple state-of-the-art classifiers
5. **Performance Analysis**: Comprehensive metrics and visualizations

## 📚 References

- TON_IoT Dataset: [https://research.unsw.edu.au/projects/toniot-datasets](https://research.unsw.edu.au/projects/toniot-datasets)
- GAN Architecture: Based on vanilla GAN with LeakyReLU and BatchNormalization
- Feature Selection: sklearn-genetic-opt library

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional ML models (Random Forest, SVM, etc.)
- Hyperparameter tuning with Grid/Random Search
- Real-time detection capabilities
- Model interpretability (SHAP, LIME)
- Multi-class attack classification

## 📄 License

This project is provided as-is for educational and research purposes.

## 👤 Author

Network Security Research Project

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the configuration parameters
3. Ensure all dependencies are installed correctly

---

**Note**: The complete pipeline may take 30-60 minutes to run depending on your hardware (especially GAN training with 2000 epochs).
