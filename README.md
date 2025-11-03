# 🧠 Enhancing Tourette's Syndrome Classification via CSTC Circuit Segmentation and CNNs  

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Paper Status](https://img.shields.io/badge/Paper-Under_Review-blue)](#)
[![Made with Love](https://img.shields.io/badge/Made%20with❤️-by%20Murilo%20C.%20de%20Barros-red)](#)

**Murilo C. de Barros · Kaue T. N. Duarte · Chia-Jui Hsu · Wang-Tso Lee · Marco A. G. de Carvalho**

---

## 🎯 Overview  

This repository contains the **trained models** and **scripts** used in our paper:  
> **“Enhancing Tourette's Syndrome Classification via Cortico-Striatal-Thalamic-Cortical Circuit Segmentation and Convolutional Neural Networks”**

Our work focuses on leveraging **deep learning** (CNNs) to classify **Tourette’s Syndrome**, exploring the **CSTC circuit** through multiple segmentation approaches — *Whole Brain*, *SLANT*, and *DKT*.

---

## 🗂️ Directory Structure  
- `models/`
  - Contains the trained models for three different architectures: VGG16, VGG19, and ResNet50.
  - Each architecture has three different styles: `WholeBrain`, `Slant`, and `DKT`.
  - The models are saved in Keras format with the following files:
    - `keras_metadata.pb`
    - `saved_model.pb`
    - `variables/`
  - Due to file size constraints on GitHub, our models can be downloaded using the link: https://drive.google.com/drive/folders/1CpbNOLKkAmGd4gPskwe4fPs8nnvwMeFv?usp=sharing 
- `scripts/`
  - Contains the scripts used during training. We employed the python scripts using SLURMs. The models were trained using Weights and Biases, please refer to their official website for further information
    - `run_monai.py`: This scripts run our codes without W&B.
    - `run_monai_wandb.py`: This scripts run our codes with W&B.
    - `gpu_batch_2D.sh`: Main script responsible for running the python codes.

## Models
The models provided in this repository are named as follows:
```bash
.
├── models/
│   ├── WholeBrain_VGG16/
│   ├── WholeBrain_VGG19/
│   ├── WholeBrain_ResNet50/
│   ├── Slant_VGG16/
│   ├── Slant_VGG19/
│   ├── Slant_ResNet50/
│   ├── DKT_VGG16/
│   ├── DKT_VGG19/
│   └── DKT_ResNet50/
│
└── scripts/
    ├── run_monai.py
    ├── run_monai_wandb.py
    └── gpu_batch_2D.sh
```
### 🔗 Model Access  
Due to GitHub storage limits, trained models can be downloaded here:  
👉 [**Google Drive Folder**](https://drive.google.com/drive/folders/1CpbNOLKkAmGd4gPskwe4fPs8nnvwMeFv?usp=sharing)

---

## 🧩 Model Architectures  

Each model combines one of three CNN architectures with one of three segmentation strategies:

| Architecture | Whole Brain | SLANT | DKT |
|---------------|-------------|--------|------|
| **VGG16**     | ✅ | ✅ | ✅ |
| **VGG19**     | ✅ | ✅ | ✅ |
| **ResNet50**  | ✅ | ✅ | ✅ |

**Naming convention:**

## Usage
To use these models for transfer learning or inference, please follow the instructions below.

### Loading Models
You can load the models using Keras. Below is a Python script to load a model and use it for transfer learning.

```python
import tensorflow as tf
from tensorflow.keras.models import load_model

def load_keras_model(model_name):
    model_path = f'models/{model_name}'
    model = tf.keras.models.load_model(model_path)
    return model

def print_model_summary(model):
    model.summary()

def transfer_learning_example(model):
    # Freezing the layers except the last 4 layers
    for layer in model.layers[:-4]:
        layer.trainable = False
    
    # Adding custom layers for transfer learning
    x = model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    # Creating the final model
    model_final = tf.keras.models.Model(inputs=model.input, outputs=predictions)
    
    # Compile the model
    model_final.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model_final

if __name__ == "__main__":
    model_name = 'vgg16_whole_brain'  # Example model name
    model = load_keras_model(model_name)
    print_model_summary(model)
    model_final = transfer_learning_example(model)
    print_model_summary(model_final)
```

## Running the Script
Ensure you have TensorFlow installed: pip install tensorflow

Run the script: python load_model.py

## License

This project is licensed under the MIT License — you’re free to use, modify, and distribute with attribution.

## Citation

The citation will be updated once this paper is published. Our citation template is:

```vbnet
@article{TS_costa2024,
  title={Enhancing Tourette's Syndrome Classification via Cortico-Striatal-Thalamic-Cortical Circuit Segmentation and Convolutional Neural Networks},
  author={Murilo C. de Barros, Kaue T. N. Duarte, Chia-Jui Hsu, Wang-Tso Lee, Marco A. G. de Carvalho},
  journal={IEEE Latin America},
  year={2024},
  volume={NA},
  pages={NA},
  doi={NA}
}
```
