# Enhancing Tourette's Syndrome Classification via Cortico-Striatal-Thalamic-Cortical Circuit Segmentation and Convolutional Neural Networks

## Purpose
This repository contains the trained models and scripts used for our paper titled "Enhancing Tourette's Syndrome Classification via Cortico-Striatal-Thalamic-Cortical Circuit Segmentation and Convolutional Neural Networks". Our work focuses on utilizing convolutional neural networks (CNNs) to classify Tourette's Syndrome using various brain segmentation techniques.

## Directory Structure
- `models/`
  - Contains the trained models for three different architectures: VGG16, VGG19, and ResNet50.
  - Each architecture has three different styles: `WholeBrain`, `Slant`, and `DKT`.
  - The models are saved in Keras format with the following files:
    - `keras_metadata.pb`
    - `saved_model.pb`
    - `variables/`
- scripts/
  - Contains the scripts used during training. We employed the python scripts using SLURMs. The models were trained using Weights and Biases, please refer to their official website for further information
    - `run_monai.py`: This scripts run our codes without W&B.
    - `run_monai_wandb.py`: This scripts run our codes with W&B.
    - `gpu_batch_2D.sh`: Main script responsible for running the python codes.
## Models
The models provided in this repository are named as follows:
- `WholeBrain_VGG16`
- `WholeBrain_VGG19`
- `WholeBrain_ResNet50`
- `Slant_VGG16`
- `Slant_VGG19`
- `Slant_ResNet50`
- `DKT_VGG16`
- `DKT_VGG19`
- `DKT_ResNet50`

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

This repository is licensed under the MIT License.

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