"""
Convert Keras models to TensorFlow Lite format.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import List


INPUT_SHAPE = (2, 381, 40)
NUM_CLASSES = 9
FEATURE_INPUT_DIM = 25
FEATURE_NUM_CLASSES = 5
FEATURE_BN_EPSILON = 1e-3


def build_crnn_model() -> keras.Model:
    inputs = keras.Input(shape=INPUT_SHAPE)
    x = keras.layers.Conv2D(32, (5, 5), padding="same", activation="relu", data_format="channels_first")(inputs)
    x = keras.layers.MaxPool2D(pool_size=(1, 3), data_format="channels_first")(x)
    x = keras.layers.BatchNormalization(axis=1)(x)
    x = keras.layers.Dropout(0.0)(x)
    x = keras.layers.Permute((2, 1, 3))(x)
    x = keras.layers.Reshape((INPUT_SHAPE[1], -1))(x)
    x = keras.layers.LSTM(32, return_sequences=True, dropout=0.0, recurrent_dropout=0.0)(x)
    x = keras.layers.TimeDistributed(keras.layers.Dense(32, activation="relu"))(x)
    x = keras.layers.Lambda(lambda t: tf.reduce_mean(t, axis=-1))(x)
    outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def build_feature_model() -> keras.Model:
    inputs = keras.Input(shape=(FEATURE_INPUT_DIM,))
    x = keras.layers.BatchNormalization()(inputs)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.0)(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.0)(x)
    outputs = keras.layers.Dense(FEATURE_NUM_CLASSES, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def load_feature_weights_npz(weights_path: Path) -> List[np.ndarray]:
    data = np.load(weights_path, allow_pickle=True)
    ordered = sorted(data.files, key=lambda name: int(name.split("_")[1]))
    return [data[name] for name in ordered]


def build_feature_fused_model(weights: List[np.ndarray]) -> keras.Model:
    gamma, beta, mean, var, W1, b1, W2, b2, W3, b3 = weights
    scale = gamma / np.sqrt(var + FEATURE_BN_EPSILON)
    offset = beta - mean * scale

    W1_prime = W1 * scale[:, np.newaxis]
    b1_prime = offset @ W1 + b1

    inputs = keras.Input(shape=(FEATURE_INPUT_DIM,))
    x = keras.layers.Dense(128, activation="relu")(inputs)
    x = keras.layers.Dense(64, activation="relu")(x)
    outputs = keras.layers.Dense(FEATURE_NUM_CLASSES, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.layers[1].set_weights([W1_prime.astype(np.float32), b1_prime.astype(np.float32)])
    model.layers[2].set_weights([W2.astype(np.float32), b2.astype(np.float32)])
    model.layers[3].set_weights([W3.astype(np.float32), b3.astype(np.float32)])
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the source Keras model (.h5 or .keras).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path for the .tflite file. Defaults to <model>.tflite.",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply dynamic range quantization.",
    )
    parser.add_argument(
        "--arch",
        choices=["crnn", "feature", "generic"],
        default="crnn",
        help="Select architecture preset: 'crnn', 'feature', or 'generic' for direct conversion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    output_path = args.output.resolve() if args.output else model_path.with_suffix(".tflite")

    if args.arch == "crnn":
        model = build_crnn_model()
        model.load_weights(model_path)
    elif args.arch == "feature":
        weights = load_feature_weights_npz(model_path)
        model = build_feature_fused_model(weights)
    else:
        model = tf.keras.models.load_model(model_path, compile=False)

    if args.arch == "crnn":
        temp_saved_model = output_path.parent / "_tmp_saved_model"
        if temp_saved_model.exists():
            shutil.rmtree(temp_saved_model)
        tf.saved_model.save(model, temp_saved_model)

        converter = tf.lite.TFLiteConverter.from_saved_model(str(temp_saved_model))
        converter.experimental_enable_resource_variables = True
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
    else:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if args.quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)
    print(f"Converted {model_path} -> {output_path}")

    if args.arch == "crnn":
        shutil.rmtree(temp_saved_model, ignore_errors=True)


if __name__ == "__main__":
    main()


