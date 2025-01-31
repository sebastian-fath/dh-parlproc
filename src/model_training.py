"""
author: Sebastian Fath
purpose: create neural-net model based on data for categorisation of text
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses

import re
import string
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# used to initialize the model structure using tensorflow.keras; TIP call tf.summary() on the resulting model to receive a nicely formatted overview :)
def init_model(output_amount: int, max_features: int = 100000, embedding_dim: int = 16):
    # model contains different preconfigured layers from tensorflow.keras; 
    # init model largely adapted from tensorflow.org/tutorials/keras/text_classification
    model = tf.keras.Sequential([
        layers.Embedding(max_features, embedding_dim),
        layers.Dropout(0.2),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.2),
        layers.Dense(output_amount, activation='sigmoid')
    ])
    model.compile(
        loss = losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer = 'adam',
        metrics = ['accuracy']
    )
    return model

def create_vec_layer(max_features: int =  100000, sequence_length: int = 500):
    vectorize_layer = layers.TextVectorization(
        standardize=custom_standardization,
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length
        )
    return vectorize_layer

def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '\\n', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation),
                                  '')

def train_model(model, training_data, validation_data, epochs: int = 20):
    history = model.fit(training_data, validation_data=validation_data, epochs=epochs)
    return model, history

def add_vec_to_model(model, vectorisation_layer):
    export_model = tf.keras.Sequential([
        vectorisation_layer,
        model,
        layers.Activation('sigmoid')
    ])
    export_model.compile(
        loss=losses.SparseCategoricalCrossentropy(from_logits=True), optimizer="adam", metrics=['accuracy']
    )

    return export_model

def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label



# entrypoint - only run if called from cli
if __name__ == '__main__':
    
    # load data
    data = pd.read_csv("data/data_impure.csv", delimiter="|")
    # look at data
    print("data from csv", data.head(10))

    class_names = data["class"].unique()

    # create folderstructure
    for class_name in class_names:
        class_dir = Path(f"data/dataset/{class_name}")
        class_dir.mkdir(parents=True, exist_ok=True)

    # create data
    for index, row in data.iterrows():
        # print(index, row)
        with open(f"data/dataset/{row["class"]}/{index}.txt", "w") as f:
            f.write(row[1])
    
    # turn data into dataset that is (easier) readable 
    seed = 101
    tf_train_data = tf.keras.utils.text_dataset_from_directory("data/dataset", batch_size=32, validation_split=0.2, subset='training', seed=seed)
    tf_val_data = tf.keras.utils.text_dataset_from_directory("data/dataset", batch_size=32, validation_split=0.2, subset='validation', seed=seed)

    vectorize_layer = create_vec_layer()

    # Make a text-only dataset (without labels), then call adapt
    train_text = tf_train_data.map(lambda x, y: x)
    vectorize_layer.adapt(train_text)

    # adopt data for training 
    train_data = tf_train_data.map(vectorize_text)
    val_data = tf_val_data.map(vectorize_text)
    test_data = tf_train_data.map(vectorize_text)

    model_init = init_model(output_amount=len(tf_train_data.class_names))
    print("Summary of Initial Model:")
    model_init.summary()

    # train model
    model, history = train_model(model_init, train_data, val_data, epochs=12)

    model.summary()

    # add vec_layer to trained model
    final_model = add_vec_to_model(model, vectorize_layer)
    print("Summary of final Model:")
    final_model.summary()

    # eval
    print("final evaluation:")
    metrics = final_model.evaluate(tf_train_data, return_dict=True)
    print(metrics)

    # save model to disk as .keras
    Path("out/").mkdir(parents=True, exist_ok=True)
    model.save('out/parlamentary_actions_cat.keras')