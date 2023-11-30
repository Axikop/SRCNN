import os
import tensorflow as tf
import numpy

def load_and_preprocess_image(image_path, target_size):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_size)
    return image

def create_image_dataset(directory, target_size):
    image_paths = tf.data.Dataset.list_files(os.path.join(directory, '*'))
    image_dataset = image_paths.map(lambda x: load_and_preprocess_image(x, target_size))
    return image_dataset.prefetch(tf.data.experimental.AUTOTUNE)

def load_dataset(batch_size, validation_split=0.2):
    high_res_dataset = create_image_dataset("dataset/Raw Data/high_res", (256, 256))
    low_res_dataset = create_image_dataset("dataset/Raw Data/low_res", (64, 64))
    
    dataset = tf.data.Dataset.zip((low_res_dataset, high_res_dataset))
    
    
    num_samples = tf.data.experimental.cardinality(dataset).numpy()
    num_val_samples = int(num_samples * validation_split)

    
    val_dataset = dataset.take(num_val_samples)
    train_dataset = dataset.skip(num_val_samples)
    
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    val_dataset = val_dataset.batch(batch_size, drop_remainder=True)
    
    return train_dataset, val_dataset


if __name__ == "__main__":
    train_dataset, val_dataset = load_dataset(batch_size=32)

    for low_res, high_res in train_dataset.take(1):
        print(low_res.shape, high_res.shape)