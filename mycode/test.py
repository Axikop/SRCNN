import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from model import build_srcnn
from data_preprocess import load_and_preprocess_image

def main():
    
    model = build_srcnn()
    model.load_weights('models/srcnn.h5')

    
    test_image = load_and_preprocess_image(r'C:\Users\adi20\Desktop\letsee.jpg', (64, 64))
    test_image = tf.expand_dims(test_image, axis=0)

    
    predicted_image = model.predict(test_image)

    
    predicted_image = tf.squeeze(predicted_image, axis=0)
    predicted_image = tf.clip_by_value(predicted_image, 0, 1)

    
    actual_image = load_and_preprocess_image(r'C:\Users\adi20\Desktop\letsee.jpg', (256, 256))

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(tf.squeeze(test_image, axis=0))
    plt.title('Original Image')
    plt.subplot(1, 3, 2)
    plt.imshow(predicted_image)
    plt.title('Predicted Image')
    plt.subplot(1, 3, 3)
    plt.imshow(actual_image)
    plt.title('Actual Image')
    plt.show()

if __name__ == "__main__":
    main()