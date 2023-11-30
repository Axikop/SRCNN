import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from data_preprocess import load_dataset
import keras 
from model import build_srcnn

def train_model(train_dataset, val_dataset, model, epochs, model_path):
    checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, 
              callbacks=[checkpoint, early_stopping])

def main():
    
    train_dataset, val_dataset = load_dataset(batch_size=32)

    
    model = build_srcnn()

    
    train_model(train_dataset, val_dataset, model, epochs=100, model_path='models/srcnn.h5')
    

if __name__ == "__main__":
    main()