import tensorflow as tf
from keras.layers import UpSampling2D
from keras.models import Sequential
from keras.layers import Conv2D
from keras import optimizers

def build_srcnn(input_shape=(None, None, 3), 
                num_filters=[64, 32], 
                kernel_sizes=[(9, 9), (1, 1), (5, 5)], 
                activation='relu', 
                final_activation='linear', 
                optimizer='adam', 
                loss='mean_squared_error'):
    
    assert len(num_filters) + 1 == len(kernel_sizes), "Number of kernel_sizes should be one more than num_filters"

    model = Sequential()
    
    
    model.add(UpSampling2D(size=(4, 4), input_shape=input_shape))
    
    model.add(Conv2D(num_filters[0], kernel_sizes[0], activation=activation, padding='same'))
    
    for filters, kernel_size in zip(num_filters[1:], kernel_sizes[1:-1]):
        model.add(Conv2D(filters, kernel_size, activation=activation, padding='same'))
    
    model.add(Conv2D(input_shape[-1], kernel_sizes[-1], activation=final_activation, padding='same'))
    
    model.compile(optimizer=optimizer, loss=loss)
    
    return model