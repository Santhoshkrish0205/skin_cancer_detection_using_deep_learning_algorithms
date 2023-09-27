import tensorflow as tf
from keras.layers import Conv2D, Flatten, Dense, MaxPool2D,BatchNormalization,DepthwiseConv2D,Dropout
from keras.models import Sequential

classes = {
    0: ("Actinic keratoses and intraepithelial carcinomae(Cancer)"),
    1: ("Basal cell carcinoma(Cancer)"),
    2: ("Benign keratosis-like lesions(Non-Cancerous)"),
    3: ("Dermatofibroma(Non-Cancerous)"),
    4: ("Melanocytic nevus(Non-Cancerous)"),
    5: ("Pyogenic granulomas and hemorrhage(Can lead to cancer)"),
    6: ("Melanoma(Cancer)"),
}


model = Sequential()

model.add(Conv2D(16, 
                 kernel_size = (3,3), 
                 input_shape = (28, 28, 3), 
                 activation = 'relu', 
                 padding = 'same'))

model.add(MaxPool2D(pool_size = (2,2)))
model.add(BatchNormalization())

model.add(DepthwiseConv2D(kernel_size = (3,3),
                          activation = 'relu',
                          padding = 'same'))
model.add(Conv2D(filters = 32, kernel_size = (1,1), activation = 'relu'))
model.add(BatchNormalization())

model.add(DepthwiseConv2D(kernel_size = (3,3),
                          activation = 'relu',
                          padding = 'same'))
model.add(Conv2D(filters = 64, kernel_size = (1,1), activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(BatchNormalization())

model.add(DepthwiseConv2D(kernel_size = (3,3),
                          activation = 'relu',
                          padding = 'same'))
model.add(Conv2D(filters = 128, kernel_size = (1,1), activation = 'relu'))
model.add(BatchNormalization())

model.add(DepthwiseConv2D(kernel_size = (3,3),
                          activation = 'relu',
                          padding = 'same'))
model.add(Conv2D(filters = 256, kernel_size = (1,1), activation = 'relu'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))

model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))

model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))

model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))

model.add(BatchNormalization())
model.add(Dense(7, activation='softmax'))

model.summary()

model.load_weights('dcnn_model.h5')