import tensorflow as tf
import keras
import keras.layers.experimental.preprocessing as preprocessing
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Input, \
    AveragePooling2D
from keras.models import Sequential
import numpy as np
from PIL import Image

# GPU 장치 확인
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available:", len(physical_devices))

# 필요한 경우 GPU 메모리 사용 동적 조정
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# 모든 GPU 메모리 사용 허용
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


class InceptionLayer(tf.keras.layers.Layer):
    def __init__(self, num_1x1, num_3x3_reduce, num_3x3, num_5x5_reduce, num_5x5, num_pool_proj):
        super(InceptionLayer, self).__init__()
        self.num_1x1 = num_1x1
        self.num_3x3_reduce = num_3x3_reduce
        self.num_3x3 = num_3x3
        self.num_5x5_reduce = num_5x5_reduce
        self.num_5x5 = num_5x5
        self.num_pool_proj = num_pool_proj

    # 가중치 초기화 (path1, path2, path3, path4 레이어 정의)
    def build(self, input_shape):
        self.path1 = Conv2D(filters=self.num_1x1, kernel_size=(1, 1), strides=1, padding='same', activation='relu',
                            kernel_initializer='he_normal')

        self.path2 = Sequential([
            Conv2D(filters=self.num_3x3_reduce, kernel_size=(1, 1), strides=1, padding='same', activation='relu',
                   kernel_initializer='he_normal'),
            Conv2D(filters=self.num_3x3, kernel_size=(1, 1), strides=1, padding='same', activation='relu',
                   kernel_initializer='he_normal')
        ])

        self.path3 = Sequential([
            Conv2D(filters=self.num_5x5_reduce, kernel_size=(1, 1), strides=1, padding='same', activation='relu',
                   kernel_initializer='he_normal'),
            Conv2D(filters=self.num_5x5, kernel_size=(1, 1), strides=1, padding='same', activation='relu',
                   kernel_initializer='he_normal')
        ])

        self.path4 = Sequential([
            MaxPooling2D(strides=1, pool_size=(3, 3), padding='same'),
            Conv2D(filters=self.num_pool_proj, kernel_size=(1, 1), strides=1, padding='same', activation='relu',
                   kernel_initializer='he_normal')
        ])

    # 레이어 로직 구현 (레이어 사용)
    def call(self, inputs):
        result_path1 = self.path1(inputs)
        result_path2 = self.path2(inputs)
        result_path3 = self.path3(inputs)
        result_path4 = self.path4(inputs)

        return Concatenate()([result_path1, result_path2, result_path3, result_path4])


inputs = Input(shape=(224, 224, 3))

# GoogleNet 모델 초기 부분
x = Conv2D(filters=64, kernel_size=(7,7), strides=2, padding='same', activation='relu',kernel_initializer='he_normal')(inputs)
x = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(filters=192, kernel_size=(1,1), strides=1, padding='valid', activation='relu',kernel_initializer='he_normal')(x)
x = Conv2D(filters=192, kernel_size=(3,3), strides=1, padding='same', activation='relu',kernel_initializer='he_normal')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(x)

# Inception (3a)
x = InceptionLayer(64, 96, 128, 16, 32, 32)(x)
# Inception (3b)
x = InceptionLayer(128,128,192,32,96,64)(x)
# maxPooling
x = MaxPooling2D(strides=2, pool_size=(3,3), padding='same')(x)
# Inception(4a)
x = InceptionLayer(192, 96, 208, 16, 48, 64)(x)

# auxiliary classifier (4a)
x1 = x
x1 = AveragePooling2D(strides=3, pool_size=(5,5), padding='valid')(x1)
x1 = Conv2D(filters=128, strides=1, kernel_size=(1,1), padding='same', activation='relu',kernel_initializer='he_normal')(x1)
x1 = Flatten()(x1)
x1 = Dense(units=1024, activation='relu',kernel_initializer='he_normal')(x1)
x1 = Dropout(0.7)(x1)
x1 = Dense(units=10, activation='softmax',kernel_initializer='he_normal')(x1)

# Inception(4b)
x = InceptionLayer(160, 112, 224, 24, 64, 64)(x)
# Inception(4c)
x = InceptionLayer(128, 128, 256, 24, 64, 64)(x)
# Inception(4d)
x = InceptionLayer(112, 144,288,32,64,64)(x)

# auxiliary classifier2 (4d)
x2 = x
x2 = AveragePooling2D(strides=3, pool_size=(5,5), padding='same')(x2)
x2 = Conv2D(filters=128, strides=1, kernel_size=(1,1), padding='same', activation='relu',kernel_initializer='he_normal')(x2)
x2 = Flatten()(x2)
x2 = Dense(units=1024, activation='relu',kernel_initializer='he_normal')(x2)
x2 = Dropout(0.7)(x2)
x2 = Dense(units=10, activation='softmax',kernel_initializer='he_normal')(x2)

# Inception(4e)
x = InceptionLayer(256, 160, 320, 32, 128, 128)(x)
# maxPooling
x = MaxPooling2D(strides=2, pool_size=(3,3), padding='same')(x)
# Inception(5a)
x = InceptionLayer(256, 160, 320, 32, 128, 128)(x)
# Inception(5b)
x = InceptionLayer(384, 192, 384, 48, 128, 128)(x)
# AveragePooling
x = AveragePooling2D(pool_size=(7,7), strides=1, padding='valid')(x)
# dropout
x = Dropout(0.4)(x)
# linear
x = Dense(1000, activation='relu',kernel_initializer='he_normal')(x)
x = Flatten()(x)
# softmax
out = Dense(units=10, activation='softmax',kernel_initializer='he_normal')(x)

model = keras.Model(inputs, outputs=[out, x1, x2])
model.load_weights('/home/ivpl-d28/2024(1)_study/GoogleNet/GoogleNet_param_weight.h5')
# reduce_lr
class CustomLearningRateScheduler(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if epoch > 0 and epoch % 8 == 0:
            new_lr = self.model.optimizer.lr * 0.96
            tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
            print(f"Learning rate이 {new_lr}로 감소하였습니다.")

# model compile
model.compile(loss=[keras.losses.sparse_categorical_crossentropy,
                    keras.losses.sparse_categorical_crossentropy,
                    keras.losses.sparse_categorical_crossentropy],
              loss_weights=[1,0.3,0.3],
              optimizer=keras.optimizers.SGD(momentum=0.9, learning_rate=0.001),
              metrics=['sparse_categorical_accuracy', keras.metrics.TopKCategoricalAccuracy(k=5)])

class_name = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse","ship", "truck"]

image = Image.open('/home/ivpl-d28/2024(1)_study/ship.jpg')

data_augmentation = keras.Sequential([
    preprocessing.Resizing(224, 224),
    preprocessing.Rescaling(1. / 255)
])

image = np.expand_dims(np.array(image), axis=0)
image = data_augmentation(image)

predictions = model.predict(image)

main_predictions = predictions[0][0]

for class_idx, class_probability in enumerate(main_predictions):
    print(f"{class_name[class_idx]}: {class_probability*100:.2f}%\n")

predicted_class_index = np.argmax(main_predictions)
highest_probability = main_predictions[predicted_class_index]
print(f"\nPrediction: {class_name[predicted_class_index]}")






