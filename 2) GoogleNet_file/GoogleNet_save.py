import tensorflow as tf
import keras
import keras.layers.experimental.preprocessing as preprocessing
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Input, AveragePooling2D
from keras.models import Sequential
from datetime import datetime
import datetime as dt
import os
from sklearn.model_selection import train_test_split

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

# 데이터 불러오기
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# tf.data.Dataset으로 변환
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

# 데이터 전처리
data_augmentation = keras.Sequential([
    preprocessing.Resizing(224, 224), # 크기 조정
    preprocessing.Rescaling(1./255, 1./255) # RGB 값이 [0,1] 범위 내에 있도록 표준화
])

# 데이터 증강 적용 함수s
def augment_data(image, label):
    image = data_augmentation(image)
    return image, label

# 데이터 전처리 적용
augmented_train_data = train_dataset.map(augment_data).batch(32)
augmented_test_data = test_dataset.map(augment_data).batch(32)

# 학습 효율을 위한 prefetch
augmented_train_data = augmented_train_data.prefetch(tf.data.experimental.AUTOTUNE)
augmented_test_data = augmented_test_data.prefetch(tf.data.experimental.AUTOTUNE)

# Inception model
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
    # 가중치 초기화 (path1, path2, path3, path4 레이어 정의
    def build(self, input_shape):

        self.path1 = Conv2D(filters=self.num_1x1, kernel_size=(1, 1), strides=1, padding='same', activation='relu', kernel_initializer='he_normal')

        self.path2 = Sequential([
            Conv2D(filters=self.num_3x3_reduce, kernel_size=(1, 1), strides=1, padding='same', activation='relu',kernel_initializer='he_normal'),
            Conv2D(filters=self.num_3x3, kernel_size=(1, 1), strides=1, padding='same', activation='relu',kernel_initializer='he_normal')
        ])

        self.path3 = Sequential([
            Conv2D(filters=self.num_5x5_reduce, kernel_size=(1, 1), strides=1, padding='same', activation='relu',kernel_initializer='he_normal'),
            Conv2D(filters=self.num_5x5, kernel_size=(1, 1), strides=1, padding='same', activation='relu',kernel_initializer='he_normal')
        ])

        self.path4 = Sequential([
            MaxPooling2D(strides=1, pool_size=(3, 3), padding='same'),
            Conv2D(filters=self.num_pool_proj, kernel_size=(1, 1), strides=1, padding='same', activation='relu',kernel_initializer='he_normal')
        ])

    # 레이어 로직 구현 (레이어 사용)
    def call(self, inputs):
        result_path1 = self.path1(inputs)
        result_path2 = self.path2(inputs)
        result_path3 = self.path3(inputs)
        result_path4 = self.path4(inputs)

        return Concatenate()([result_path1, result_path2, result_path3, result_path4])

# auxiliary classifier처럼 중간에 들어가는 구조는 Sequential에서 추가하기 어려움
# Sequential은 레이어가 순차적으로 연결된 간단한 모델에 최적화되어 있음
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
model_test=keras.Model(inputs, out)
model.summary()

# reduce_lr
class CustomLearningRateScheduler(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if epoch > 0 and epoch % 8 == 0:
            new_lr = self.model.optimizer.lr * 0.96
            tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
            print(f"Learning rate이 {new_lr}로 감소하였습니다.")

# tensorboard
log_dir = "logs/model"+"/"+ dt.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# model compile
model.compile(loss=[keras.losses.sparse_categorical_crossentropy,
                    keras.losses.sparse_categorical_crossentropy,
                    keras.losses.sparse_categorical_crossentropy],
              loss_weights=[1,0.3,0.3],
              optimizer=keras.optimizers.SGD(momentum=0.9, learning_rate=0.001),
              metrics=['sparse_categorical_accuracy', keras.metrics.TopKCategoricalAccuracy(k=5)])

# train model
history= model.fit(augmented_train_data, epochs=50, verbose=1,
                    callbacks=[CustomLearningRateScheduler(), tensorboard_callback])
# weight 저장
# current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
# model_path = f"/home/nrchoi/2024_study(1)/Saved_model/GoogleNet_param_{current_time}.h5"
#
# if os.path.exists(model_path):
#     os.remove(model_path)
# model.save(model_path)
model.save_weights('/home/ivpl-d28/2024(1)_study/GoogleNet_file/GoogleNet_param_weight.h5')

model_test.compile(loss=keras.losses.sparse_categorical_crossentropy,
                   optimizer=keras.optimizers.SGD(momentum=0.9, learning_rate=0.001),
                   metrics=['sparse_categorical_accuracy', keras.metrics.TopKCategoricalAccuracy(k=5)])

# 테스트 모델 평가
loss, accuracy, top_5_accuracy = model_test.evaluate(augmented_test_data)

print('Test loss:', loss)
print('Test top-1 error rate:', 1-accuracy)
print('Test top-5 error rate:', 1-top_5_accuracy)