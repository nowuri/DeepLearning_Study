import tensorflow.keras as keras
import tensorflow as tf
tf.executing_eagerly()
import tensorflow.keras.layers.experimental.preprocessing as preprocessing
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, Flatten
import matplotlib.pyplot as plt
import tensorflow_transform as tft

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

# GPU 메모리 사용 확인
if len(physical_devices) > 0:
    for device in physical_devices:
        try:
            print("GPU Memory Usage: ")
            tf.config.experimental.set_memory_growth(device, True)
            logical_devices = tf.config.experimental.list_logical_devices('GPU')
        except RuntimeError as e:
            print(e)
else:
    print("No GPUs available")

data_augmentation = keras.Sequential(
    [
        preprocessing.Resizing(256,256),
        preprocessing.CenterCrop(227,227),
        preprocessing.RandomFlip(mode='horizontal')
    ]
)

def pca_augmentation(images):
    # 이미지 데이터를 float32 타입으로 변환
    images = tf.cast(images, tf.float32)

    # 이미지의 평균을 계산
    mean = tf.reduce_mean(images, axis=[1,2], keepdims=True)

    # 이미지에서 평균을 빼서 중심을 맞춤
    centered_images = images - mean

    # 각 이미지의 공분산 행렬을 계산
    covariance = tf.linalg.einsum('bijc,bijd->bcd', centered_images, centered_images) / tf.cast((tf.shape(images)[1] * tf.shape(images)[2]), tf.float32)

    # 공분산 행렬의 고유값과 고유벡터를 찾음
    e, v = tf.linalg.eigh(covariance)

    # 고유벡터(주성분)에 랜덤 노이즈를 곱하고 원래 이미지에 더함
    rand = tf.random.normal(shape=(tf.shape(images)[0], 3, 1, 1), mean=0., stddev=0.1)
    e = tf.expand_dims(e, axis=-1)
    e = tf.expand_dims(e, axis=-1)
    v = tf.transpose(v, perm=[0, 2, 1])
    v = tf.expand_dims(v, axis=-1)
    delta = tf.reduce_sum(tf.multiply(v, rand * e), axis=2)
    delta = tf.squeeze(delta, axis=-1)
    delta = tf.expand_dims(tf.expand_dims(delta, 1), 1)
    delta = tf.tile(delta, [1, tf.shape(images)[1], tf.shape(images)[2], 1])

    # 증강된 이미지 생성
    augmented_images = images + delta

    # 증강된 이미지의 값이 0과 255 사이가 되도록 클리핑
    augmented_images = tf.clip_by_value(augmented_images, 0., 255.)

    return augmented_images


# 데이터 로드
imagenet_train_data = keras.preprocessing.image_dataset_from_directory('/home/ivpl-d28/dataset/Imagenet/imagenet/train')
imagenet_test_data = keras.preprocessing.image_dataset_from_directory('/home/ivpl-d28/dataset/Imagenet/imagenet/test')

# 데이터 증강
# map(): 입력 데이터셋의 각 요소에 함수를 적용하고 결과를 반환
# 아래 코드는 이미지와 레이블 쌍 (x, y)에 대해 data_augmentation 함수 적용해서 레이블 y 반환
# training = True : 데이터 증강 적용, False: 데이터 증강 적용 X
augmented_train_data = imagenet_train_data.map(lambda x, y: (data_augmentation(x, training=True), y))
augmented_test_data = imagenet_test_data.map(lambda x, y: (data_augmentation(x, training=True), y))

augmented_train_data = augmented_train_data.prefetch(tf.data.experimental.AUTOTUNE)
augmented_test_data = augmented_test_data.prefetch(tf.data.experimental.AUTOTUNE)

# PCA 증강 적용
output_dim = 3  # RGB 채널이므로 3개의 주성분 선택
height, width, channels = 227,227,3
augmented_train_data = augmented_train_data.map(lambda x, y: (pca_augmentation(x), y))
augmented_test_data = augmented_test_data

# PCA 증강 적용 후 첫 번째 이미지의 형태 확인
# for images, labels in augmented_train_data.take(1):
#     print(images[0].shape)

model = Sequential()

# C1 layer
model.add(Conv2D(filters = 96, strides = 4, kernel_size =(11, 11), activation = 'relu', input_shape=(height, width, channels)))
# tf.nn.local_response_normalization: tensorflow의 low-level API 함수라 Sequential 모델에 직접 사용 불가
model.add(BatchNormalization())
model.add(MaxPooling2D(strides=2, pool_size=(3,3), padding = 'valid'))

# C2 layer
# GPU 1개로 구현해서 5x5x96으로 만듦
model.add(Conv2D(filters = 245, strides = 1, kernel_size = (5,5), activation='relu', padding = 'same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(strides=2, pool_size=(3,3), padding = 'valid'))

# C3 layer
model.add(Conv2D(filters = 384, strides = 1, kernel_size=(3,3), activation='relu',padding='same'))

# C4 layer
model.add(Conv2D(filters = 384, strides = 1, kernel_size=(3,3), activation='relu', padding = 'same'))

# C5 layer
model.add(Conv2D(filters = 256, strides = 1, kernel_size=(3,3), activation='relu', padding = 'same'))
model.add(MaxPooling2D(strides=2, pool_size=(3,3), padding='valid'))
model.add(Flatten())

# F6 layer
model.add(Dense(units = 4096, activation = 'relu'))
model.add(Dropout(0.5))

# F7 layer
model.add(Dense(units = 4096, activation = 'relu'))
model.add(Dropout(0.5))

# F8 layer (Output layer)
model.add(Dense(units = 1000, activation='softmax'))

# model compile (accuracy: top-1 error rate을 위함, TopKCategoricalAccuracy: top-5 error rate을 위함)
model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, weight_decay=0.05),
              metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=5)])

# 모델이 10 epochs 동안 성능 개선이 되지 않으면, 10으로 나눔
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, min_lr=0.0001)

# train model
history = model.fit(augmented_train_data, batch_size = 128, epochs = 90, verbose = 1,
                    validation_data= augmented_test_data, callbacks=[reduce_lr])

# test model
score = model.evaluate(augmented_test_data, verbose = 0)
print('Test loss: ', score[0])
print('Test top-1 error rate: ', 1-score[1])
print('Test top-5 error rate: ', 1-score[2])

# loss function 그리기
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Accuracy 그리기
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label = 'Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()