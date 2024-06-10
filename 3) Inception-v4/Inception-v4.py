import tensorflow as tf
import matplotlib.pyplot as plt
import keras
import keras.layers.experimental.preprocessing as preprocessing
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Concatenate, Input, AveragePooling2D, Activation
import datetime
import tensorflow.keras.datasets as df

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
imagenet_train_data = keras.preprocessing.image_dataset_from_directory(
    '/home/ivpl-d28/dataset/Imagenet/imagenet/train',
    image_size=(512,512))
imagenet_valid_data = keras.preprocessing.image_dataset_from_directory(
    '/home/ivpl-d28/dataset/Imagenet/imagenet/val',
    image_size=(512,512))
imagenet_test_data = keras.preprocessing.image_dataset_from_directory(
    '/home/ivpl-d28/dataset/Imagenet/imagenet/test',
    image_size=(512, 512))

print(f'train: {len(imagenet_train_data)}, valid: {len(imagenet_valid_data)}, test: {len(imagenet_test_data)}')

# # 데이터 샘플링
# train_sample = imagenet_train_data.take(2000)
# valid_sample = imagenet_valid_data.take(200)
# test_sample = imagenet_test_data.take(200)

# 데이터 전처리
data_augmentation = keras.Sequential([
    preprocessing.Rescaling(1./255, 1./255),
    preprocessing.RandomRotation(0.1),
    preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
    preprocessing.RandomZoom(0.1),
    preprocessing.RandomFlip(mode='horizontal'),
    preprocessing.Resizing(299, 299) # 크기 조정

])

# pca 함수
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

# 평균차감
def subtract_mean(image):
    image = tf.cast(image, tf.float32)
    mean = tf.reduce_mean(image, axis=[1,2], keepdims=True)
    centered_image = image - mean

    return centered_image

# 데이터 전처리 적용
augmented_train_data = imagenet_train_data.map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
augmented_valid_data = imagenet_valid_data.map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
augmented_test_data = imagenet_test_data.map(lambda x, y: (data_augmentation(x), y), num_parallel_calls = tf.data.experimental.AUTOTUNE)

# 평균 차감
augmented_train_data = augmented_train_data.map(lambda x, y: (subtract_mean(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
augmented_valid_data = augmented_valid_data.map(lambda x, y: (subtract_mean(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
augmented_test_data = augmented_test_data.map(lambda x, y: (subtract_mean(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)

# for images, labels in augmented_train_data.take(1):
#     image = images[0].numpy()
#     label = labels[0].numpy()
#
#     print(f'Min pixel value: {image.min()}, Max pixel value: {image.max()}')
#     print(f'Size of Image: {image.shape}')
#     plt.imshow(image)
#     plt.title(f'Label: {label}')
#     plt.show()

# 학습 효율을 위한 prefetch
augmented_train_data = augmented_train_data.prefetch(tf.data.experimental.AUTOTUNE)
augmented_valid_data = augmented_valid_data.prefetch(tf.data.experimental.AUTOTUNE)
augmented_test_data = augmented_test_data.prefetch(tf.data.experimental.AUTOTUNE)

print(f"augmented_train_data: {len(augmented_train_data)}")
print(f"augmented_valid_data: {len(augmented_valid_data)}")
print(f"augmented_test_data: {len(augmented_test_data)}")

def Inception_A(x):
    x1 = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(x)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(filters=96, kernel_size=(1,1), strides=(1,1), activation='relu', padding='same', kernel_initializer='he_normal')(x1)

    x2 = Conv2D(filters=96, kernel_size=(1,1), strides=(1,1), activation='relu', padding='same', kernel_initializer='he_normal')(x)

    x3 = Conv2D(filters=64, kernel_size=(1,1), strides=(1,1), activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x3 = Conv2D(filters=96, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', kernel_initializer='he_normal')(x3)

    x4 = Conv2D(filters=64, kernel_size=(1,1), strides=(1,1), activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x4 = Conv2D(filters=96, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', kernel_initializer='he_normal')(x4)
    x4 = Conv2D(filters=96, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', kernel_initializer='he_normal')(x4)

    result = Concatenate()([x1, x2, x3, x4])

    return result


def Inception_B(x):
    x1 = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(x)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(filters=128, kernel_size=(1,1), strides=(1,1), activation='relu', padding='same', kernel_initializer='he_normal')(x1)

    x2 = Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), activation='relu', padding='same', kernel_initializer='he_normal')(x)

    x3 = Conv2D(filters=192, kernel_size=(1,1), strides=(1,1), activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x3 = Conv2D(filters=224, kernel_size=(1,7), strides=(1,1), activation='relu', padding='same', kernel_initializer='he_normal')(x3)
    x3 = Conv2D(filters=256, kernel_size=(7,1), strides=(1,1), activation='relu', padding='same', kernel_initializer='he_normal')(x3)

    x4 = Conv2D(filters=192, kernel_size=(1,1), strides=(1,1), activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x4 = Conv2D(filters=192, kernel_size=(1,7), strides=(1,1), activation='relu', padding='same', kernel_initializer='he_normal')(x4)
    x4 = Conv2D(filters=224, kernel_size=(7,1), strides=(1,1), activation='relu', padding='same', kernel_initializer='he_normal')(x4)
    x4 = Conv2D(filters=224, kernel_size=(1,7), strides=(1,1), activation='relu', padding='same', kernel_initializer='he_normal')(x4)
    x4 = Conv2D(filters=256, kernel_size=(7,1), strides=(1,1), activation='relu', padding='same', kernel_initializer='he_normal')(x4)

    result = Concatenate()([x1, x2, x3, x4])

    return result

def Inception_C(x):
    x1 = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(x)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding='same', kernel_initializer='he_normal')(x1)

    x2 = Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding='same', kernel_initializer='he_normal')(x)

    x3 = Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x3_1 = Conv2D(filters=256, kernel_size=(1,3), strides=(1,1), activation='relu', padding='same', kernel_initializer='he_normal')(x3)
    x3_2 = Conv2D(filters=256, kernel_size=(3,1), strides=(1,1), activation='relu', padding='same', kernel_initializer='he_normal')(x3)

    x4 = Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x4 = Conv2D(filters=448, kernel_size=(1,3), strides=(1,1), activation='relu', padding='same', kernel_initializer='he_normal')(x4)
    x4 = Conv2D(filters=512, kernel_size=(3,1), strides=(1,1), activation='relu', padding='same', kernel_initializer='he_normal')(x4)
    x4_1 = Conv2D(filters=256, kernel_size=(3,1), strides=(1,1), activation='relu', padding='same', kernel_initializer='he_normal')(x4)
    x4_2 = Conv2D(filters=256, kernel_size=(1,3), strides=(1,1), activation='relu', padding='same', kernel_initializer='he_normal')(x4)

    result = Concatenate()([x1, x2, x3_1, x3_2, x4_1, x4_2])

    return result


input = Input(shape=(299, 299, 3))

# Stem
x = Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), activation='relu', padding='valid', kernel_initializer='he_normal')(input)
x = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu', padding='valid', kernel_initializer='he_normal')(x)
x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', kernel_initializer='he_normal')(x)

x1 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid')(x)
x1 = Activation('relu')(x1)
x2 = Conv2D(filters=96, kernel_size=(3,3), strides=(2,2), padding='valid', activation='relu', kernel_initializer='he_normal')(x)
x = Concatenate()([x1, x2])
print(f'stem concate1: {x.shape}')

x1 = Conv2D(filters=64, kernel_size=(1,1), strides=(1,1), activation='relu', padding='same', kernel_initializer='he_normal')(x)
x1 = Conv2D(filters=96, kernel_size=(3,3), strides=(1,1), activation='relu', padding='valid', kernel_initializer='he_normal')(x1)

x2 = Conv2D(filters=64, kernel_size=(1,1), strides=(1,1), activation='relu', padding='same', kernel_initializer='he_normal')(x)
x2 = Conv2D(filters=64, kernel_size=(7,1), strides=(1,1), activation='relu', padding='same', kernel_initializer='he_normal')(x2)
x2 = Conv2D(filters=64, kernel_size=(1,7), strides=(1,1), activation='relu', padding='same', kernel_initializer='he_normal')(x2)
x2 = Conv2D(filters=96, kernel_size=(3,3), strides=(1,1), activation='relu', padding='valid', kernel_initializer='he_normal')(x2)
x = Concatenate()([x1, x2])
print(f'stem concate2: {x.shape}')

x1 = Conv2D(filters=192, kernel_size=(3,3), strides=(2,2), activation='relu', padding='valid', kernel_initializer='he_normal')(x)

x2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid')(x)
x2 = Activation('relu')(x2)
x = Concatenate()([x1, x2])
print(f'stem output shape: {x.shape}')

# Inception-A
x = Inception_A(x)
x = Inception_A(x)
x = Inception_A(x)
x = Inception_A(x)
print(f'Inception-A output shape: {x.shape}')

# k = 192, l = 224, m = 256, n = 384
# Reduction-A
x1 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid')(x)
x1 = Activation('relu')(x1)

x2 = Conv2D(filters=384, kernel_size=(3,3), strides=(2,2), activation='relu', padding='valid', kernel_initializer='he_normal')(x)

x3 = Conv2D(filters=192, kernel_size=(1,1), strides=(1,1), activation='relu', padding='same', kernel_initializer='he_normal')(x)
x3 = Conv2D(filters=224, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', kernel_initializer='he_normal')(x3)
x3 = Conv2D(filters=256, kernel_size=(3,3), strides=(2,2), activation='relu', padding='valid', kernel_initializer='he_normal')(x3)

x = Concatenate()([x1, x2, x3])
print(f'Reduction-A output shape: {x.shape}')

# Inception-B
x = Inception_B(x)
x = Inception_B(x)
x = Inception_B(x)
x = Inception_B(x)
x = Inception_B(x)
x = Inception_B(x)
x = Inception_B(x)
print(f'Inception-B output shape: {x.shape}')

# Reduction-B
x1 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid')(x)
x1 = Activation('relu')(x1)

x2 = Conv2D(filters=192, kernel_size=(1,1), strides=(1,1), activation='relu', padding='same', kernel_initializer='he_normal')(x)
x2 = Conv2D(filters=192, kernel_size=(3,3), strides=(2,2), activation='relu', padding='valid', kernel_initializer='he_normal')(x2)

x3 = Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding='same', kernel_initializer='he_normal')(x)
x3 = Conv2D(filters=256, kernel_size=(1,7), strides=(1,1), activation='relu', padding='same', kernel_initializer='he_normal')(x3)
x3 = Conv2D(filters=320, kernel_size=(7,1), strides=(1,1), activation='relu', padding='same', kernel_initializer='he_normal')(x3)
x3 = Conv2D(filters=320, kernel_size=(3,3), strides=(2,2), activation='relu', padding='valid', kernel_initializer='he_normal')(x3)

x = Concatenate()([x1, x2, x3])
print(f'Reduction-B output shape: {x.shape}')

# Inception-C
x = Inception_C(x)
x = Inception_C(x)
x = Inception_C(x)
print(f'Inception-C output shape: {x.shape}')

# Average Pooling
x = AveragePooling2D(pool_size=(8,8), strides=(1,1), padding='valid')(x)
x = Activation('relu')(x)
x = Flatten()(x)
print(f'AveragePooling2D output shape: {x.shape}')

# Dropout
x = Dropout(0.2)(x)
print(f'Dropout-B output shape: {x.shape}')

# Softmax
out = Dense(1000, activation='softmax', kernel_initializer='he_normal')(x)
print(f'Softmax output shape: {out.shape}')

# model
model = keras.Model(input, out)

model.summary()

pass

# redeuce_lr
class CustomLearningRateScheduler(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if epoch > 0 and epoch % 2 == 0 :
            new_lr = self.model.optimizer.lr * 0.94
            tf. keras.backend.set_value(self.model.optimizer.lr, new_lr)
            print(f'Learning rate이 {new_lr}로 감소하였습니다.')

# tensorboard
log_dir = "logs/model"+"/"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch='3,7')


# model complile
model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.RMSprop(rho=0.9, epsilon=1.0, learning_rate=0.045),
              metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=5)])

# train model
history = model.fit(augmented_train_data,
                    batch_size=128,
                    epochs=30,
                    validation_data= augmented_valid_data,
                    callbacks=[CustomLearningRateScheduler(), tensorboard_callback])

# model test
score = model.evaluate(augmented_test_data)
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