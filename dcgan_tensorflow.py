import os
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
from math import floor

from keras.layers import Conv2D, LeakyReLU, BatchNormalization, Dense, AveragePooling2D, GaussianNoise
from keras.layers import Reshape, UpSampling2D, Activation, Dropout, Flatten, Conv2DTranspose
from keras.models import model_from_json, Sequential
from keras.optimizers import Adam


def zero():
    return np.random.uniform(0.0, 0.01, size=[1])


def one():
    return np.random.uniform(0.99, 1.0, size=[1])


def noise(n):
    return np.random.uniform(-1.0, 1.0, size=[n, 4096])


all_images = []
images_path = "data/DeGods"
files = os.listdir(images_path)
for file in files:
    image = Image.open(f"{images_path}/{file}")
    image = image.resize((256, 256))
    image = np.array(image.convert('RGB'), dtype='float32')
    all_images.append(image / 255)
    print(f"{images_path}/{file} added!")


class GAN(object):
    def __init__(self):

        # Models
        self.D = None
        self.G = None

        self.OD = None

        self.DM = None
        self.AM = None

        # Config
        self.LR = 0.0001
        self.steps = 1

    def discriminator(self):

        if self.D:
            return self.D

        self.D = Sequential()

        # add Gaussian noise to prevent Discriminator overfitting
        self.D.add(GaussianNoise(0.2, input_shape=[256, 256, 3]))

        # 256x256x3 Image
        self.D.add(Conv2D(filters=8, kernel_size=3, padding='same'))
        self.D.add(LeakyReLU(0.2))
        self.D.add(Dropout(0.25))
        self.D.add(AveragePooling2D())

        # 128x128x8
        self.D.add(Conv2D(filters=16, kernel_size=3, padding='same'))
        self.D.add(BatchNormalization(momentum=0.7))
        self.D.add(LeakyReLU(0.2))
        self.D.add(Dropout(0.25))
        self.D.add(AveragePooling2D())

        # 64x64x16
        self.D.add(Conv2D(filters=32, kernel_size=3, padding='same'))
        self.D.add(BatchNormalization(momentum=0.7))
        self.D.add(LeakyReLU(0.2))
        self.D.add(Dropout(0.25))
        self.D.add(AveragePooling2D())

        # 32x32x32
        self.D.add(Conv2D(filters=64, kernel_size=3, padding='same'))
        self.D.add(BatchNormalization(momentum=0.7))
        self.D.add(LeakyReLU(0.2))
        self.D.add(Dropout(0.25))
        self.D.add(AveragePooling2D())

        # 16x16x64
        self.D.add(Conv2D(filters=128, kernel_size=3, padding='same'))
        self.D.add(BatchNormalization(momentum=0.7))
        self.D.add(LeakyReLU(0.2))
        self.D.add(Dropout(0.25))
        self.D.add(AveragePooling2D())

        # 8x8x128
        self.D.add(Conv2D(filters=256, kernel_size=3, padding='same'))
        self.D.add(BatchNormalization(momentum=0.7))
        self.D.add(LeakyReLU(0.2))
        self.D.add(Dropout(0.25))
        self.D.add(AveragePooling2D())

        # 4x4x256
        self.D.add(Flatten())

        # 256
        self.D.add(Dense(128))
        self.D.add(LeakyReLU(0.2))

        self.D.add(Dense(1, activation='sigmoid'))

        return self.D

    def generator(self):

        if self.G:
            return self.G

        self.G = Sequential()

        self.G.add(Reshape(target_shape=[1, 1, 4096], input_shape=[4096]))

        # 1x1x4096
        self.G.add(Conv2DTranspose(filters=256, kernel_size=4))
        self.G.add(Activation('relu'))

        # 4x4x256 - kernel sized increased by 1
        self.G.add(Conv2D(filters=256, kernel_size=4, padding='same'))
        self.G.add(BatchNormalization(momentum=0.7))
        self.G.add(Activation('relu'))
        self.G.add(UpSampling2D())

        # 8x8x256 - kernel sized increased by 1
        self.G.add(Conv2D(filters=128, kernel_size=4, padding='same'))
        self.G.add(BatchNormalization(momentum=0.7))
        self.G.add(Activation('relu'))
        self.G.add(UpSampling2D())

        # 16x16x128
        self.G.add(Conv2D(filters=64, kernel_size=3, padding='same'))
        self.G.add(BatchNormalization(momentum=0.7))
        self.G.add(Activation('relu'))
        self.G.add(UpSampling2D())

        # 32x32x64
        self.G.add(Conv2D(filters=32, kernel_size=3, padding='same'))
        self.G.add(BatchNormalization(momentum=0.7))
        self.G.add(Activation('relu'))
        self.G.add(UpSampling2D())

        # 64x64x32
        self.G.add(Conv2D(filters=16, kernel_size=3, padding='same'))
        self.G.add(BatchNormalization(momentum=0.7))
        self.G.add(Activation('relu'))
        self.G.add(UpSampling2D())

        # 128x128x16
        self.G.add(Conv2D(filters=8, kernel_size=3, padding='same'))
        self.G.add(Activation('relu'))
        self.G.add(UpSampling2D())

        # 256x256x8
        self.G.add(Conv2D(filters=3, kernel_size=3, padding='same'))
        self.G.add(Activation('sigmoid'))

        return self.G

    def DisModel(self):
        if self.DM is None:
            self.DM = Sequential()
            self.DM.add(self.discriminator())

        self.DM.compile(optimizer=Adam(lr=self.LR * (0.85 ** floor(self.steps / 10000))), loss='binary_crossentropy')

        return self.DM

    def AdModel(self):
        if self.AM is None:
            self.AM = Sequential()
            self.AM.add(self.generator())
            self.AM.add(self.discriminator())

        self.AM.compile(optimizer=Adam(lr=self.LR * (0.85 ** floor(self.steps / 10000))), loss='binary_crossentropy')

        return self.AM

    def sod(self):
        self.OD = self.D.get_weights()

    def lod(self):
        self.D.set_weights(self.OD)


class Model_GAN(object):
    def __init__(self):

        self.GAN = GAN()
        self.DisModel = self.GAN.DisModel()
        self.AdModel = self.GAN.AdModel()
        self.generator = self.GAN.generator()

    def train(self, batch=128):
        (a, b) = self.train_dis(batch)
        c = self.train_gen(batch)

        print(f"D Real: {str(a)}, D Fake: {str(b)}, G All: {str(c)}")

        if self.GAN.steps % 5000 == 0:
            self.GAN.AM = None
            self.GAN.DM = None
            self.AdModel = self.GAN.AdModel()
            self.DisModel = self.GAN.DisModel()

        self.GAN.steps = self.GAN.steps + 1

    def train_dis(self, batch):
        # Get Real Images
        im_no = random.randint(0, len(all_images) - batch - 1)
        train_data = all_images[im_no: im_no + int(batch / 2)]
        label_data = []
        for i in range(int(batch / 2)):
            # label_data.append(one())
            label_data.append(zero())

        d_loss_real = self.DisModel.train_on_batch(np.array(train_data), np.array(label_data))

        # Get Fake Images
        train_data = self.generator.predict(noise(int(batch / 2)))
        label_data = []
        for i in range(int(batch / 2)):
            # label_data.append(zero())
            label_data.append(one())

        d_loss_fake = self.DisModel.train_on_batch(train_data, np.array(label_data))

        return d_loss_real, d_loss_fake

    def train_gen(self, batch):

        self.GAN.sod()

        label_data = []
        for i in range(int(batch)):
            label_data.append(zero())

        g_loss = self.AdModel.train_on_batch(noise(batch), np.array(label_data))

        self.GAN.lod()

        return g_loss


len(files)
model = Model_GAN()

print("Starting training Loop...")
while model.GAN.steps < 500000:
    model.train()
    pred = model.GAN.G.predict(noise(1)).squeeze()
    x = Image.fromarray(np.uint8(pred * 255))
    x.save(f"data/DeGAN"
           f"/{model.GAN.steps - 1}.jpg")
