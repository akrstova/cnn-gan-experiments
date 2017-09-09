from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
import numpy as np
from skimage import color, exposure, transform
from skimage import io

import matplotlib.pyplot as plt


class GAN(object):
    def __init__(self, img_rows=128, img_columns=128, channel=1):

        self.img_rows = img_rows
        self.img_columns = img_columns
        self.channel = channel
        self.D = None  # discriminator
        self.G = None  # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model

    def discriminator(self):
        if self.D:
            return self.D
        self.D = Sequential()
        depth = 64
        dropout = 0.4
        input_shape = (self.img_rows, self.img_columns, self.channel)
        self.D.add(Conv2D(depth * 1, 5, strides=2, input_shape=input_shape,\
                          padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth * 2, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth * 4, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth * 8, 5, strides=1, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        # Output: 1 probability score (image is fake or real)
        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        self.D.summary()
        return self.D

    def generator(self):
        if self.G:
            return self.G
        self.G = Sequential()
        dropout = 0.4
        depth = 64 + 64 + 64 + 64
        dim = 7
        # In: 100
        # Out: img_rows x img_columns x depth
        self.G.add(Dense(dim * dim * depth, input_dim=128))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        self.G.add(Reshape((dim, dim, depth)))
        self.G.add(Dropout(dropout))

        # In: img_rows x img_columns x depth
        # Out: 2 * img_rows x 2 * img_columns x depth
        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth / 2), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth / 4), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(Conv2DTranspose(int(depth / 8), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        # Out: 28 x 28 x 1 grayscale image 
        self.G.add(Conv2DTranspose(1, 5, padding='same'))
        self.G.add(Activation('sigmoid'))
        self.G.summary()
        return self.G

    def discriminator_model(self):
        if self.DM:
            return self.DM
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        # The discriminator is trying to solve a binary classification problem:
        # Whether the input image is fake or not
        self.DM.compile(loss='binary_crossentropy', optimizer='adam', \
                        metrics=['accuracy'])
        return self.DM

    def adversarial_model(self):
        if self.AM:
            return self.AM
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator())
        self.AM.compile(loss='binary_crossentropy', optimizer='adam', \
                        metrics=['accuracy'])

        return self.AM


class ImageNetGAN(object):
    def __init__(self):
        self.img_rows = 28
        self.img_columns = 28
        self.channel = 1

        # Try to generate images of apples
        root_dir = "D:/ImageNet/dataset/training_set/second_attempt/apple"
        imgs = []
        labels = []

        # First on two images as originals
        all_img_paths = ["D:/ImageNet/dataset/training_set/second_attempt/apple/n07742012_3153.jpeg",
                         "D:/ImageNet/dataset/training_set/second_attempt/apple/n07742012_3180.jpeg"]
        for img_path in all_img_paths:
            img = self.preprocess_img(io.imread(img_path))
            label = 1
            imgs.append(img)
            labels.append(label)

        X = np.array(imgs, dtype='float32')
        # Make encoded targets (label "apple" is encoded to 1)
        Y = np.eye(1, dtype='uint8')

        self.x_train = X

        # Initialize the model
        self.GAN = GAN()
        self.discriminator = self.GAN.discriminator_model()
        self.adversarial = self.GAN.adversarial_model()
        self.generator = self.GAN.generator()

    # Because the generated images will be grayscale, it makes sense to turn the originals to grayscale as well
    def preprocess_img(self, img):
        # Histogram normalization in v channel
        hsv = color.rgb2hsv(img)
        hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
        img = color.hsv2rgb(hsv)

        # Central square crop
        min_side = min(img.shape[:-1])
        # Floor division to find halves of the side lengths
        centre = img.shape[0] // 2, img.shape[1] // 2
        img = img[centre[0] - min_side // 2:centre[0] + min_side // 2,
              centre[1] - min_side // 2:centre[1] + min_side // 2,
              :]

        # Rescale to standard size
        img = transform.resize(img, (128, 128))

        # Roll color axis to axis 0
        img = np.rollaxis(img, -1)
        return img

    def train(self, train_steps=2, batch_size=2, save_interval=0):
        noise_input = None
        if save_interval > 0:
            noise_input = np.random.uniform(-1.0, 1.0, size=[128, 128])
        for i in range(train_steps):
            images_train = self.x_train[np.random.randint(0, self.x_train.shape[0], size=batch_size), :, :, :]
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 128])

            images_fake = self.generator.predict(noise)

            x = np.concatenate((images_train, images_fake))
            y = np.ones([2*batch_size, 1])
            y[batch_size:, :] = 0
            d_loss = self.discriminator.train_on_batch(x, y)

            y = np.ones([batch_size, 1])
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 128])
            a_loss = self.adversarial.train_on_batch(noise, y)
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)
            if save_interval>0:
                if (i+1)%save_interval==0:
                    self.plot_images(save_to_file=True, samples=noise_input.shape[0], \
                             noise=noise_input, step=(i + 1))

    # Helper function to display the generated images
    def plot_images(self, save_to_file=False, fake=True, samples=16, noise=None, step=0):
        filename = 'image_gen.png'
        if fake:
            if noise is None:
                noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
            else:
                filename = "image_%d.png" % step
            images = self.generator.predict(noise)
        else:
            i = np.random.randint(0, self.x_train.shape[0], samples)
            images = self.x_train[i, :, :, :]

        plt.figure(figsize=(10, 10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i + 7)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.img_rows, self.img_columns])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        if save_to_file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()


if __name__ == '__main__':
    ImageNetGAN = ImageNetGAN()
    ImageNetGAN.train(train_steps=2, batch_size=1, save_interval=500)
    ImageNetGAN.plot_images(fake=True)
    ImageNetGAN.plot_images(fake=False, save_to_file=True)