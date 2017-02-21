from PIL import Image, ImageDraw
import input_data
import numpy as np


def get_image(path):
    im = Image.open(path)
    im = im.resize((28, 28))
    im1 = im.convert('L')
    pix = im1.load()
    im1.save('../3_.jpg')
    width = im1.size[0]
    height = im1.size[1]
    dat = []
    for x in range(width):
        for y in range(height):
            r = pix[y, x]
            v = (255 - r) / 255.0
            dat.append(v)
    array_to_image(dat, '../test.jpg')
    data = np.array(dat)
    label = np.array([9])
    label = input_data.dense_to_one_hot(label)
    data = data.reshape(1, 28, 28, 1)
    image_data = input_data.DataSet(data, label)
    return image_data


def array_to_image(arr, filePath):
    im = Image.new('RGB', (28, 28))
    # draw = ImageDraw.Draw(im)

    for x in range(28):
        for y in range(28):
            v = int(255 - (arr[x * 28 + y] * 255))
            im.putpixel((y, x), (v, v, v))
    im.save(filePath)


image = get_image('../9_.jpg')
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
array_to_image(mnist.test.images[3], '../test1.jpg')
