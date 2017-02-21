from PIL import Image
import input_data
import numpy as np
im = Image.open('../9_.jpg')
im1 = im.convert('L')
pix = im1.load()

width = im1.size[0]
height = im1.size[1]
dat = []
for x in range(width):
    for y in range(height):
        r = pix[x, y]
        v = (float)(255-r)/255.0
        dat.append(v)

data = np.array(dat)
label = np.array([9])
data = data.reshape(1,28,28,1)
print data