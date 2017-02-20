from PIL import Image
im = Image.open('/home/lucas/Desktop/9_.jpg')
im1 = im.convert('L')
pix = im1.load()

width = im1.size[0]
height = im1.size[1]
for x in range(width):
    for y in range(height):
        r = pix[x, y]
        print (float)(255-r)/255.0