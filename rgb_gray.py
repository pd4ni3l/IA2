from skimage.color import rgb2gray
from skimage import data,io

img = data.astronaut()
img_gray = rgb2gray(img)


io.imshow(img)
io.show()
io.imshow(img_gray)
io.show()
