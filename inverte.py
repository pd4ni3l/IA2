from skimage import util
from skimage import data, io, filters


img = data.camera()
inverted_img = util.invert(img)

io.imshow(img)
io.show()

io.imshow(inverted_img)
io.show()