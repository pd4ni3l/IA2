from skimage import io, filters, color

image = io.imread('lena.jpg') # Load the image

io.imshow(image)
io.show()
print ('image matrix size: ', image.shape, image.dtype)# print the size of image

image_gray = color.rgb2gray(image)      # Convert the image to grayscale (1 channel)
io.imshow(image_gray)
io.show()
print ('image gray matrix size: ', image_gray.shape, image_gray.dtype)      # print the size of image
print ('\n First 5 columns and rows of the image gray matrix: \n', image_gray[:5,:5]*255) 

edges = filters.sobel(image_gray)

io.imshow(edges)
io.show()
print ('image gray matrix size: ', edges.shape, edges.dtype)      # print the size of image
print ('\n First 5 columns and rows of the image edges matrix: \n', edges[:5,:5]*255) 

