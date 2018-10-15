# Principal-Component-Analysis-on-Images
Application of Principal Component on Images.

An implementation of PCA on Images to apply reconstruction of images to find variance from the distribution, inspired by EigenFaces.

Given similar images, a PCA is applied on the dataset. Any image is reconstructed using the PCA result. If the image belongs to the dataset or is similar to the dataset, the variance between the original and reconstructed images is small, else it is huge.

Thus, PCA can be applied to determine if a given image resembles images in a dataset. This is particularly useful if we need to build a classifier that does not have negative samples. The accuracy is not great.

## Note
Since the project is not out to the public, the dataset is not provided. Any dataset of choice can be used instead, provided the dimensions of images are fixed.
