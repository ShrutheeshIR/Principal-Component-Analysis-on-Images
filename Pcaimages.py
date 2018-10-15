import cv2
import os
import numpy as np
import scipy.linalg as sl


# Read Images into Array

def readImages(dirName):
    matr = []
    for i,file in enumerate(os.listdir(dirName)):
        if i<3000:
            #print(file)
            #print(i)
            img = cv2.imread(str(dirName + "/" + file),0)
            img = cv2.resize(img, (64,64))
            #print(img)
            #print(img.shape)
            matr.append(img)
            #x = np.array(matr)
            #print(x.shape)
    return matr

#Create a data Matrix

def createDataMatrix(images):
    print("Creating data matrix",end=" ... ")
    ''' 
    Allocate space for all images in one data matrix. 
        The size of the data matrix is
        ( w  * h  * 3, numImages )
        
        where,
         
        w = width of an image in the dataset.
        h = height of an image in the dataset.
        3 is for the 3 color channels.
        '''
   
    numImages = len(images)
    sz = images[0].shape
    data = np.zeros((numImages, sz[0] * sz[1]), dtype=np.float32)
    for i in range(0, numImages):
        image = images[i].flatten()
        data[i,:] = image
     
    print("DONE")
    return data


# Directory containing images
dirName = "GImagesTrain/GoodImages"

# Read images
images = readImages(dirName)

# Size of images
sz = images[0].shape

# Create data matrix for PCA.
data = createDataMatrix(images)

sigm = np.cov(data, rowvar = False)
print(sigm.shape)



# Find eigen values and eigen vectors of the matrix

eigenValues, u = sl.eig(sigm)

# Sort by eigenvalues

idx = eigenValues.argsort()[::-1]   
eigenValues = eigenValues[idx]





u = u[:,idx]



# Read a random Image and reconstruct the image 

x = data[250]






x = np.matrix(x)
x = x.transpose()




ims = x.reshape(64, 64)
ims = np.array(ims)
print(ims.shape)
#print(ims)



# Read the first 75 eigenvectors

u = u.transpose()
eigenvectors = u[:75].real
print(eigenvectors.shape)

z = eigenvectors*x

print(z.shape)


#Reconstruct the image

reconstucted = np.matmul(eigenvectors.transpose(),z)


recons = reconstucted.reshape(64,64)
print(reconstucted.shape)





import matplotlib.pyplot as plt

recons /= 255
ims /= 255

#cv2.namedWindow('image',cv2.WINDOW_NORMAL)
#cv2.resizeWindow('image', 600,600)
#cv2.imshow('Original', ims)

#cv2.namedWindow('image',cv2.WINDOW_NORMAL)
#cv2.resizeWindow('image', 600,600)
#cv2.imshow('Reconstructed', recon2)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

print(recons)
#print(ims)


plt.imshow(recons)
plt.show()
plt.imshow(ims)
plt.show()
