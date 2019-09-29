from imgspectral import ImgSpectral
import matplotlib.image as image
import matplotlib.pyplot as plt
import numpy as np

ImgSpectral.cache = True


# Guardar una imagen PNG
def saveImagePng(img, filePath, cmap=None):
    image.imsave(filePath, img, cmap=cmap)

imgH = ImgSpectral("./imagery/EO1H0070582015062110KF.hyperion")

imgH_ss1 = imgH.subset((2.840, -73.0412, 8400, 8100))

red = imgH_ss1.band(29)
print(red)
nir = imgH_ss1.band(52)
print(nir)

# plt.figure()
# plt.imshow(red, cmap='gray')

# plt.figure()
# plt.imshow(nir, cmap='gray')

#plt.figure()
#ndvi = (nir - red)/(nir + red)
#plt.imshow(ndvi, cmap='inferno')

plt.figure()
plt.imshow(imgH_ss1.ndvi(), cmap='Greens')
plt.show()
