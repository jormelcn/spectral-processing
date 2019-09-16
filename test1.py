from imgspectral import ImgSpectral
import matplotlib.image as image
import matplotlib.pyplot as plt

ImgSpectral.cache = True
ImgSpectral.verbose = True

# Guardar una imagen PNG
def saveImagePng(img, filePath, cmap=None):
    image.imsave(filePath, img, cmap=cmap)

imgL7_L1 = ImgSpectral("./imagery/LE07_L1TP_007058_20141220_20161030_01_T1.landsat7")
imgL7_L2 = ImgSpectral("./imagery/LE070070582014122001T1-SC20190912174145.landsat7")
imgL8_L1 = ImgSpectral("./imagery/LC08_L1TP_007058_20140110_20170426_01_T1.landsat8")
imgL8_L2 = ImgSpectral("./imagery/LC080070582014011001T1-SC20190912191355.landsat8")
imgS2 = ImgSpectral("./imagery/S2A_OPER_MSI_L1C_TL_EPA__20151208T152341_20170529T085215_A002410_T18NYJ_N02_04_01.sentinel2")
imgH = ImgSpectral("./imagery/EO1H0070582015062110KF.hyperion")

# rf = 1
# saveImagePng(imgL7_L1.resize(rf).rgb(), "imgL7_L1.png")
# saveImagePng(imgL7_L2.resize(rf).rgb(), "imgL7_L2.png")
# saveImagePng(imgL8_L1.resize(rf).rgb(), "imgL8_L1.png")
# saveImagePng(imgL8_L2.resize(rf).rgb(), "imgL8_L2.png")
# saveImagePng(imgS2.resize(rf).rgb(), "imgS2.png")
# saveImagePng(imgH.resize(rf).rgb(), "imgH.png")


imgs = [imgL8_L1, imgL8_L2, imgS2.resize(3), imgH]
imgs_names = ["imgL8_L1", "imgL8_L2", "imgS2", "imgH"]

ss1 = (2.840, -73.0412, 8400, 8100)
ss2 = (3.100, -72.9875, 8400, 8100)
ss1_imgs = [img.subset(ss1) for img in imgs]
ss2_imgs = [img.subset(ss2) for img in imgs]
for i in range(len(imgs)):
    saveImagePng(ss1_imgs[i].rgb(), "SS1_" + imgs_names[i] + ".png")  
    saveImagePng(ss2_imgs[i].rgb(), "SS2_" + imgs_names[i] + ".png")  

for i in range(len(imgs)):
    saveImagePng(ss1_imgs[i].ndvi(), "SS1_" + imgs_names[i] + "_NDVI.png", cmap="inferno")
    saveImagePng(ss2_imgs[i].ndvi(), "SS2_" + imgs_names[i] + "_NDVI.png", cmap="inferno")

plt.show()
    

