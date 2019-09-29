import imgspectral as ims
import matplotlib.image as image

ims.ImgSpectral.verbose = True

guav_l8 = ims.open("./imagery/guaviare_l8_l2", "landsat8")
guav_hy = ims.open("./imagery/guaviare_hy_l1", "hyperion")
guav_s2 = ims.open("./imagery/guaviare_s2_l1", "sentinel2")
barr_s2 = ims.open("./imagery/barrancabermeja_s2_l1", "sentinel2")
barr_hy = ims.open("./imagery/barrancabermeja_hy_l1", "hyperion")

names = ["guaviare_l8", "guaviare_hy", "guaviare_s2", "barrancabermeja_s2", "barrancabermeja_hy"]
images = [guav_l8.resize(3), guav_hy.resize(3), guav_s2.resize(9), barr_s2.resize(9), barr_hy.resize(3)]

for i in range(len(images)):
    image.imsave("./graphics/{}.png".format(names[i]), images[i].rgb())    
