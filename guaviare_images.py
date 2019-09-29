import imgspectral as ims
import matplotlib.image as image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# Open Spectral Images
guav_l8 = ims.open("./imagery/guaviare_l8_l2", "landsat8")
guav_hy = ims.open("./imagery/guaviare_hy_l1", "hyperion")
guav_s2 = ims.open("./imagery/guaviare_s2_l1", "sentinel2")

# Make list of images for different sensors, 
# resize sentinel2 image for equal pixel resolution
guav_imgs = [guav_l8, guav_hy, guav_s2]
sensor_names = ["l8", "hy", "s2"]

# Areas of study (latitude, longitude, width, height)
area1 = (2.840, -73.0412, 8400, 8100)
area2 = (3.100, -72.9875, 8400, 8100)

# Select images subsets for areas of study
guav_area1 = [img.subset(area1) for img in guav_imgs]
guav_area2 = [img.subset(area2) for img in guav_imgs]

# Generate Natural RGB Images, and save Images
for i in range(len(guav_imgs)):
    area1_rgb = guav_area1[i].rgb()
    area2_rgb = guav_area2[i].rgb()
    image.imsave("./graphics/guav_area1_" + sensor_names[i] + ".png", area1_rgb)  
    image.imsave("./graphics/guav_area2_" + sensor_names[i] + ".png", area2_rgb)  

# Aply PCA to Multispectral image of Landsat8
l8_bands = guav_area1[0].bandsGroup([1,2,3,4,5,6,7])
l8_samples = l8_bands.reshape((l8_bands.shape[0]*l8_bands.shape[1], l8_bands.shape[2]))
pca = PCA(n_components=5)
pca.fit(l8_samples)
cum_explained_variance = np.cumsum(pca.explained_variance_ratio_)
print("\nLandsat 8 Explained Variance:")
print(cum_explained_variance)

n_components = np.argmax(cum_explained_variance > 0.98) + 1
print("\n N = {}".format(n_components))

pca = PCA(n_components=n_components)
l8_samples_pca = pca.fit_transform(l8_samples)

for i in range(n_components):
    img_pca = l8_samples_pca[:,i].reshape((l8_bands.shape[0], l8_bands.shape[1]))
    img_pca_color = ims.color(img_pca)
    image.imsave("./graphics/guav_area1_l8_pca_{}.png".format(i+1), img_pca_color, cmap='gray') 

# Generate NDVI and save Images with green color map
color_map = "Greens"
for i in range(len(guav_imgs)):
    area1_ndvi = guav_area1[i].ndvi()
    area2_ndvi = guav_area2[i].ndvi()
    image.imsave("./graphics/guav_area1_" + sensor_names[i] + "_ndvi.png", area1_ndvi, cmap=color_map)
    image.imsave("./graphics/guav_area2_" + sensor_names[i] + "_ndvi.png", area2_ndvi, cmap=color_map)

# Generate MSR and save Images with green color map
color_map = "Greens"
for i in range(len(guav_imgs)):
    area1_msr = guav_area1[i].msr()
    area2_msr = guav_area2[i].msr()
    image.imsave("./graphics/guav_area1_" + sensor_names[i] + "_msr.png", area1_msr, cmap=color_map)
    image.imsave("./graphics/guav_area2_" + sensor_names[i] + "_msr.png", area2_msr, cmap=color_map)

# Generate TVI and save Images with green color map
color_map = "Greens"
for i in range(len(guav_imgs)):
    area1_tvi = guav_area1[i].tvi()
    area2_tvi = guav_area2[i].tvi()
    image.imsave("./graphics/guav_area1_" + sensor_names[i] + "_tvi.png", area1_tvi, cmap=color_map)
    image.imsave("./graphics/guav_area2_" + sensor_names[i] + "_tvi.png", area2_tvi, cmap=color_map)
