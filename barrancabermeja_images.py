import imgspectral as ims
import matplotlib.image as image
import numpy as np
from sklearn.decomposition import PCA

ims.ImgSpectral.cache = False

# Open Spectral Images
barr_hy = ims.open("./imagery/barrancabermeja_hy_l1", "hyperion")
barr_s2 = ims.open("./imagery/barrancabermeja_s2_l1", "sentinel2")

# Make list of images for different sensors
barr_imgs = [barr_hy, barr_s2]
sensor_names = ["hy", "s2"]

# Areas of study (latitude, longitude, width, height)
area1 = (7.0405,-73.875, 8400, 7200)

# Select images subsets for areas of study
barr_area1 = [img.subset(area1) for img in barr_imgs]

# Show single band on gray scale
barr_hy_b40 = barr_area1[0].band(40)        # Load band 40 of hyperion
barr_s2_b8 = barr_area1[1].band(8)          # Load band 8 of sentinel2
barr_hy_b40_color = ims.color(barr_hy_b40)  # Color correction
barr_s2_b8_color = ims.color(barr_s2_b8)    # Color correction
image.imsave("./graphics/barr_area1_hy_b40.png", barr_hy_b40_color, cmap='gray')
image.imsave("./graphics/barr_area1_s2_b8.png", barr_s2_b8_color, cmap='gray')

# Generate Natural RGB Images, and save Images
for i in range(len(barr_imgs)):
    area1_rgb = barr_area1[i].rgb()
    image.imsave("./graphics/barr_area1_" + sensor_names[i] + ".png", area1_rgb)  

# Generate RGB Images with selected bands, and save Images
area1_rgb = barr_area1[1].rgb(bands=[3, 4, 2])
image.imsave("./graphics/barr_area1_custom_rgb" + sensor_names[1] + ".png", area1_rgb)

# Generate Infrared Color RGB Images, and save Images
for i in range(len(barr_imgs)):
    area1_rgb = barr_area1[i].infrared()
    image.imsave("./graphics/barr_area1_" + sensor_names[i] + "_cir.png", area1_rgb)  

# Generate gray scale images of hyperion bands, for select one noise band
for i in range(1,243):
    hy_band = barr_area1[0].band(i)
    hy_band_color = ims.color(hy_band)
    image.imsave("./graphics/barr_area1_hy_b{:03}.png".format(i), hy_band_color, cmap='gray')

# Filter selected bands
selected_bands = [8, 57, 82]
filters = ["gaussian_1", "gaussian_2", "median_3", "median_5", "uniform_3", "uniform_5"]
for i in selected_bands:
    hy_band = barr_area1[0].band(i)
    #Scaling Hyperion Data [0, 2*16] -> [0,1]
    np.divide(hy_band, 2**16, out = hy_band)        
    hy_band_gaussian_1 = hy_band.gaussian_filter(sigma=1, apply = False)
    hy_band_gaussian_2 = hy_band.gaussian_filter(sigma=2, apply = False)
    hy_band_median_3 =   hy_band.median_filter(size=3, apply = False)
    hy_band_median_5 =   hy_band.median_filter(size=5, apply = False)
    hy_band_uniform_3 =  hy_band.uniform_filter(size=3, apply = False)
    hy_band_uniform_5 =  hy_band.uniform_filter(size=5, apply = False)
    filtered_bands = [  hy_band_gaussian_1, hy_band_gaussian_2, 
                        hy_band_median_3,   hy_band_median_5,
                        hy_band_uniform_3,  hy_band_uniform_5]
    print("\nHyperion Band {}".format(i))
    for j in range(len(filters)):
        rmse = ((filtered_bands[j]- hy_band)**2).mean()**0.5
        print("{} filter, \trmse = {:.3}".format(filters[j], rmse))
        img_color = ims.color(filtered_bands[j])
        image.imsave("./graphics/barr_area1_hy_b{:03}_{}.png".format(i, filters[j]), img_color, cmap='gray')

# Aply PCA to hyperspectral image of hyperion
# Select bands [8, 57] + [77, 119] + [189, 217]
bands = list(range(8,58)) + list(range(77,120)) + list(range(189,218))  
hy_bands = barr_area1[0].bandsGroup(bands)
np.divide(hy_bands, 2**16, out = hy_bands) #Scaling Hyperion Data [0, 2*16] -> [0,1]        
hy_samples = hy_bands.reshape((hy_bands.shape[0]*hy_bands.shape[1], hy_bands.shape[2]))
pca = PCA(n_components=5)
pca.fit(hy_samples)
cum_explained_variance = np.cumsum(pca.explained_variance_ratio_)
print("\nHyperion Explained Variance:")
print(cum_explained_variance)

n_components = np.argmax(cum_explained_variance > 0.98) + 1
print("\n N = {}".format(n_components))

pca = PCA(n_components=n_components)
hy_samples_pca = pca.fit_transform(hy_samples)

for i in range(n_components):
    img_pca = hy_samples_pca[:,i].reshape((hy_bands.shape[0], hy_bands.shape[1]))
    img_pca_color = ims.color(img_pca)
    image.imsave("./graphics/barr_area1_hy_pca_{}.png".format(i+1), img_pca_color, cmap='gray')  

# Aply PCA to Multispectral image of sentinel 2
band1 = ims.resample(barr_area1[1].band(1), 6)      # Resample band 1 for 6
band2 = barr_area1[1].band(2)
band3 = barr_area1[1].band(3)
band4 = barr_area1[1].band(4)
band5 = ims.resample(barr_area1[1].band(5), 2)      # Resample band 5 for 2
band6 = ims.resample(barr_area1[1].band(6), 2)      # Resample band 6 for 2
band7 = ims.resample(barr_area1[1].band(7), 2)      # Resample band 7 for 2
band8 = barr_area1[1].band(8)
band8a = ims.resample(barr_area1[1].band('8A'), 2)  # Resample band 8a for 2
band9 = ims.resample(barr_area1[1].band(9), 6)      # Resample band 9  for 6
band10 = ims.resample(barr_area1[1].band(10), 6)    # Resample band 10 for 6
band11 = ims.resample(barr_area1[1].band(11), 2)    # Resample band 11 for 2
band12 = ims.resample(barr_area1[1].band(12), 2)    # Resample band 12 for 2

bands = [   band1, band2, band3, band4, band5, band6, band7, band8, 
            band8a, band9, band10, band11, band12]
bands = [ims.cut(band, band2.shape) for band in bands]

s2_bands = np.dstack(bands)

s2_samples = s2_bands.reshape((s2_bands.shape[0]*s2_bands.shape[1], s2_bands.shape[2]))
pca = PCA(n_components=5)
pca.fit(s2_samples)
cum_explained_variance = np.cumsum(pca.explained_variance_ratio_)
print("\nSentinel 2 Explained Variance:")
print(cum_explained_variance)

n_components = np.argmax(cum_explained_variance > 0.98) + 1
print("\n N = {}".format(n_components))

pca = PCA(n_components=n_components)
s2_samples_pca = pca.fit_transform(s2_samples)

for i in range(n_components):
    img_pca = s2_samples_pca[:,i].reshape((s2_bands.shape[0], s2_bands.shape[1]))
    img_pca_color = ims.color(img_pca)
    image.imsave("./graphics/barr_area1_s2_pca_{}.png".format(i+1), img_pca_color, cmap='gray')  


# Generate NDVI and save Images with green color map
color_map = "Greens"
for i in range(len(barr_imgs)):
    area1_ndvi = barr_area1[i].ndvi()
    image.imsave("./graphics/barr_area1_" + sensor_names[i] + "_ndvi.png", area1_ndvi, cmap=color_map)
