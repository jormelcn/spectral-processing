from imgspectral import ImgSpectral
import pickle

ImgSpectral.cache = True


imgH = ImgSpectral("./imagery/EO1H0070582015062110KF.hyperion")
imgH_ss1 = imgH.subset((2.840, -73.0412, 8400, 8100))
a = imgH_ss1.bandsGroup([29,13])

with open('filename.pkl', 'wb') as handle:
    pickle.dump(a.dataset, handle)

with open('filename.pkl', 'rb') as handle:
    b = pickle.load(handle)

print(a.dataset)
print(b)

