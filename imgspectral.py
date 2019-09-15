import numpy as np
import gdal as gd
from pathlib import Path
from osgeo import ogr
from osgeo import osr
from collections.abc import Sequence

class ImgBand(np.ndarray):

    def __make__(self, dataset, resizeFactor, subset, subsetMode, verbose):
        self.verbose = verbose
        self.dataset = dataset
        self.resizeFactor = int(resizeFactor)
        self.subset = subset
        self.subsetMode = subsetMode
        self.__generateCoordinateSystem__()
        self.__calculateShapes__()
        self.__verifyShapes__()
    
    @staticmethod
    def make(dataset, resizeFactor, subset=None, subsetMode = 'latlon', verbose=False, cache=None):
        if cache is None:
            imgBand = ImgBand(0)
            imgBand.__make__(dataset, resizeFactor, subset, subsetMode, verbose)    
            imgBand.resize(imgBand.factorShape, refcheck=False)
            imgBand.__load__()
        else:
            imgBand = ImgBand(cache.shape, dtype=cache.dtype, buffer=cache, offset=0)
            imgBand.__make__(dataset, resizeFactor, subset, subsetMode, verbose)
        return imgBand
    
    def __generateCoordinateSystem__(self):
        self.geoTransform = self.dataset.GetGeoTransform()
        source = osr.SpatialReference()
        source.ImportFromWkt(self.dataset.GetProjection())
        target = source.CloneGeogCS()
        self.transToLatLon = osr.CoordinateTransformation(source, target)
        self.transToXY = osr.CoordinateTransformation(target, source)    

    def __xy2pixel__(self, xy):
        gt = self.geoTransform
        return (int((xy[0] - gt[0])/gt[1]), int((xy[1] - gt[3])/gt[5]))

    def __width2size__(self, l):
        return   int(l/abs(self.geoTransform[1]))

    def __height2size__(self, l):
        return   int(l/abs(self.geoTransform[5]))

    def __calculateShapes__(self):
        rasterShape = np.array((self.dataset.RasterYSize, self.dataset.RasterXSize))
        self.rasterShape = tuple(rasterShape)
        if self.subset is None:
            factorShape = np.floor(rasterShape/self.resizeFactor).astype(int)
            self.factorShape = tuple(factorShape)
            self.realSubset = (0,0) + tuple((factorShape*self.resizeFactor).astype(int))[::-1]
            return
        if self.subsetMode == 'latlon':
            ss_center = self.__xy2pixel__(self.transToXY.TransformPoint(self.subset[1], self.subset[0]))
            ss_width = self.__width2size__(self.subset[2])
            while ss_width%2 == 0 or ss_width%self.resizeFactor != 0:
                ss_width += 1
            ss_heigth = self.__height2size__(self.subset[3])
            while ss_heigth%2 == 0 or ss_heigth%self.resizeFactor != 0:
                ss_heigth += 1
            self.factorShape = (int(ss_heigth/self.resizeFactor), int(ss_width/self.resizeFactor))
            self.realSubset = (ss_center[0]-(ss_heigth-1)/2, ss_center[1]-(ss_width-1)/2) + (ss_width, ss_heigth)
            self.__print__("Real subset: {}".format(str(self.realSubset)))
        else :
            raise ValueError("Invalid subset Mode")

    def __verifyShapes__(self):
        pass

    def __areaMean__(self, a):
        f = self.resizeFactor
        a0 = a.reshape((a.shape[0], a.shape[1], int(a.size/(a.shape[0]*a.shape[1]))), order='F')
        a1 = a.reshape((a0.shape[0], f, int(a0.shape[1]*a0.shape[2]/f)), order='F')
        a2 = np.transpose(a1,axes=[1,0,2]).reshape((int(f**2),int(a0.size/(f**2))), order='F')
        a3 = a2.mean(axis=0).reshape((int(a0.shape[0]/f), int(a0.shape[1]/f), a0.shape[2]), order='F')
        return a3.reshape(a3.shape[0:len(a.shape)])

    def __refactor__(self, raw, row):
        self[row : row + int(raw.shape[0]/self.resizeFactor),:] = self.__areaMean__(raw)

    def __print__(self, str, end='\n'):
        if self.verbose:
            print(str, end=end)

    def __load__(self):
        max_load_memory = 1250000
        load_block_rows = int(max_load_memory/self.realSubset[2])
        load_block_rows -= load_block_rows%self.resizeFactor
        load_block = [min(int(load_block_rows), self.realSubset[3]), self.realSubset[2]]
        row = int(0)
        while row < self.factorShape[0]:
            raw = self.dataset.ReadAsArray( int(self.realSubset[0]), 
                                            int(self.realSubset[1] + row*self.resizeFactor), 
                                            int(load_block[1]), int(load_block[0]))
            self.__refactor__(raw, row)
            row += int(load_block[0]/self.resizeFactor)    
            if row*self.resizeFactor + load_block[0] > self.rasterShape[0]:
                load_block[0] = (self.factorShape[0] - row)*self.resizeFactor
            loaded = (row/self.factorShape[0])
            self.__print__('  loaded {:.0f} %'.format(loaded*100.0), end = '\r')
        self.__print__("                                          ", end='\r')
        

class ImgBandsGroup(ImgBand):

    @staticmethod
    def make(datasets, resizeFactor, subset=None, subsetMode='latlon', verbose=False, cache = None):
        ImgBandsGroup.__verifyDatasets__(datasets)
        if cache is None:
            imgBandsGroup = ImgBandsGroup(0)
            imgBandsGroup.datasets = datasets
            imgBandsGroup.__make__(datasets[0], resizeFactor, subset, subsetMode, verbose)
            imgBandsGroup.resize(imgBandsGroup.factorShape + (len(datasets),), refcheck=False)
            imgBandsGroup.__load__()
        else:
            imgBandsGroup = ImgBandsGroup(cache.shape, dtype=cache.dtype, buffer=cache, offset=0)
            imgBandsGroup.datasets = datasets
            imgBandsGroup.__make__(datasets[0], resizeFactor, subset, subsetMode, verbose)
        return imgBandsGroup

    @staticmethod
    def __verifyDatasets__(datasets):
        pass

    def __refactor__(self, raw, row, band):
        self[row : row + int(raw.shape[0]/self.resizeFactor),:,band] = self.__areaMean__(raw)

    def __load__(self):
        max_load_memory = 1250000
        load_block_rows = int(max_load_memory/self.realSubset[2])
        load_block_rows -= load_block_rows%self.resizeFactor
        for band in range(len(self.datasets)):
            dataset = self.datasets[band]
            load_block = [min(int(load_block_rows), self.realSubset[3]), self.realSubset[2]]
            row = int(0)
            while row < self.factorShape[0]:
                raw = dataset.ReadAsArray(  int(self.realSubset[0]), 
                                            int(self.realSubset[1] + row*self.resizeFactor), 
                                            int(load_block[1]), int(load_block[0]))
                self.__refactor__(raw, row, band)
                row += int(load_block[0]/self.resizeFactor)    
                if row*self.resizeFactor + load_block[0] > self.rasterShape[0]:
                    load_block[0] = (self.factorShape[0] - row)*self.resizeFactor
                loaded = (band + row/self.factorShape[0])/len(self.datasets)
                self.__print__('  loaded {:.0f} %'.format(loaded*100.0), end = '\r')
        self.__print__("                                              ", end='\r')
            


class ImgSpectral:
#********************************* Default Config ************************************
    validExtensions = ['sentinel2', 'landsat8', 'landsat7', 'hyperion']
    extensionsConfig = {
        'hyperion' : {
            'bandsExtensions' : ['tif', 'TIF'],
            'nameSeparator' : '_',
            'bandsPrefixes' : ['band', 'b', 'B', 'BAND', 'B0', 'BAND0', 'B00', 'BAND00'],
            'rgbBands' : ['29', '23', '16'],
            'nirBand' : '52'
        },
        'sentinel2' : {
            'bandsExtensions' : ['jp2', 'JP2'],
            'nameSeparator' : '_',
            'bandsPrefixes' : ['band', 'b', 'B', 'BAND', 'B0', 'BAND0'],
            'rgbBands' : ['4', '3', '2'],
            'nirBand' : '8'
        },
        'landsat8' : {
            'bandsExtensions' : ['tif', 'TIF'],
            'nameSeparator' : '_',
            'bandsPrefixes' : ['band', 'b', 'B', 'BAND', 'B0', 'BAND0'],
            'rgbBands' : ['4', '3', '2'],
            'nirBand' : '5'
        },
        'landsat7' : {
            'bandsExtensions' : ['tif', 'TIF'],
            'nameSeparator' : '_',
            'bandsPrefixes' : ['band', 'b', 'B', 'BAND', 'B0', 'BAND0'],
            'rgbBands' : ['3', '2', '1'],
            'nirBand' : '4'
        },
    }

    resizeFactor = 1
    cache = True
    verbose = False

    RGBDinamicRange = .93
    colorSpacePower = .5

    linearSpace = np.linspace(0, 1, 100)
    colorSpace = 2 - 2/(linearSpace**colorSpacePower + 1)

#********************************* Constructor **************************************

    def __init__(self, path, resizeFactor = None, cache = None, subset= None, subsetMode='latlon') :
        path_split = path.split('.')
        self.path = path
        self.resizeFactor = resizeFactor if not resizeFactor is None else ImgSpectral.resizeFactor
        self.cache = cache if not (cache is None) else ImgSpectral.cache
        self.extension = path_split[-1]
        if not(self.extension in ImgSpectral.validExtensions) :
            raise ValueError('Invalid Path Extension')
        self.pathFolder = Path('.'.join(path_split[0:-1]))
        if not self.pathFolder.exists() :
            raise FileNotFoundError('Data Folder Not Exist') 
        self.config = ImgSpectral.extensionsConfig[self.extension]
        self.__subset = subset
        self.__subsetMode = subsetMode
        self.__selectBandsFiles__()
        cacheFolderName = str(self.resizeFactor)
        self.pathCahe = self.pathFolder/'.imgspectral/cache'/cacheFolderName
        self.__bandsGroups__ = {}
        self.datasets = {}

#**********************************  User Methods ***********************************
    def resize(self, resizeFactor):
        return ImgSpectral(self.path, resizeFactor, self.cache, self.__subset, self.__subsetMode)

    def subset(self, subset, subsetMode='latlon'):
        return ImgSpectral(self.path, self.resizeFactor, self.cache, subset, subsetMode)

    def bandsGroup(self, bands):
        if len(bands) < 1:
            raise ValueError("Not can load zero bands")
        bands = [str(band) for band in bands]
        group = self.__groupName__(bands)
        if not group in self.__bandsGroups__:
            ds =  self.__openBandsGroup__(bands)
            if len(bands) == 1:
                self.__bandsGroups__[group] = self.__loadBand__(group, ds[0])
            else:
                self.__bandsGroups__[group] = self.__loadBandsGroup__(group, ds)
        return self.__bandsGroups__[group]

    def band(self, band):
        return self.bandsGroup([band])

    def rgb(self, bands = None):
        bands = self.config["rgbBands"] if bands is None else bands
        return ImgSpectral.color(self.bandsGroup(bands))

    def ndvi(self, bands = None):
        bands = [self.config["nirBand"], self.config["rgbBands"][0]] if bands is None else bands
        nir, red = self.band(bands[0]), self.band(bands[1])
        ndvi = np.ndarray(nir.shape)
        np.subtract(nir,red, out=ndvi)
        np.add(nir, red, out=nir)
        np.divide(ndvi, nir, out=ndvi)
        ndvi[np.isnan(ndvi)] = 0
        return ndvi
    
    # def pixelLatLon(self, x, y):
    #     gt = self.geoTransform
    #     _x, _y = gt[0] + x*gt[1] + y*gt[2], gt[3] + x*gt[4] + y*gt[5]
    #     lon, lat, _ = self.transToLatLon.TransformPoint(_x, _y)
    #     return (lat, lon)        
    
    @staticmethod
    def color(reflectance):
        if ImgSpectral.verbose:
            print("Processing Color...")
        _img = reflectance.astype(np.float16)
        ImgSpectral.__normalice__(_img)
        np.power(_img, np.array(ImgSpectral.colorSpacePower, dtype=np.float16),out=_img)
        np.add(_img, np.array(1, dtype=np.float16), out=_img)
        np.power(_img, np.array(-1, dtype=np.float16), out=_img)
        np.multiply(_img, np.array(2, dtype=np.float16), out=_img)
        np.subtract(np.array(2, dtype=np.float16), _img, out=_img)
        return _img

    @staticmethod
    def composite(*rgb):
        if len(rgb) == 1 and len(rgb[0].shape) == 3 and rgb[0].shape[2] == 3:
            return ImgSpectral.color(rgb[0])
        elif len(rgb) == 3:
            return ImgSpectral.color(np.dstack((rgb[0], rgb[1], rgb[2])))


#******************************** Intern Methods ************************************

    def __groupName__(self, bands):
        bands_name = '_'.join(bands)
        subset_name = "None" if self.__subset is None else '_'.join([str(i) for i in self.__subset])
        return bands_name + "x" + subset_name + "x" + self.__subsetMode

    def __loadBand__(self, cachePathFile, dataset):
        band = None
        if self.cache :
            band = self.__loadBandFromCache__(cachePathFile, dataset)
        if band is None:
            self.__print__('Loading %s %s image from Dataset...' % (self.pathFolder.name, cachePathFile))
            band = ImgBand.make(dataset, self.resizeFactor, self.__subset, self.__subsetMode, self.verbose)
            if self.cache :
                self.__saveCache__(cachePathFile, band)
        return band

    def __loadBandsGroup__(self, cachePathFile, datasetGroup):
        bandsGroup = None
        if self.cache :
            bandsGroup = self.__loadGroupFromCache__(cachePathFile, datasetGroup)
        if bandsGroup is None:
            self.__print__('Loading %s %s image from Dataset...' % (self.pathFolder.name, cachePathFile))
            bandsGroup = ImgBandsGroup.make(datasetGroup, self.resizeFactor, self.__subset, self.__subsetMode, self.verbose)
            if self.cache :
                self.__saveCache__(cachePathFile, bandsGroup)
        return bandsGroup

    def __print__(self, message, end = '\n'):
        if ImgSpectral.verbose:
            print(message, end=end)

    @staticmethod
    def __normalice__(_img):
        _nan = max(-100, np.min(_img))
        _min = np.min(_img[_img > _nan])
        _max = np.max(_img)
        np.subtract(_img, _min, out=_img)
        af = 1/_max if _min == _max else 1/(_max - _min)
        np.multiply(_img, np.array(af, dtype=np.float16) , out=_img)
        np.clip(_img, 0, 1, out=_img)
        return _img

    def __openBandsGroup__(self, bands):
        datasets = []
        for band in bands:
            if not band in self.datasets:
                bandPath = self.__findBandPathFile__(band)
                if bandPath is None:
                    raise ValueError('Error: %s band %s not Found...' % (self.pathFolder.name, band)) 
                self.datasets[band] = gd.Open(str(self.pathFolder/bandPath))
            datasets.append(self.datasets[band]) 
        return datasets

    def __selectBandsFiles__(self):
        self.bandsFiles = []
        for ext in self.config['bandsExtensions']:
            paths = list(self.pathFolder.glob('**/*.' + ext))
            for path in paths:
                self.bandsFiles.append(path.name.split('.'))    
        
    def __findBandPathFile__(self, bandName):
        for name, ext in self.bandsFiles:
            for part in name.split(self.config['nameSeparator']):
                for prefix in self.config['bandsPrefixes']:
                    if prefix + bandName == part:
                        return name + '.' + ext
        return None

    def __loadBandFromCache__(self, filePath, dataset) :
        if not (self.pathCahe/(filePath + '.npy')).exists() :
            return None
        self.__print__('Loading %s %s data from Cache' % (self.pathFolder.name, filePath))
        cache = np.load(self.pathCahe/(filePath + '.npy'))
        return ImgBand.make(dataset, self.resizeFactor, self.__subset, self.__subsetMode, self.verbose, cache)

    def __loadGroupFromCache__(self, filePath, datasets) :
        if not (self.pathCahe/(filePath + '.npy')).exists() :
            return None
        self.__print__('Loading %s %s data from Cache' % (self.pathFolder.name, filePath))
        cache = np.load(self.pathCahe/(filePath + '.npy'))
        return ImgBandsGroup.make(datasets, self.resizeFactor, self.__subset, self.__subsetMode, self.verbose, cache)
        
    def __saveCache__(self, filePath, data) :
        if not self.pathCahe.exists() : 
            self.pathCahe.mkdir(parents = True, exist_ok=True)
        np.save(self.pathCahe/(filePath + '.npy'), data)
