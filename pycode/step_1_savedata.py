import numpy as np
import bisect
import math
from getAtlMeasuredSwath_auto import getAtlMeasuredSwath
from getAtlTruthSwath_auto import getAtlTruthSwath
from getMeasurementError_auto import getMeasurementError, offsetsStruct
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

atl03FilePath = 'G:\\xcsdata\\virgin\\ICEsat2\\ATL03_20201213060025_12230901_004_01.h5'
gtNums = ['gt1l','gt1r']
daytime = atl03FilePath[-33:-25]
atl08FilePath = []

outFilePath = 'G:\\xcsdata\\virgin\\ICEsat2\\' # 文件存放位置
trimInfo = 'auto'  # OPTIONS: ('none', 'auto', or 'manual')

# none - does not trim data at all
# auto - trims ATL03 track to extent of truth bounding region
# manual - trims ATL03 track by latitude or time
# Example: 'manual,lat,38,39'
# Only uses data between latitude 38 and 39 deg
# Example: 'manual,time,3,4'
# Only uses data between time 3 and 4 seconds

createAtl03LasFile = False  # Option to create output measured ATL03 .las file
createAtl03KmlFile = False  # Option to create output measured ATL03 .kml file
createAtl03CsvFile = False  # Option to create output measured ATL03 .csv file
createAtl08KmlFile = False  # Option to create output measured ATL08 .kml file
createAtl08CsvFile = False  # Option to create output measured ATL08 .csv file

atlMeasuredDataAll = []
atlTruthDataAll = []
atlCorrectionsAll = []

for i in range(0, len(gtNums)):
    gtNum = gtNums[i]
    n_time = outFilePath+'it' + atl03FilePath[11:15] + daytime + str(gtNum) + '.npy'
    n_at =  outFilePath+'iat' + atl03FilePath[11:15] + daytime + str(gtNum) + '.npy'
    n_height =  outFilePath+'ih' + atl03FilePath[11:15] + daytime + str(gtNum) + '.npy'
    n_sc =  outFilePath+'isc' + atl03FilePath[11:15] + daytime + str(gtNum) + '.npy'
    n_lon = outFilePath+ 'ilon' + atl03FilePath[11:15] + daytime + str(gtNum) + '.npy'
    n_lat =  outFilePath+'ilat' + atl03FilePath[11:15] + daytime + str(gtNum) + '.npy'
    # Call getAtlMeasuredSwath
    print('RUNNING getAtlMeasuredSwath...\n')
    atl03Data, atl08Data, rotationData = getAtlMeasuredSwath(atl03FilePath, atl08FilePath, outFilePath, gtNum, trimInfo,
                                                             createAtl03LasFile, createAtl03KmlFile, createAtl08KmlFile,
                                                             createAtl03CsvFile, createAtl08CsvFile)
    np.save(n_time,atl03Data.time)
    np.save(n_at,atl03Data.alongTrack)
    np.save(n_height,atl03Data.z)
    np.save(n_sc,atl03Data.signalConf)
    np.save(n_lon,atl03Data.lon)
    np.save(n_lat,atl03Data.lat)