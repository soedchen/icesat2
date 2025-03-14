import os
import pandas as pd
import numpy as np
df = np.array([])
path = 'E:\\xcsdata\\xisha\\controlpoints\\origindata\\去噪后点\\'
fileList = os.listdir(path)
for eachFile in fileList:
    FilePath = path + '/' + eachFile
    tem = pd.read_csv(FilePath)
    df = tem if eachFile == fileList[0] else np.concatenate((df, tem),axis=0)

save_csv = pd.DataFrame(df)
f_path = path + '/' +'all_med_xisha.csv'
save_csv.to_csv(f_path,index=False,header=0)