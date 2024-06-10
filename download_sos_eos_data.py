#%%
'''
# Author: Nirajan Luintel
# Date: 2024-06-07
# Created with: Visual Studio Code
# Purpose: To download sos and eos data from wekeo platform
Mamba Environment: wekeo
'''
#%%
"""
I tried data download using the web portal and finally landed upon this sample that worked
https://github.com/ecmwf/hda/blob/master/demos/demo.py
But I edited a bit to make sure it suits me better
"""
import json
import os
import glob
from hda import Client
#%%
#set working directory so that data is downloaded where it should be
wd = '/data/yipeeo_wd/Data/VPP_Wekeo'
os.chdir(wd)

#get the client with configuration which reads the user details automatically
c = Client()
print (c.__dict__)
#%%
#Write the query which you can get from the web interface as well
query = {
    "dataset_id": "EO:EEA:DAT:CLMS_HRVPP_VPP",
    "productType": "QFLAG", #["QFLAG", 'SOSD', 'QFLAG'],
    "start": "2016-10-01T00:00:00.000Z",
    "end": "2024-05-30T00:00:00.000Z",
    "bbox": [
      -4.3,
      39.85,
      -3.0599999999999996,
      40.9
    ],
    "itemsPerPage": 200,
    "startIndex": 0
  }

matches = c.search(query)
print(matches)
# matches.download()
# %%
#process stopped after some operation and 
# I suspect that it is because it exceeded 
# limit of 100 files in one hour
filelist_downloaded = glob.glob('*.tif') #in wd only
filelist_id = [i[:-4] for i in filelist_downloaded]

# filelist_search = [matches.results[i]['id'] for i in range(len(matches))]
# search_filter = [matches[i] for i in range(len(matches)) if matches.results[i]['id'] not in filelist_id]
# %%
for i in range(len(matches)):
    if matches.results[i]['id'] not in filelist_id:
        print(matches.results[i]['id'])
        matches[i].download()
# %%
