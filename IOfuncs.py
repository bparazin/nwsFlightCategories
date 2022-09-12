import pandas as pd
import numpy as np
import requests
from fractions import Fraction
from tqdm import tqdm
from os import listdir, mkdir
import xarray as xr
import cartopy.crs as ccrs
import s3fs
import numcodecs as ncd
import pickle

fs = s3fs.S3FileSystem(anon=True)

#this assumes that visibility is always the first value reported in standard miles (ending in SM) in the metar
def find_visibility(metar):
    metar_list = metar.split(' ')
    for i,datapoint in enumerate(metar_list):
        if datapoint[-2:] == 'SM':
            if metar_list[i][0] == 'M':
                #special case to cover >1/4SM visibility
                return 0.125
            else:
                return float(Fraction(metar_list[i][:-2]))
#these two rely on broken and overcast being the first two things with the leading characters BKN and OVC. Both return None 
#if they don't feature in the METAR
def find_broken_height(metar):
    metar_list = metar.split(' ')
    for i,datapoint in enumerate(metar_list):
        if datapoint[:3] == 'BKN':
            return 100 * int(metar_list[i][3:6])
    return None

def find_overcast_height(metar):
    metar_list = metar.split(' ')
    for i,datapoint in enumerate(metar_list):
        if datapoint[:3] == 'OVC':
            return 100 * int(metar_list[i][3:6])
    return None

#Just combines the two above, handling all the NONE cases
def find_ceiling_height(metar):
    if find_overcast_height(metar) is None and find_broken_height(metar) is None:
        return None
    if find_overcast_height(metar) is None:
        return find_broken_height(metar)
    if find_broken_height(metar) is None:
        return find_overcast_height(metar)
    return min(find_overcast_height(metar), find_broken_height(metar))

#This uses the fact that the timestamp of the METAR is always ddttttZ, allowing for easy conversion to 24-hour Zulu time by trimming the ends
#Then this just looks at the last 6-hour mark preceding that timestamp
def GLAMPstamp(metar):
    metar_list = metar.split(' ')
    datapoint = metar_list[1]
    initalization_time = None
    timestamp = int(datapoint[2:-1])
    if timestamp < 600:
        initalization_time = '00:00'
    elif timestamp < 1200:
        initalization_time = '06:00'
    elif timestamp < 1800:
        initalization_time = '12:00'
    else: 
        initalization_time = '18:00'
    return initalization_time

#Gets the 2 hour for the HRRR before the metar
def HRRRstamp(metar):
    metar_list = metar.split(' ')
    datapoint = metar_list[1]
    hour = datapoint.split(':')[0]
    hour = str((int(hour) - 1) % 24)
    if len(hour) == 1:
        hour = '0' + hour
    return hour

#This takes a date in mm/dd/yyyy and converts it to yyyy-mm-dd
def format_date(validString, hrrrDate=False):
    dmy = validString.split(' ')[0]
    month, day, year = dmy.split('/')
    if len(day) == 1:
        day = '0' + day
    if len(month) == 1:
        month = '0' + month
    if hrrrDate:
        return f'{year}{month}{day}'
    else:
        return f'{year}-{month}-{day}'
    
    
def retrieve_data(s3_url):
    with fs.open(s3_url, 'rb') as compressed_data: # using s3fs
        buffer = ncd.blosc.decompress(compressed_data.read())

        dtype = "<f2"
        if "surface/PRES" in s3_url: # surface/PRES is the only variable with a larger data type
            dtype = "<f4"

        chunk = np.frombuffer(buffer, dtype=dtype)
        
        entry_size = 150*150
        num_entries = len(chunk)//entry_size

        if num_entries == 1: # analysis file is 2d
            data_array = np.reshape(chunk, (150, 150))
        else:
            data_array = np.reshape(chunk, (num_entries, 150, 150))

    return data_array
