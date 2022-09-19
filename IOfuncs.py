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
import datetime as dt

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

def download_hrrr(datetime, point_lat, point_lon, var_list=None):
    #set default var_list
    if var_list is None:
        var_list = [('TMP', 'surface'),
            ('TMP', '500mb'),
            ('TMP', '700mb'),
            ('TMP', '850mb'),
            ('TMP', '925mb'),
            ('TMP', '1000mb'),
            ('VGRD', '10m_above_ground'),
            ('UGRD', '10m_above_ground'),
            ('VGRD', '250mb'),
            ('UGRD', '250mb'),
            ('VGRD', '300mb'),
            ('UGRD', '300mb'),
            ('VGRD', '500mb'),
            ('UGRD', '500mb'),
            ('VGRD', '700mb'),
            ('UGRD', '700mb'),
            ('VGRD', '850mb'),
            ('UGRD', '850mb'),
            ('VGRD', '925mb'),
            ('UGRD', '925mb'),
            ('VGRD', '1000mb'),
            ('UGRD', '1000mb'),
            ('DPT', '2m_above_ground'),
            ('DPT', '500mb'),
            ('DPT', '700mb'),
            ('DPT', '850mb'),
            ('DPT', '925mb'),
            ('DPT', '1000mb'),
            ('HGT', 'cloud_base'),
            ('HGT', 'cloud_ceiling'),
            ('VIS', 'surface')]
        
    chunk_index = xr.open_zarr(s3fs.S3Map("s3://hrrrzarr/grid/HRRR_chunk_index.zarr", s3=fs))
    #this is just the projection that hrrr uses
    projection = ccrs.LambertConformal(central_longitude=262.5, 
                                       central_latitude=38.5, 
                                       standard_parallels=(38.5, 38.5),
                                        globe=ccrs.Globe(semimajor_axis=6371229,
                                                         semiminor_axis=6371229))

    x, y = projection.transform_point(point_lon, point_lat, ccrs.PlateCarree())

    nearest_point = chunk_index.sel(x=x, y=y, method="nearest")
    fcst_chunk_id = f"0.{nearest_point.chunk_id.values}"

    date = str(datetime)[:10].replace('-','')
    hr = str(datetime)[11:13]
    hrrr_df = pd.DataFrame()
    for (var, level) in var_list:
        data_url = f'hrrrzarr/sfc/{date}/{date}_{hr}z_fcst.zarr/{level}/{var}/{level}/{var}/'
        data = retrieve_data(data_url + fcst_chunk_id)
        gridpoint_forecast = data[:, nearest_point.in_chunk_x, nearest_point.in_chunk_y]
        hrrr_df[f'{var}_{level}'] = gridpoint_forecast
    return hrrr_df

def download_glamp(datetime, station):
    base_url = 'https://mesonet.agron.iastate.edu/api/1/mos.json'
    
    params = {'station': station,
              'model': 'LAV',
              'runtime': datetime}
    response = requests.get(base_url, params=params)
    if response.status_code==404:
        raise FileNotFoundError('File not found on database')
    data = response.json()['data']
    result = pd.DataFrame(data)
    return result

#helper func for get_metar_at_time
def get_seconds(delta):
    return delta.total_seconds()

def parse_hourly_precip(value):
    if value == 'T':
        return 0.05
    else:
        return value
    
def parse_metar_wxcode(code):
    WX_CAT = {'MI': 2, 'PR': 3, 'BC': 5, 'DR': 7, 'BL': 11, 'SH': 13, 'TS': 17, 'FZ': 19, 'DZ': 23, 'RA': 29, 'SN': 31, 'SG': 37, 'IC': 41, 'PL': 43,
              'GR': 47, 'GS': 53, 'UP': 59, 'BR': 61, 'FG': 67, 'FU': 71, 'VA': 73, 'DU': 79, 'SA': 83, 'HZ': 89, 'PY': 97, 'PO': 101, 'SQ': 103, 'FC': 107,
              'SS': 109}
    #These are all prime numbers, by multiplying out the codes that appear each unique wx code produces a unique number by
    #unique factorization
    result = 1
    if isinstance(code, float):
        return code
    
    for wxcode in WX_CAT:
        if wxcode in code:
            result *= WX_CAT[wxcode]
    if '+' in code:
        result += 0.5
    if '-' in code:
        result -= 0.5
    return result
    

def read_metar(metar_path):
    
    CLOUD_CATS = {'FEW': 0, 'SCT': 1, 'BKN': 2, 'OVC': 3}
    
    metar = pd.read_csv(metar_path)
    metar['p01i'] = metar['p01i'].map(parse_hourly_precip)
    metar['skyc1'] = metar['skyc1'].map(CLOUD_CATS)
    metar['skyc2'] = metar['skyc2'].map(CLOUD_CATS)
    metar['skyc3'] = metar['skyc3'].map(CLOUD_CATS)
    metar['skyc4'] = metar['skyc4'].map(CLOUD_CATS)
    
    metar['wxcodes'] = metar['wxcodes'].map(parse_metar_wxcode)
    
    return metar

def get_metar_at_time(datetime, full_metar_list):
    if isinstance(full_metar_list, str):
        full_metar_list = pd.read_csv(full_metar_list)
    date_list = np.abs(pd.to_datetime(full_metar_list['valid']) - datetime)
    metar_index = np.argmin(date_list)
    return full_metar_list.iloc[metar_index]

def get_glamp_at_time(datetime, path, station, download=False):
    timestamp = int(str(datetime)[11:13])
    if timestamp < 6:
        initalization_time = '00:00'
    elif timestamp < 12:
        initalization_time = '06:00'
    elif timestamp < 18:
        initalization_time = '12:00'
    else: 
        initalization_time = '18:00'
    glamp_run_time = f'{str(datetime)[:10]}T{initalization_time}Z'
    
    CC1_CATS = {'N': 0, 'L': 1, 'M': 2, 'H': 3}
    CLD_CATS = {'CL': 0, 'FW': 1, 'SC': 2, 'BK': 3, 'OV': 4}
    OBV_CATS = {'N': 0, 'HZ': 1, 'BR': 2, 'FG': 3, 'BL':4 }
    PC_CATS = {'N': 0, 'Y': 1}
    TYP_CATS = {'S': 0, 'Z': 1, 'R': 2, 'X': 3}
    
    fname = glamp_run_time[:13]+'Z.csv'
    if fname in listdir(path):
        result = pd.read_csv(path + fname)
    else:
        if download:
            result = download_glamp(glamp_run_time, station)
            result.to_csv(path + fname)
            
    result['cc1'] = result['cc1'].map(CC1_CATS)
    result['cld'] = result['cld'].map(CLD_CATS)
    result['lc1'] = result['lc1'].map(CC1_CATS)
    result['obv'] = result['obv'].map(OBV_CATS)
    result['pco'] = result['pco'].map(PC_CATS)
    result['pc1'] = result['pc1'].map(PC_CATS)
    result['typ'] = result['typ'].map(TYP_CATS)
    
    return result

def get_hrrr_at_time(datetime, path, lat, lon, download=False, var_list=None):
    date = str(datetime)[:10].replace('-','')
    hr = str(datetime)[11:13]
    fname = f'{date}{hr}_{lat}_{lon}.csv'
    
    if fname in listdir(path):
        data = pd.read_csv(path + fname) 
    else:
        if download:
            data = download_hrrr(datetime, lat, lon, var_list=var_list)
            data.to_csv(path + fname)
    data = data.replace([np.inf, -np.inf], -99999)
    return data