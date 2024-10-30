# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import datetime
import json
import cfgrib
from typing import List

import numpy as np
import xarray
from modulus.utils import filesystem

import earth2mip.grid
from earth2mip.datasets.era5 import METADATA
from earth2mip.initial_conditions import base


def _get_filename(time: datetime.datetime, lead_time: str):
    # date_format = f"%Y%m%d/%Hz/0p4-beta/oper/%Y%m%d%H%M%S-{lead_time}-oper-fc.grib2"
    date_format = f"%Y%m%d/%Hz/ifs/0p4-beta/oper/%Y%m%d%H%M%S-{lead_time}-oper-fc.grib2"
    return time.strftime(date_format)


def _get_channel(c: str, **kwargs) -> xarray.DataArray:
    """

    Parameters:
    -----------
    c: channel id
    **kwargs: variables in ecmwf data
    """
    # handle 2d inputs
    if c in kwargs:
        print("returning {} in kwargs".format(c))
        return kwargs[c]
    else:
        varcode, pressure_level = c[0], int(c[1:])
        print("providing {} andd {}".format(varcode, pressure_level))
        return kwargs[varcode].interp(isobaricInhPa=pressure_level)


def get(time: datetime.datetime, channels: List[str]):
    # this data product is outdated (as of 10/2024) and no longer exists
    # also it is no longer necessary to get these data files separately
    # either update to GCS or completely rely on local files
    # root = "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com/"
    root = "/dev/shm/"
    # oper for testing
    # filename = "20241020120000-0h-oper-fc.grib2"
    # ifs ensemble 
    filename = "20241020120000-0h-enfo-ef.grib2"
    #path = root + _get_filename(time, "0h")
    path = root + filename
    print("path is {}".format(path))
    # local_path = filesystem._download_cached(path)
    # dataset_0h = xarray.open_dataset(local_path, engine="cfgrib")
    #dataset_0h = xarray.open_dataset(path, filter_by_keys={'dataType': 'pf', 'typeOfLevel': 'heightAboveGround', 'level': 2}, engine="cfgrib")
    dataset_0h = cfgrib.open_datasets(path)

    ## TO-DO add as argument
    ensemble_member = 2

    # print(dataset_0h)
    print([ds['u10'] for ds in dataset_0h if 'u10' in ds.data_vars][0])
    print([ds['u10'] for ds in dataset_0h if 'u10' in ds.data_vars][0].sel(number=ensemble_member))
    # print(dataset_0h.data_vars)
    # get t2m and other things from 12 hour forecast initialized 12 hours before
    # The HRES is only initialized every 12 hours
    #path = root + _get_filename(time - datetime.timedelta(hours=12), "12h")
    #print("second filepath is {}".format(path))
    #local_path = filesystem._download_cached(path)
    #forecast_12h = xarray.open_dataset(local_path, engine="cfgrib")

    channel_data = [
        _get_channel(
            c,
            #u10m=dataset_0h.u10,
            u10m=[ds['u10'] for ds in dataset_0h if 'u10' in ds.data_vars][0].sel(number=ensemble_member),
            #v10m=dataset_0h.v10,
            v10m=[ds['v10'] for ds in dataset_0h if 'v10' in ds.data_vars][0].sel(number=ensemble_member),
            # u100m=dataset_0h.u10,
            u100m=[ds['u100'] for ds in dataset_0h if 'u100' in ds.data_vars][0].sel(number=ensemble_member),
            #v100m=dataset_0h.v10,
            v100m=[ds['v100'] for ds in dataset_0h if 'v100' in ds.data_vars][0].sel(number=ensemble_member),
            #sp=dataset_0h.sp,
            sp=[ds['sp']  for ds in dataset_0h if 'sp' in ds.data_vars][0].sel(number=ensemble_member),
            # t2m=forecast_12h.t2m,
            #t2m=dataset_0h.t2m,
            t2m=[ds['t2m'] for ds in dataset_0h if 't2m' in ds.data_vars][0].sel(number=ensemble_member),
            #msl=forecast_12h.msl,
            #msl=dataset_0h.msl,
            msl=[ds['msl'] for ds in dataset_0h if 'msl' in ds.data_vars][0].sel(number=ensemble_member),
            # tcwv=forecast_12h.tciwv,
            #tcwv=dataset_0h.tciwv,
            tcwv=[ds['tcwv'] for ds in dataset_0h if 'tcwv' in ds.data_vars][0].sel(number=ensemble_member),
            #t=dataset_0h.t,
            t=[ds['t'] for ds in dataset_0h if 't' in ds.data_vars][0].sel(number=ensemble_member),
            #u=dataset_0h.u,
            u=[ds['u'] for ds in dataset_0h if 'u' in ds.data_vars][0].sel(number=ensemble_member),
            #v=dataset_0h.v,
            v=[ds['v'] for ds in dataset_0h if 'v' in ds.data_vars][0].sel(number=ensemble_member),
            #r=dataset_0h.r,
            r=[ds['r'] for ds in dataset_0h if 'r' in ds.data_vars][0].sel(number=ensemble_member),
            #z=dataset_0h.gh * 9.81,
            z=[ds['gh'] for ds in dataset_0h if 'gh' in ds.data_vars][0].sel(number=ensemble_member) * 9.81,
        )
        for c in channels
    ]

    # dataset_0h is list of Datasets, grab first one 
    # for creating new array
    ds_ex = dataset_0h[0]

    array = np.stack([d for d in channel_data], axis=0)
    print("shape of array is {}".format(array.shape))
    darray = xarray.DataArray(
        array,
        dims=["channel", "lat", "lon"],
        # dims=["channel", "ensemble_member", "lat", "lon"],
        coords={
            "channel": channels,
            # "ensemble_member": ds_ex.number.values,
            # "lon": dataset_0h.longitude.values,
            "lon": ds_ex.longitude.values,
            #"lat": dataset_0h.latitude.values,
            "lat": ds_ex.latitude.values,
            "time": time,
        },
    )
    print("created array")
    return darray


@dataclasses.dataclass
class DataSource(base.DataSource):
    def __init__(self, channel_names: List[str]):
        self._channel_names = channel_names

    @property
    def channel_names(self) -> List[str]:
        return self._channel_names

    @property
    def grid(self) -> earth2mip.grid.LatLonGrid:
        return earth2mip.grid.equiangular_lat_lon_grid(721, 1440)

    def __getitem__(self, time: datetime.datetime) -> np.ndarray:
        ds = get(time, self.channel_names)
        # ds = ds.expand_dims("time", axis=0)
        # move to earth2mip.channels

        # TODO refactor interpolation to another place
        metadata = json.loads(METADATA.read_text())
        lat = np.array(metadata["coords"]["lat"])
        lon = np.array(metadata["coords"]["lon"])
        ds = ds.roll(lon=len(ds.lon) // 2, roll_coords=True)
        print("rolled longitudes") 
        ds["lon"] = ds.lon.where(ds.lon >= 0, ds.lon + 360)
        print("reassigned rolled longitudes")
        assert min(ds.lon) >= 0, min(ds.lon)  # noqa
        # return ds.interp(lat=lat, lon=lon, kwargs={"fill_value": "extrapolate"})
        return ds
