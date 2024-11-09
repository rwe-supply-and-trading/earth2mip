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
import os
import numpy as np
import xarray
from modulus.utils import filesystem

import earth2mip.grid
from earth2mip.datasets.era5 import METADATA
from earth2mip.initial_conditions import base


def _get_filename(time: datetime.datetime, lead_time: str) -> str:
    """
    Returns the IFS forecast grib file containing control and perturbed forecasts
    given the specified datetime and lead time.
    Note that the filename format is specific to the ECMWF IFS forecasts,
    and may need to be adjusted for future versions.
    """
    file_format = f"%Y%m%d%H0000-{lead_time}-enfo-ef.grib2"
    return time.strftime(file_format)


def _get_channel(c: str, **kwargs) -> xarray.DataArray:
    """

    Parameters:
    -----------
    c: channel id
    **kwargs: variables in ecmwf data
    """
    # handle 2d inputs
    if c in kwargs:
        return kwargs[c]
    else:
        varcode, pressure_level = c[0], int(c[1:])
        return kwargs[varcode].interp(isobaricInhPa=pressure_level)


def get(time: datetime.datetime, channels: List[str], ensemble_member: int, 
        root_path: str, hens: bool) -> xarray.DataArray:
    path = os.path.join(root_path, _get_filename(time, "0h"))
    # open as list of Datasets given structure of grib 
    dataset_0h = cfgrib.open_datasets(path)
    
    if ensemble_member == 0:
        # control forecast
        dset = [ds for ds in dataset_0h if 0 in ds['number']]
        def _subset(varname):
            return [ds[varname] for ds in dset if varname in ds.data_vars][0]
    else:
        # perturbed forecasts
        dset = [ds for ds in dataset_0h if 0 not in ds['number']]
        def _subset(varname):
            return [ds[varname] for ds in dset if varname in ds.data_vars][0].sel(number=ensemble_member)

    # channel variables that do not require renaming
    channel_vars = ['sp', 't2m', 'msl', 'tcwv', 't', 'u', 'v', 'r']

    # add 2d just to test if it works
    channels.append('d2m')

    if hens:
        # add dewpoint temperature 
        channel_vars.append('d2m')
    
    channel_data = [
        _get_channel(
            c, 
            **{var: _subset(var) for var in channel_vars},
            u10m=_subset('u10'),
            v10m=_subset('v10'),
            u100m=_subset('u100'),
            v100m=_subset('v100'),
            z=_subset('gh') * 9.81,
        )   
        for c in channels
    ] 

    # dataset_0h is list of Datasets, grab first one 
    # for creating new array, variable doesn't matter 
    ds_ex = dataset_0h[0]

    array = np.stack([d for d in channel_data], axis=0)
    
    darray = xarray.DataArray(
        array,
        dims=["channel", "lat", "lon"],
        coords={
            "channel": channels,
            "lon": ds_ex.longitude.values,
            "lat": ds_ex.latitude.values,
            "time": time,
        },
    )
    
    return darray


@dataclasses.dataclass
class DataSource(base.DataSource):
    def __init__(self, channel_names: List[str], from_path: str, ensemble_member: int = 0, hens: bool = False):
        self._channel_names = channel_names
        self._ensemble_member = ensemble_member
        self._root_path = from_path
        self._hens = hens

    @property
    def channel_names(self) -> List[str]:
        return self._channel_names

    @property
    def hens(self) -> bool:
        # boolean flag that includes dewpoint temperature (2d) if set to True
        # note that SFNO version that has dewpoint temperature as input channel must be used
        return self._hens

    @property
    def ensemble_member(self) -> int:
        return self._ensemble_member

    @property
    def root_path(self) -> str:
        return self._root_path

    @property
    def grid(self) -> earth2mip.grid.LatLonGrid:
        return earth2mip.grid.equiangular_lat_lon_grid(721, 1440)

    def __getitem__(self, time: datetime.datetime) -> np.ndarray:
        ds = get(time, self.channel_names, self.ensemble_member, 
                self.root_path, self.hens)

        # move to earth2mip.channels
        metadata = json.loads(METADATA.read_text())
        lat = np.array(metadata["coords"]["lat"])
        lon = np.array(metadata["coords"]["lon"])
        ds = ds.roll(lon=len(ds.lon) // 2, roll_coords=True)
        ds["lon"] = ds.lon.where(ds.lon >= 0, ds.lon + 360)
        assert min(ds.lon) >= 0, min(ds.lon)  # noqa
        return ds
