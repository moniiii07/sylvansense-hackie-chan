import ee
import numpy as np
import requests
from dataclasses import dataclass

ee.Initialize(project='composed-arch-476417-e5')

S2_BAND_MAP = {"B2": "B02", "B3": "B03", "B4": "B04", "B8": "B08", "B11": "B11", "B12": "B12"}


@dataclass
class SceneStack:
    bands: dict
    transform: object
    crs: str
    shape: tuple


def _ee_image_to_array(image, region, scale, bands):
    url = image.select(bands).getDownloadURL({
        "region": region, "scale": scale, "format": "GEO_TIFF",
    })
    resp = requests.get(url)
    resp.raise_for_status()

    import rasterio
    with rasterio.MemoryFile(resp.content) as memfile:
        with memfile.open() as src:
            arr = src.read()
            transform, crs, shape = src.transform, src.crs.to_string(), (src.height, src.width)
    return arr, transform, crs, shape


def fetch_sentinel_scene(bbox: dict, start: str = "2024-01-01", end: str = "2024-06-30",
                          scale: int = 10) -> SceneStack:
    aoi = ee.Geometry.Rectangle([bbox["min_lon"], bbox["min_lat"], bbox["max_lon"], bbox["max_lat"]])

    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 15))
            .median())
    s2_bands_real = list(S2_BAND_MAP.keys())
    s2_arr, transform, crs, shape = _ee_image_to_array(s2, aoi, scale, s2_bands_real)
    s2_arr = s2_arr.astype(np.float32) / 10000.0

    bands = {}
    for i, real_name in enumerate(s2_bands_real):
        bands[S2_BAND_MAP[real_name]] = s2_arr[i]

    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(aoi)
            .filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .select(["VV", "VH"])
            .median())
    s1_arr, _, _, _ = _ee_image_to_array(s1, aoi, scale, ["VV", "VH"])
    bands["VV"] = s1_arr[0].astype(np.float32)
    bands["VH"] = s1_arr[1].astype(np.float32)

    chm = ee.Image("users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1")
    chm_arr, _, _, _ = _ee_image_to_array(chm, aoi, scale, [chm.bandNames().get(0).getInfo()])
    bands["CHM"] = chm_arr[0].astype(np.float32)

    return SceneStack(bands=bands, transform=transform, crs=crs, shape=shape)
