"""
Author: Clement <<EMAIL>>
License: MIT
"""


import os
from dataclasses import  dataclass
from  loguru import logger
from  dotenv import load_dotenv
from typing import List
import ee
from pathlib import Path
import geopandas as gpd
import urllib.request
from tqdm import tqdm
import rioxarray as rio


@dataclass
class DataProcessor:
    start_date: str
    end_date: str
    bands_S2 : List[str]
    cloud_percentage : int =30
    resolution: int = 10

    @classmethod
    def read_file(cls, file_path: Path) -> gpd.GeoDataFrame:
        """
        :param file_path:
        :return: file reading
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist")
        dataset = gpd.read_file(file_path)

        return dataset


    @staticmethod
    def mask_s2_clouds(image: ee.Image) -> ee.Image:
        """
        :param image:
        :return:
        """
        qa = image.select('QA60')
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11

        mask = (
            qa.bitwiseAnd(cloud_bit_mask).eq(0)
            .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        )
        return image.updateMask(mask).divide(10000)


    def get_masked_sentinel2_image(self, file_path) -> ee.Image:
        """
        Returns a Sentinel-2 image collection with the required bands,
        cloud mask applied, and selected bands.

        :return: ee.ImageCollection
        """
        dataset = self.read_file(file_path)
        aoi = ee.Geometry.Polygon(dataset.geometry.unary_union.__geo_interface__["coordinates"]) #

        image_collection = (
            ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
            .filterBounds(aoi)  # Filtering by geometry
            .filterDate(self.start_date, self.end_date)  # Date range filter
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', self.cloud_percentage))  # Cloud filter
            .map(self.mask_s2_clouds)  # Apply cloud mask
            .select(self.bands_S2)  # Select desired bands
        )
        image = image_collection.median().clip(aoi)
        return image

    def export_image_to_drive(self, image: ee.Image, raster_name: str, output_dir: str, scale: int =10):
        """
        Export image to Google Drive
        :param image: The image to export
        :param raster_name: Name for the exported file
        :param output_dir: The folder in Google Drive where the file will be saved
        :param scale: Resolution (in meters)
        :return: None
        """
        # Define the export task
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=raster_name,
            folder=output_dir,
            fileNamePrefix=raster_name,
            scale=scale,
            region=image.geometry(),  # Specify the geometry (use the AOI)
            fileFormat='GeoTIFF',
            formatOptions={'cloudOptimized': True}  # Cloud optimized format
        )
        logger.info(f"Starting export task for {raster_name}")
        task.start()

        # Wait for the task to complete (you can also implement better task monitoring here)
        task_status = task.status()
        logger.info(f"Export status for {raster_name}: {task_status['state']}")

        return task

    def download_to_local_xarray(self, image: ee.Image, file_name: str, output_dir: str):
        """
        Download a Sentinel-2 image to a local directory.
        :param image:
        :param file_name:
        :param output_dir:
        :return:
        """

        try:
            # Generate download URL
            logger.info(f"Generate download link for {file_name}...")
            url = image.getDownloadURL({
                'scale': int(self.resolution),
                'region': image.geometry().bounds().getInfo(),
                'format': 'GeoTIFF',
                'maxPixels': 1e13
            })

            output_path = Path(output_dir) / f"{file_name}.tif"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            response = urllib.request.urlopen(url)
            total_size = int(response.getheader('Content-Length'))
            # Download image
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {file_name}") as t:
                # Utilisation de urllib.request.urlretrieve avec une mise à jour de tqdm manuelle
                urllib.request.urlretrieve(url, output_path,
                                           reporthook=lambda count, block_size, total_size: t.update(block_size))

            #urllib.request.urlretrieve(url, output_path)
            logger.info(f"Image successfully downloaded : {output_path}")

            # loading with xarray
            dataset = rio.open_rasterio(output_path)
            dataset = dataset.assign_coords({"band": self.bands_S2})  # Combine  bands

            # NetCDF backup
            #netcdf_path = Path(output_dir) / f"{file_name}.nc"
            #dataset.to_netcdf(netcdf_path)
            #logger.info(f"Image converted and saved in NetCDF : {netcdf_path}")
            # Close datasets cleanly to avoid locked files
            dataset.close()
            return dataset
        except Exception as e:
            logger.error(f"Error downloading or converting {file_name}: {e}")
            return None


if __name__ == "__main__":
    load_dotenv(override=True)
    ee.Initialize(project=os.getenv("EE_PROJECT"))

    data_ops = DataProcessor(
        start_date="2023-01-01",
        end_date="2023-12-31",
        bands_S2=['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],

    )

    shapefile_path = Path(str(os.getenv("FILE_DATA")))
    sentinel2_image = data_ops.get_masked_sentinel2_image(shapefile_path)

    dataset_xr = data_ops.export_image_to_drive(sentinel2_image, "bangalore_raster", output_dir=os.getenv("DATA_DIR"))



