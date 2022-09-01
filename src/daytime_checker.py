import geopandas as gpd
import ephem
from datetime import datetime

import config


class DaytimeChecker:
    """
    Checker if it is daytime for a given time and region
    """

    def __init__(self):
        """
        Loads onshore and offshore shapefiles. Calculates the centroid of the region to determine the daytime at this
        location
        """
        self.shape_onshore = gpd.read_file(config.paths["onshore_shape"])
        self.shape_onshore.index = self.shape_onshore["name"]
        self.shape_onshore.drop(columns=["name"], inplace=True)

        self.shape_offshore = gpd.read_file(config.paths["offshore_shape"])
        self.shape_offshore.index = self.shape_offshore["name"]
        self.shape_offshore.drop(columns=["name"], inplace=True)

        self._add_centroids()

    def _add_centroids(self):
        """
        Helper function that adds the centroid.
        """
        # self.shape_onshore["centroid"] = self.shape_onshore.centroid
        self.shape_onshore["centroid_cea"] = self.shape_onshore.to_crs('+proj=cea').centroid.to_crs(
            self.shape_onshore.crs)
        # self.shape_offshore["centroid"] = self.shape_offshore.centroid
        self.shape_offshore["centroid_cea"] = self.shape_offshore.to_crs('+proj=cea').centroid.to_crs(
            self.shape_offshore.crs)

    def get_centroid_cea(self, is_onshore: bool, region: str) -> (str, str):
        """
        Gets the longitude and latitude coordinates of the centroid of the given region

        :param is_onshore: True if onshore region, False if offshore region
        :param region: Name of the region
        :return: String Tupel of longitude and latitude coordinates.
        """
        if is_onshore:
            lon = self.shape_onshore.loc[[region]]["centroid_cea"].x.values[0]
            lat = self.shape_onshore.loc[[region]]["centroid_cea"].y.values[0]
        else:
            lon = self.shape_offshore.loc[[region]]["centroid_cea"].x.values[0]
            lat = self.shape_offshore.loc[[region]]["centroid_cea"].y.values[0]
        return lon, lat

    def is_daytime(self, lon: str, lat: str, time=None) -> bool:
        """
        Checks if it is daytime at a given location and the given time.
        :param lon: longitude coordinate of the location
        :param lat: latitude coordinate of the location
        :param time: time. If no time is given, then the current utc time is used
        :return: True if it is day, False otherwise
        """
        obs = ephem.Observer()

        if time is None:
            obs.date = datetime.utcnow()
        else:
            obs.date = time
        obs.lat = str(lat)
        obs.lon = str(lon)

        print("UTC date: ", obs.date)

        next_sunrise = obs.next_rising(ephem.Sun())
        print("Next sunrise:", next_sunrise)

        next_sunset = obs.next_setting(ephem.Sun())
        print("Next sunset:", next_sunset)

        if next_sunset < next_sunrise:
            print("It is daytime: ", time)
            return True
        else:
            print("It is nighttime: ", time)
            return False
