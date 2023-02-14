import pandas
import glob
import numpy as np
from loguru import logger
from typing import List


def read_all_recordings_from_csv(base_path: str = "../data/") -> List[dict]:
    """
    Read tracks and meta information for all recordings in a directory
    Warning: This might need a lot of memory!
    :param base_path: Directory containing all csv files of the dataset
    :return: Tuple of tracks, tracks meta and recording meta
    """
    tracks_files = sorted(glob.glob(base_path + "*_tracks.csv"))
    tracks_meta_files = sorted(glob.glob(base_path + "*_tracksMeta.csv"))
    recording_meta_files = sorted(glob.glob(base_path + "*_recordingMeta.csv"))

    recordings = []
    for track_file, tracks_meta_file, recording_meta_file in zip(tracks_files,
                                                                 tracks_meta_files,
                                                                 recording_meta_files):
        logger.info("Loading csv files {}, {} and {}", track_file, tracks_meta_file, recording_meta_file)
        tracks, tracks_meta, recording_meta = read_from_csv(track_file, tracks_meta_file, recording_meta_file)
        recordings.append({"tracks": tracks, "tracks_meta": tracks_meta, "recording_meta": recording_meta})

    return recordings


def read_from_csv(tracks_file: str, tracks_meta_file: str,
                  recording_meta_file: str, include_px_coordinates: bool=False, scale: float=1) -> (list, list, dict):
    """
    This method reads tracks and meta data for a single recording from csv files
    :param tracks_file: Path of a tracks csv file
    :param tracks_meta_file: Path of a tracks meta csv file
    :param recording_meta_file: Path of a recording meta csv file
    :param include_px_coordinates: Set to true, if the tracks are used for the visualizer
    :param scale: Scale factor to scale the px_coordinates
    :return: Tuple of (tracks, tracks meta, recording meta)
    """
    recording_meta = read_recording_meta(recording_meta_file)
    tracks_meta = read_tracks_meta(tracks_meta_file)
    tracks = read_tracks(tracks_file, recording_meta, include_px_coordinates, scale)
    return tracks, tracks_meta, recording_meta


def read_tracks(tracks_file: str, recording_meta: dict, include_px_coordinates: bool=False, scale: float=1) -> List[dict]:
    """
    Read tracks from a csv file
    :param tracks_file: Path of a tracks csv file
    :param recording_meta: Loaded meta of the corresponding recording
    :param include_px_coordinates: Set to true, if the tracks are used for the visualizer
    :param scale: Scale factor to scale the px_coordinates
    :return: A list of tracks represented as dictionary each
    """
    # To extract every track, group the rows by the track id
    raw_tracks = pandas.read_csv(tracks_file).groupby(["trackId"], sort=True)
    ortho_px_to_meter = recording_meta["orthoPxToMeter"]*scale

    # Convert groups of rows to tracks
    tracks = []
    for track_id, track_rows in raw_tracks:
        track = track_rows.to_dict(orient="list")

        # Convert lists to numpy arrays
        for key, value in track.items():
            if key in ["trackId", "recordingId"]:
                track[key] = value[0]
            else:
                track[key] = np.array(value)

        track["center"] = np.stack([track["xCenter"], track["yCenter"]], axis=-1)
        if np.count_nonzero(track["length"]) and np.count_nonzero(track["width"]):
            # Only calculate bounding box of objects with a width and length (e.g. cars)
            track["bbox"] = get_rotated_bbox(track["xCenter"], track["yCenter"],
                                             track["length"], track["width"],
                                             np.deg2rad(track["heading"]))
        else:
            track["bbox"] = None

        if include_px_coordinates:
            # As the tracks are given in utm coordinates, transform these to pixel coordinates for visualization
            track["xCenterVis"] = track["xCenter"] / ortho_px_to_meter
            track["yCenterVis"] = -track["yCenter"] / ortho_px_to_meter
            track["centerVis"] = np.stack([track["xCenterVis"], track["yCenterVis"]], axis=-1)
            track["widthVis"] = track["width"] / ortho_px_to_meter
            track["lengthVis"] = track["length"] / ortho_px_to_meter
            track["headingVis"] = track["heading"] * -1
            track["headingVis"][track["headingVis"] < 0] += 360
            if np.count_nonzero(track["length"]) and np.count_nonzero(track["width"]):
                # Only calculate bounding box of objects with a width and length (e.g. cars)
                track["bboxVis"] = get_rotated_bbox(track["xCenterVis"], track["yCenterVis"],
                                                    track["lengthVis"], track["widthVis"],
                                                    np.deg2rad(track["headingVis"]))
            else:
                track["bboxVis"] = None

        tracks.append(track)
    return tracks


def read_tracks_meta(tracks_meta_file: str) -> List[dict]:
    """
    Read tracks meta from a csv file
    :param tracks_meta_file: Path of a tracks meta csv file
    :return: List of tracks meta represented as dictionary each
    """
    return sorted(pandas.read_csv(tracks_meta_file).to_dict(orient="records"), key=lambda entry: entry["trackId"])


def read_recording_meta(recording_meta_file: str) -> dict:
    """
    Read recording meta from a csv file
    :param recording_meta_file: Path of a recording meta csv file
    :return: Dictionary of the recording meta
    """
    return pandas.read_csv(recording_meta_file).to_dict(orient="records")[0]


def get_rotated_bbox(x_center: np.ndarray, y_center: np.ndarray,
                     length: np.ndarray, width: np.ndarray, heading: np.ndarray) -> np.ndarray:
    """
    Calculate the corners of a rotated bbox from the position, shape and heading for every timestamp.

    :param x_center: x coordinates of the object center positions [num_timesteps]
    :param y_center: y coordinates of the object center positions [num_timesteps]
    :param length: objects lengths [num_timesteps]
    :param width: object widths [num_timesteps]
    :param heading: object heading (rad) [num_timesteps]
    :return: Numpy array in the shape [num_timesteps, 4 (corners), 2 (dimensions)]
    """
    centroids = np.column_stack([x_center, y_center])

    # Precalculate all components needed for the corner calculation
    l = length / 2
    w = width / 2
    c = np.cos(heading)
    s = np.sin(heading)

    lc = l * c
    ls = l * s
    wc = w * c
    ws = w * s

    # Calculate all four rotated bbox corner positions assuming the object is located at the origin.
    # To do so, rotate the corners at [+/- length/2, +/- width/2] as given by the orientation.
    # Use a vectorized approach using precalculated components for maximum efficiency
    rotated_bbox_vertices = np.empty((centroids.shape[0], 4, 2))

    # Front-right corner
    rotated_bbox_vertices[:, 0, 0] = lc - ws
    rotated_bbox_vertices[:, 0, 1] = ls + wc

    # Rear-right corner
    rotated_bbox_vertices[:, 1, 0] = -lc - ws
    rotated_bbox_vertices[:, 1, 1] = -ls + wc

    # Rear-left corner
    rotated_bbox_vertices[:, 2, 0] = -lc + ws
    rotated_bbox_vertices[:, 2, 1] = -ls - wc

    # Front-left corner
    rotated_bbox_vertices[:, 3, 0] = lc + ws
    rotated_bbox_vertices[:, 3, 1] = ls - wc

    # Move corners of rotated bounding box from the origin to the object's location
    rotated_bbox_vertices = rotated_bbox_vertices + np.expand_dims(centroids, axis=1)
    return rotated_bbox_vertices
