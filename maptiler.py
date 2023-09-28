import requests
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

from utils import MAPS_API_KEY
import os

def latLngToPoint(mapWidth, mapHeight, lat, lng):

    x = (lng + 180) * (mapWidth/360)
    y = ((1 - math.log(math.tan(lat * math.pi / 180) + 1 / math.cos(lat * math.pi / 180)) / math.pi) / 2) * mapHeight

    return(x, y)

def pointToLatLng(mapWidth, mapHeight, x, y):

    lng = x / mapWidth * 360 - 180

    n = math.pi - 2 * math.pi * y / mapHeight
    lat = (180 / math.pi * math. atan(0.5 * (math.exp(n) - math.exp(-n))))

    return(lat, lng)

def getImageBounds(mapWidth, mapHeight, xScale, yScale, lat, lng):

    centreX, centreY = latLngToPoint(mapWidth, mapHeight, lat, lng)

    southWestX = centreX - (mapWidth/2)/ xScale
    southWestY = centreY + (mapHeight/2)/ yScale
    SWlat, SWlng = pointToLatLng(mapWidth, mapHeight, southWestX, southWestY)

    northEastX = centreX + (mapWidth/2)/ xScale
    northEastY = centreY - (mapHeight/2)/ yScale
    NElat, NElng = pointToLatLng(mapWidth, mapHeight, northEastX, northEastY)

    return[SWlat, SWlng, NElat, NElng]

def getLatStep(mapWidth, mapHeight, yScale, lat, lng):

    pointX, pointY = latLngToPoint(mapWidth, mapHeight, lat, lng)

    steppedPointY = pointY - ((mapHeight)/ yScale)
    newLat, originalLng = pointToLatLng(mapWidth, mapHeight, pointX, steppedPointY)

    latStep = lat - newLat

    return (latStep)

def requestImage(AreaID, picHeight, picWidth, zoom, scale, maptype, lat, lng, row, col):

    center = str(lat) + "," + str(lng)
    url = "https://maps.googleapis.com/maps/api/staticmap?center=" + center + "&zoom=" + str(zoom) + "&size=" + str(picWidth) + "x" + str(picHeight) + "&key=" + MAPS_API_KEY + "&maptype=" + maptype + "&scale=" + str(scale)
    print(url)
    filename = AreaID + "-" + str(abs(col)) + "," + str(row) + ".png"

    r = requests.get(url)
    f = open(filename, 'wb')
    f.write(r.content)
    f.close()

    print("writtern to file: " + filename)

def requestImageWithCrop(AreaID, picHeight, picWidth, zoom, scale, maptype, lat, lng, row, col, additional_parameters):
    center = str(lat) + "," + str(lng)
    if additional_parameters is None:
        url = "https://maps.googleapis.com/maps/api/staticmap?center=" + center + "&zoom=" + str(zoom) + "&size=" + str(picWidth) + "x" + str(picHeight) + "&key=" + MAPS_API_KEY + "&maptype=" + maptype + "&scale=" + str(scale)
    else:
        url = "https://maps.googleapis.com/maps/api/staticmap?center=" + center + "&zoom=" + str(zoom) + "&size=" + str(picWidth) + "x" + str(picHeight) + "&key=" + MAPS_API_KEY + "&scale=" + str(scale) + additional_parameters

    filename = AreaID + "-" + str(abs(col)) + "," + str(row) + ".png"

    r = requests.get(url)

    if r.status_code == 200:
        with open(filename, 'wb') as f:
            image = Image.open(io.BytesIO(r.content))
            # Crop 30 pixels from the bottom
            cropped_image = image.crop((0, 0, picWidth*scale, picHeight*scale - 45))
            cropped_image.save(f, 'PNG')
        print("Written to file: " + filename)
    else:
        print("Failed to retrieve image for", lat, lng, "Status Code:", r.status_code)

def getLatStepWithCrop(mapWidth, mapHeight, yScale, lat, lng, crop_pixels):
    pointX, pointY = latLngToPoint(mapWidth, mapHeight, lat, lng)

    # Calculate the stepped point by subtracting the cropped pixels
    steppedPointY = pointY - ((mapHeight - crop_pixels) / yScale)
    newLat, originalLng = pointToLatLng(mapWidth, mapHeight, pointX, steppedPointY)

    latStep = lat - newLat

    return latStep

def get_tiling_images(area_id, center_Lat, center_Lng, padding=0.012, ground_truth=False):
    # Bounding box for area to be scanned. AreaID is added to file name.
    center_Lat = center_Lat #52.315375
    center_Lng = center_Lng #9.756498

    padding = 0.012

    northWestLat = np.round(center_Lat+padding/2,7)
    northWestLng = np.round(center_Lng-padding,7)
    southEastLat = np.round(center_Lat-padding/2,7)
    southEastLng = np.round(center_Lng+padding,7)

    # Variables for API request
    api_key = "AIzaSyA9D01ZYb1tYq44l6-S_3BUYtgY8Pzjizk"
    zoom = 17
    picHeight = 640
    picWidth = 640
    scale = 2
    maptype = "satellite"
    additional_parameters = None if not ground_truth else "&map_id=c2e4254a97b86e42&style=feature:all|element:labels|visibility:off"
    # --- do not zchange variables below this point ---

    mapHeight = 256
    mapWidth = 256
    xScale = math.pow(2, zoom) / (picWidth/mapWidth)
    yScale = math.pow(2, zoom) / (picHeight/mapWidth)

    startLat = northWestLat
    startLng = northWestLng

    startCorners = getImageBounds(mapWidth, mapHeight, xScale, yScale, startLat, startLng)
    lngStep = startCorners[3] - startCorners[1]

    col = 0
    lat = startLat

    while (lat >= southEastLat):
        lng = startLng
        row = 0

        while lng <= southEastLng:
            requestImageWithCrop(area_id, picHeight, picWidth, zoom, scale, maptype, lat, lng, row, col, additional_parameters)
            row = row + 1
            lng = lng + lngStep

        col = col - 1
        lat = lat + getLatStepWithCrop(mapWidth, mapHeight, yScale, lat, lng, 45)

    n_rows = row
    n_cols = abs(col)
    return n_rows, n_cols

def combine_tiling_images(area_id, n_rows, n_cols, ground_truth=False):
    path = ""
    extension = "png"

    min_value = min(n_rows, n_cols)

    rows = []

    for r in range(min_value):
        col = []
        for c in range(min_value):
            img = Image.open(f"{path}/{area_id}-{r},{c}.{extension}").convert("RGB")
            image = np.asarray(img, dtype="int32")
            if ground_truth:
                image = np.array(img, dtype="int32")  # Replace 'your_image' with your actual image array

                # Define the desired color and a tolerance
                desired_color = (191, 48, 191)  # Red in RGB
                tolerance = 80  # Adjust this tolerance as needed

                # Create a mask for the desired color
                lower_bound = np.array(desired_color) - tolerance
                upper_bound = np.array(desired_color) + tolerance
                mask = np.all((image >= lower_bound) & (image <= upper_bound), axis=2)

                # Color the masked pixels white
                result = np.zeros_like(image)
                result[mask] = (255,255,255)
                image = result
            col.append(image)
        rows.append(col)
    return np.vstack([np.hstack(r) for r in rows])

def largest_smaller_divisible(X, Z):
    # Start with the largest possible X (Y - 1) and decrement it until it's divisible by Z
    while X % Z != 0:
        X -= 1
    return X

def get_image_in_right_dimensions(imarray, dimensions):
    n_width = largest_smaller_divisible(imarray.shape[0], dimensions[0])
    n_height = largest_smaller_divisible(imarray.shape[1], dimensions[1])

    delta_w = int((imarray.shape[0]-n_width)/2)
    delta_h = int((imarray.shape[1]-n_height)/2)

    temp_cropped = imarray[delta_w:imarray.shape[0]-delta_w, delta_h:imarray.shape[1]-delta_h, :]

    # In case one was indivisible, there is a single pixel that we remove by going this extra step

    return temp_cropped[:n_width, :n_height, :]

def clean_up_disk(area_id):
    folder_path = ''  # Replace with the path to your folder

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if the file contains "Hem" in its name
        if area_id in filename and filename.endswith(".png"):
            # Check if the file is indeed a file (not a subdirectory)
            if os.path.isfile(file_path):
                # Delete the file
                os.remove(file_path)
                print(f"Deleted: {filename}")
