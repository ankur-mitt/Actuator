from http import HTTPStatus

import cv2
import numpy as np
import requests
from PIL import Image
from sklearn.cluster import KMeans


def capture_image(file_name: str):
    ip_webcam_url = "http://172.26.29.61:8080"
    response = requests.get(f"{ip_webcam_url}/shot.jpg")
    if response.status_code == HTTPStatus.OK:
        with open(f"raw/{file_name}", "wb") as img:
            img.write(response.content)
    else:
        raise Exception("Failed to retrieve the image from the given url.")


def crop_image(file_name: str):
    with Image.open(f"raw/{file_name}") as img:
        box = (420, 0, 1180, 720)
        img = img.crop(box)
        img.save(f"cropped/{file_name}")


def mark_image(file_name: str):
    img = cv2.imread(f"cropped/{file_name}")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(img, center, radius, (255, 0, 0), radius)
    cv2.imwrite(f"marked/{file_name}", img)


def annotate_image(file_name: str):
    points = []
    with Image.open(f"marked/{file_name}") as img:
        pixels = img.load()
        width, height = img.size
        for h in range(height):
            for w in range(width):
                r, g, b = pixels[w, h]
                if r <= 5 and g <= 5 and b >= 250:
                    points.append([w, h])
    points = np.array(points)
    kmeans = KMeans(n_clusters=3).fit(points)
    clusters: np.ndarray = np.round(kmeans.cluster_centers_)

    # img = cv2.imread(f"marked/{file_name}")
    # for cluster in clusters:
    #     location = (int(cluster[0]) - 30, int(cluster[1]) + 30)
    #     label = str(cluster.astype(int))
    #     cv2.putText(img, label, location, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    # cv2.imwrite(f"annotated/{file_name}", img)
    # with Image.open(f"annotated/{file_name}") as img:
    #     img.show()

    return clusters


def append_images(images, direction="horizontal", bg_color=(255, 255, 255), alignment="center"):
    """
    Appends images in horizontal/vertical direction.

    Args:
        images: List of PIL images
        direction: direction of concatenation, 'horizontal' or 'vertical'
        bg_color: Background color (default: white)
        alignment: alignment mode if images need padding;
           'left', 'right', 'top', 'bottom', or 'center'

    Returns:
        Concatenated image as a new PIL image object.
    """
    widths, heights = zip(*(i.size for i in images))

    if direction == "horizontal":
        new_width = sum(widths)
        new_height = max(heights)
    else:
        new_width = max(widths)
        new_height = sum(heights)

    new_img = Image.new("RGB", (new_width, new_height), color=bg_color)

    offset = 0
    for img in images:
        if direction == "horizontal":
            y = 0
            if alignment == "center":
                y = int((new_height - img.size[1]) / 2)
            elif alignment == "bottom":
                y = new_height - img.size[1]
            new_img.paste(img, (offset, y))
            offset += img.size[0]
        else:
            x = 0
            if alignment == "center":
                x = int((new_width - img.size[0]) / 2)
            elif alignment == "right":
                x = new_width - img.size[0]
            new_img.paste(img, (x, offset))
            offset += img.size[1]

    return new_img
