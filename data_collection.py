import os
import time
from concurrent.futures import ThreadPoolExecutor
from pprint import pprint

import numpy as np
import pandas as pd

from consts import Columns
from image_processing import annotate_image, capture_image, crop_image, mark_image
from motor_controller import connect_arduino


def collect_data(
        l_space: np.ndarray = np.linspace(0, 2000, 51, dtype=int),
        r_space: np.ndarray = np.linspace(-500, 0, 21, dtype=int)
):
    l_controller, r_controller = connect_arduino()
    for r_pos in r_space:
        for l_pos in l_space:
            l_controller.set_position(l_pos), r_controller.set_position(r_pos)
            input_mapping = l_controller.get_position(), r_controller.get_position()
            time.sleep(5.0)
            file_name = f"{str(input_mapping)}.jpg"
            capture_image(file_name)
            crop_image(file_name)
            mark_image(file_name)
            output_mapping = annotate_image(file_name)
            pprint(f"{input_mapping} ---> {output_mapping}")


def annotate_data():
    # collect_data(l_space, r_space)
    l_marked = list(filter(lambda file_name: file_name.endswith(".jpg"), os.listdir("marked")))
    pprint(len(l_marked))

    input_x, input_y, target_x, target_y, origin_x, origin_y, reference_x, reference_y = [], [], [], [], [], [], [], []
    with ThreadPoolExecutor() as pool:
        l_clusters = list(pool.map(annotate_image, l_marked))
        pprint(l_clusters)

        for index, cluster in enumerate(l_clusters):
            [ix, iy] = list(int(x) for x in l_marked[index].removesuffix(".jpg").strip("()").split(","))
            [tx, ty], [ox, oy], [rx, ry] = cluster.astype(int).tolist()
            input_x.append(ix), input_y.append(iy)
            target_x.append(tx), target_y.append(ty)
            origin_x.append(ox), origin_y.append(oy)
            reference_x.append(rx), reference_y.append(ry)

    df = pd.DataFrame()
    df[Columns.INPUT_X] = pd.Series(input_x)
    df[Columns.INPUT_Y] = pd.Series(input_y)
    df[Columns.ORIGIN_X] = pd.Series(origin_x)
    df[Columns.ORIGIN_Y] = pd.Series(origin_y)
    df[Columns.REFERENCE_X] = pd.Series(reference_x)
    df[Columns.REFERENCE_Y] = pd.Series(reference_y)
    df[Columns.TARGET_X] = pd.Series(target_x)
    df[Columns.TARGET_Y] = pd.Series(target_y)
    pprint(df)
    df.to_csv("guessed_dataset.csv", index=False)
