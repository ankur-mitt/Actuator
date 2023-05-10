import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from consts import AngularScales, Columns, LinearScales


def inner_angle(vector_1: np.ndarray, vector_2: np.ndarray) -> float:
    dot_product = np.dot(vector_1, vector_2)
    cross_product = np.cross(vector_1, vector_2)
    return np.arctan2(cross_product, dot_product)


def process_dataset():
    annotated_df = pd.read_csv("Actuator-Dataset - Sheet2.csv")
    # convert all missing values to NaN and drop those readings
    annotated_df = annotated_df.apply(pd.to_numeric, errors='coerce').dropna()
    # calculate input vector value and normalize input(X, Y) to make it between 0-1
    annotated_df[Columns.INPUT_VECTOR] = annotated_df.apply(lambda entry: np.array([
        entry[Columns.INPUT_X] / AngularScales.L_CONTROLLER,
        entry[Columns.INPUT_Y] / AngularScales.R_CONTROLLER,
    ]), axis=1)
    # calculate reference vector value
    annotated_df[Columns.REFERENCE_VECTOR] = annotated_df.apply(lambda entry: np.array([
        entry[Columns.REFERENCE_X] - entry[Columns.ORIGIN_X],
        # reverse direction because pixels are numbered from top left corner
        entry[Columns.ORIGIN_Y] - entry[Columns.REFERENCE_Y],
    ]), axis=1)
    # calculate target vector value
    annotated_df[Columns.TARGET_VECTOR] = annotated_df.apply(lambda entry: np.array([
        entry[Columns.TARGET_X] - entry[Columns.ORIGIN_X],
        # reverse direction because pixels are numbered from top left corner
        entry[Columns.ORIGIN_Y] - entry[Columns.TARGET_Y],
    ]), axis=1)
    # calculate angle between target and reference
    annotated_df[Columns.INNER_ANGLE] = annotated_df.apply(lambda entry: inner_angle(
        vector_1=entry[Columns.REFERENCE_VECTOR],
        vector_2=entry[Columns.TARGET_VECTOR],
    ), axis=1)
    annotated_df[Columns.INNER_ANGLE_DEG] = annotated_df[Columns.INNER_ANGLE] * (180 / np.pi)
    # calculate lengths of the output vectors
    annotated_df[Columns.OUTPUT_X] = annotated_df.apply(
        lambda entry: np.linalg.norm(entry[Columns.TARGET_VECTOR]) * np.cos(entry[Columns.INNER_ANGLE]),
        axis=1)
    annotated_df[Columns.OUTPUT_Y] = annotated_df.apply(
        lambda entry: np.linalg.norm(entry[Columns.TARGET_VECTOR]) * np.sin(entry[Columns.INNER_ANGLE]),
        axis=1)
    # calculate the final output vector
    annotated_df[Columns.OUTPUT_VECTOR] = annotated_df.apply(lambda entry: np.array([
        entry[Columns.OUTPUT_X],
        entry[Columns.OUTPUT_Y],
    ]), axis=1)
    # normalize the output vector to make it between ScaleLimits
    annotated_df[Columns.OUTPUT_VECTOR] = annotated_df.apply(lambda entry: np.round(
        entry[Columns.OUTPUT_VECTOR] * LinearScales.ScaleLength / np.linalg.norm(entry[Columns.REFERENCE_VECTOR])
    ), axis=1)

    annotated_df.to_csv("final_dataset.csv", index=False)
    # pprint(annotated_df)

    return annotated_df


def plot_curves(dataset: pd.DataFrame, key: int):
    scatter_dataset: pd.Series = dataset.apply(lambda entry: [
        entry[Columns.OUTPUT_VECTOR][0],
        entry[Columns.OUTPUT_VECTOR][1],
        entry[Columns.INPUT_X] if key == 0 else entry[Columns.INPUT_Y],
    ], axis=1)
    # pprint(scatter_dataset)
    curves = {}
    for index, [x, y, curve] in scatter_dataset.items():
        if curve not in curves.keys():
            curves[curve] = ([], [])
        curves[curve][0].append(x)
        curves[curve][1].append(y)
    for curve in sorted(curves.keys()):
        plt.plot(curves[curve][0], curves[curve][1], marker=".", linestyle="dashed",
                 label=f"{'L' if key == 0 else 'R'}={curve}°")
    plt.axis("square")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    plt.show()
