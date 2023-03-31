import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.optimize import curve_fit
from skimage.morphology import medial_axis

from pepper_ws.curve import Curve


def parabola(t, params):
    return params[0] * t ** 2 + params[1] * t + params[2]


def fit_curve_to_mask(mask, pepper_fruit_xywh, pepper_peduncle_xywh):
    curve = Curve()

    medial_img, _ = medial_axis(mask, return_distance=True)
    x, y = np.where(medial_img == 1)

    params1, _ = curve_fit(parabola, y, x)
    fit_curve_x = parabola(y, params1)
    params2, _ = curve_fit(parabola, x, y)
    fit_curve_y = parabola(x, params2)

    if np.linalg.norm(x - fit_curve_x) < np.linalg.norm(y - fit_curve_y):
        curve.parabola_direction = 'vertical'
        curve.params = params1

        # xywh: x: along columns, y: along rows
        if pepper_fruit_xywh[0] < pepper_peduncle_xywh[0]:
            # Sorted assuming that the pepper to the left of the peduncle
            # Sorting with respect to y in ascending order
            sorted_x = np.array([x for _, x in sorted(zip(y, x))])
            sorted_y = np.array([y for y, _ in sorted(zip(y, x))])
        else:
            # Sorted assuming that the pepper to the right of the peduncle
            # Sorting with respect to y in descending order
            sorted_x = np.flip(np.array([x for _, x in sorted(zip(y, x))]))
            sorted_y = np.flip(np.array([y for y, _ in sorted(zip(y, x))]))

        curve.curve_y = sorted_y
        curve.curve_x = parabola(sorted_y, curve.params)
    else:
        curve.parabola_direction = 'horizontal'
        curve.params = params2

        if pepper_fruit_xywh[1] < pepper_peduncle_xywh[1]:
            # Sorted assuming that the pepper is above the peduncle
            # Sorted with respect to x in ascending order
            sorted_x = np.array([x for x, _ in sorted(zip(x, y))])
            sorted_y = np.array([y for _, y in sorted(zip(x, y))])
        else:
            # Sorted assuming that the pepper is below the peduncle
            # Sorted with respect to x in descending order
            sorted_x = np.flip(np.array([x for x, _ in sorted(zip(x, y))]))
            sorted_y = np.flip(np.array([y for _, y in sorted(zip(x, y))]))

        curve.curve_x = sorted_x
        curve.curve_y = parabola(sorted_x, curve.params)

    return curve


def determine_poi(curve, percentage, total_curve_length):
    for idx in range(len(curve.curve_y)):
        curve_length = curve.curve_length(idx)
        if abs(curve_length - percentage * total_curve_length) < 2:
            return curve.curve_x[idx], curve.curve_y[idx]

    return curve.curve_x[len(curve.curve_y) // 2], curve.curve_y[len(curve.curve_y) // 2]


def determine_next_point(curve, poi, pepper_fruit_xywh, pepper_peduncle_xywh):
    if curve.parabola_direction == 'vertical':
        # xywh: x: along columns, y: along rows
        if pepper_fruit_xywh[0] < pepper_peduncle_xywh[0]:
            # Assuming that the pepper to the left of the peduncle
            point_y = poi[1] + 1
            point_x = parabola(point_y, curve.params)
        else:
            # Assuming that the pepper to the right of the peduncle
            point_y = poi[1] - 1
            point_x = parabola(point_y, curve.params)
    else:
        if pepper_fruit_xywh[1] < pepper_peduncle_xywh[1]:
            # Assuming that the pepper is above the peduncle
            point_x = poi[0] + 1
            point_y = parabola(point_x, curve.params)
        else:
            # Assuming that the pepper is below the peduncle
            point_x = poi[0] - 1
            point_y = parabola(point_x, curve.params)

    return point_x, point_y


def draw_poi(one_frame):
    img = np.asarray(Image.open(one_frame.img_path))
    img_name = one_frame.img_path.split('/')[-1].split('.')[0]
    plt.imshow(img)

    for pepper in one_frame.pepper_detections.values():
        poi = pepper.pepper_peduncle.poi
        plt.plot(poi[1], poi[0])

    plt.axis('off')
    plt.savefig(
        f"/home/jy/PycharmProjects/Perception-Resources/yolov8_scripts/src/results_8/{img_name}_poi_result.png",
        bbox_inches='tight', pad_inches=1)
    plt.clf()
    plt.cla()
