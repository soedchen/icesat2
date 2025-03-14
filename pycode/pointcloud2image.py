import numpy as np
from PIL import Image

# input : shape(N, 4)
#    (x, y, z, intensity)
def pointcloud2image(point_cloud):
    x_size = 640
    y_size = 640
    x_range = 60.0
    y_range = 60.0
    grid_size = np.array([2 * x_range / x_size, 2 * y_range / y_size])
    image_size = np.array([x_size, y_size])
    # [0, 2*range)
    shifted_coord = point_cloud[:, :2] + np.array([x_range, y_range])
    # image index
    index = np.floor(shifted_coord / grid_size).astype(np.int)
    # choose illegal index
    bound_x = np.logical_and(index[:, 0] >= 0, index[:, 0] < image_size[0])
    bound_y = np.logical_and(index[:, 1] >= 0, index[:, 1] < image_size[1])
    bound_box = np.logical_and(bound_x, bound_y)
    index = index[bound_box]
    # show image
    image = np.zeros((640, 640), dtype=np.uint8)
    image[index[:, 0], index[:, 1]] = 255
    res = Image.fromarray(image)
    # rgb = Image.merge('RGB', (res, res, res))
    res.show()