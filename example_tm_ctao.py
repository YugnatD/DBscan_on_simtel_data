import numpy as np
import csv

from tm_ctao import Frame
from tm_ctao import HECS
from tm_ctao import DataCube

def load_arc_points_from_csv(filename):
    arc_points = []
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            a = int(row['a'])
            r = int(row['r'])
            c = int(row['c'])
            # a = int(row['a_shrinked'])
            # r = int(row['r_shrinked'])
            # c = int(row['c_shrinked'])
            arc_points.append((a, r, c))
    return arc_points

if __name__ == '__main__':
    # h = HECS.Hecs(10, 10)
    # h.plot()
    arc_points = load_arc_points_from_csv("conversion_table_LST_clusters.csv")
    max_r = max([r for (a, r, c) in arc_points]) + 1
    max_c = max([c for (a, r, c) in arc_points]) + 1

    frame = Frame.Frame(max_r, max_c, arc_points)  # create a frame with the max r and c from the CSV
    # create a list of frames
    # make sure each frame is a copy and not a reference to the same frame
    for i in range(10):
        frame_list = [frame.copy() for i in range(10)]

    data_cube = DataCube.DataCube(frame_list)
    data_cube.dbscan_convolve(3, 3, 15)
    data_cube.plot()
    # neighbors = data_cube.get_frame(0).get_list_neighbors_epsilon(1, 29, 48, 5)
    # # set all pixels in the neighbors to 1
    # for neighbor in neighbors:
    #     data_cube.set_value(0, neighbor[0], neighbor[1], neighbor[2], 1)
    # data_cube.plot()

    # frame.plot()


