import json
import numpy as np

from trajectories import get_trajectories


def create_mask(start, end):
    result = np.ones((1, ECG_LEN, 1))
    result[:, start:end, :] = np.zeros((1, end - start, 1))

    return result


def data_generator(data, patient):
    if patient != "0":
        v6_data = data[patient]["Leads"]["v6"]["Signal"]
        v6_data = np.array(v6_data)
        v6_data = np.expand_dims(v6_data, axis=0)
    elif patient == "0":
        YS = get_trajectories(mu_desired=-5.475, std_desired=188.577, num_of_trajectories=5, nb_of_samples=5000)
        v6_data = YS[0:1]
    while True:
        yield v6_data


ECG_LEN = 5000


def open_json(link):
    f = open(link, 'r')
    data = json.load(f)
    return data