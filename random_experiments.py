from trajectories import get_trajectories, plot_them
import numpy as np
import tensorflow as tf

from models import build_model, fit
from utils import ECG_LEN

if __name__ == "__main__":
    epochs = 4000
    start_mask = 200
    end_mask = 1100
    patients = ["0"]  # ,'50483780']

    x_ph = tf.placeholder(tf.float32, [None, ECG_LEN])

    for patient in patients:
        result_table = np.zeros((10, 4))
        i = 0
        for param in [3, 7]:
            for iteration in range(10):
                print(i)
                i += 1
                model_output = build_model(param, x_ph)

                error_inside, error_outside = fit(model_output, start_mask, end_mask, epochs, patient, x_ph)
                print(patient)
                print(error_inside, error_outside)

                if param == 3:
                    result_table[iteration, 0] = error_inside
                    result_table[iteration, 1] = error_outside
                else:
                    result_table[iteration, 2] = error_inside
                    result_table[iteration, 3] = error_outside

        np.savetxt('900_' + str(patient) + '.csv', result_table, delimiter=';')
