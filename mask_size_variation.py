import numpy as np
import tensorflow as tf

from models import build_model, fit
from utils import ECG_LEN

if __name__ == "__main__":
    epochs = 4000
    end_mask = 4700
    patient = "50757660"  # ,'50483780']

    x_ph = tf.placeholder(tf.float32, [None, ECG_LEN])

    for start_mask in range(800, 3900, 300):
        result_table = np.zeros((10, 2))
        print(start_mask)
        i = 0
        param = 3
        for iteration in range(10):
            print(i)
            i += 1
            model_output = build_model(param, x_ph)

            error_inside, error_outside = fit(model_output, start_mask, end_mask, epochs, patient, x_ph)
            print(error_inside, error_outside)

            result_table[iteration, 0] = error_inside
            result_table[iteration, 1] = error_outside

        np.savetxt('mask_' + str(start_mask) + '_good.csv', result_table, delimiter=';')

    for start_mask in range(200, 3900, 300):
        result_table = np.zeros((10, 2))
        print(start_mask)
        i = 0
        param = 7
        for iteration in range(10):
            print(i)
            i += 1
            model_output = build_model(param, x_ph)

            error_inside, error_outside = fit(model_output, start_mask, end_mask, epochs, patient, x_ph)
            print(error_inside, error_outside)

            result_table[iteration, 0] = error_inside
            result_table[iteration, 1] = error_outside

        np.savetxt('mask_' + str(start_mask) + '_bad.csv', result_table, delimiter=';')
