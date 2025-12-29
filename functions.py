import numpy as np


# Return the numerator value of the closest k/4 to the number
def round_to_nearest_k_over_4(number):
    k_over_4 = 1 / 4
    rounded_number = np.round(number / k_over_4) * k_over_4
    return int(rounded_number * 4)


# Return the numerator value of the closest k/8 to the number
def round_to_nearest_k_over_8(number):
    k_over_8 = 1 / 8
    rounded_number = np.round(number / k_over_8) * k_over_8
    return int(rounded_number * 8)


# Return the numerator value of the closest k/16 to the number
def round_to_nearest_k_over_16(number):
    k_over_16 = 1 / 16
    rounded_number = np.round(number / k_over_16) * k_over_16
    return int(rounded_number * 16)


# Function to find the x-value for a given target area
def find_x_for_area(target_area, x, y):
    cumulative_area = 0
    for i in range(1, len(x)):
        x0, x1 = x[i - 1], x[i]
        y0, y1 = y[i - 1], y[i]
        trapezoid_area = (y0 + y1) * (x1 - x0) / 2

        if cumulative_area + trapezoid_area >= target_area:
            remaining_area = target_area - cumulative_area
            base = x1 - x0
            height_diff = y1 - y0
            if height_diff == 0:
                # If y0 == y1, the trapezoid is a rectangle
                return x0 + remaining_area / y0
            else:
                a = y0
                b = height_diff / base
                a_q = b
                b_q = 2 * a - 2 * x0 * b
                c_q = b * (x0 ** 2) - 2 * a * x0 - 2 * remaining_area
                discriminant = (b_q ** 2) - (4 * a_q * c_q)
                x_sol1 = (-b_q + np.sqrt(discriminant)) / (2 * a_q)
                x_sol2 = (-b_q - np.sqrt(discriminant)) / (2 * a_q)
                if ((x_sol1 >= x0) and (x_sol1 <= x1)):
                    x_solution = x_sol1
                else:
                    x_solution = x_sol2
                return x_solution
        cumulative_area += trapezoid_area
    return x[-1]


# Given a number, it creates the ISC stream for it
def create_stream(n):
    if (n % 8 != 0):
        max_value = n // 8 + 1
    else:
        max_value = n // 8

    stream = np.zeros(8, dtype=np.int_)
    sum = 0

    for i in range(8):

        diff = n - sum
        if (diff >= max_value):
            choice = np.random.randint(0, max_value + 1)
        else:
            choice = np.random.randint(0, diff + 1)

        if (diff == (max_value * (8 - i))):
            for k in range(i, 8):
                stream[k] = max_value
            return stream
        elif (diff > (max_value * (8 - i))):
            stream[i - 1] = max_value

        stream[i] = choice
        sum = np.sum(stream)

        if (i == 7):
            last_diff = n - sum
            stream[i] += last_diff

    return stream


# Given a number, it creates the ISC stream for it
def create_stream_4bit(n):
    if (n % 4 != 0):
        max_value = n // 4 + 1
    else:
        max_value = n // 4

    stream = np.zeros(4, dtype=np.int_)
    sum = 0

    for i in range(4):

        diff = n - sum
        if (diff >= max_value):
            choice = np.random.randint(0, max_value + 1)
        else:
            choice = np.random.randint(0, diff + 1)

        if (diff == (max_value * (4 - i))):
            for k in range(i, 4):
                stream[k] = max_value
            return stream
        elif (diff > (max_value * (4 - i))):
            stream[i - 1] = max_value

        stream[i] = choice
        sum = np.sum(stream)

        if (i == 3):
            last_diff = n - sum
            stream[i] += last_diff

    return stream


def create_stream_16bit(n):
    if (n % 16 != 0):
        max_value = n // 16 + 1
    else:
        max_value = n // 16

    stream = np.zeros(16, dtype=np.int_)
    sum = 0

    for i in range(16):

        diff = n - sum
        if (diff >= max_value):
            choice = np.random.randint(0, max_value + 1)
        else:
            choice = np.random.randint(0, diff + 1)

        if (diff == (max_value * (16 - i))):
            for k in range(i, 16):
                stream[k] = max_value
            return stream
        elif (diff > (max_value * (16 - i))):
            stream[i - 1] = max_value

        stream[i] = choice
        sum = np.sum(stream)

        if (i == 15):
            last_diff = n - sum
            stream[i] += last_diff

    return stream


def MAC_4bit_software(weights, inputs, iterations):
    n = weights.shape[0]
    results = np.zeros(iterations)

    for k in range(iterations):

        multiply_res = []
        for p in range(n):

            w = weights[p]
            x = inputs[p]

            # Stream for x
            ones_x = np.array([1 for _ in range(x)], dtype=np.int_)
            zeros_x = np.array([0 for _ in range(4 - x)], dtype=np.int_)
            stream_x = np.append(ones_x, zeros_x)
            np.random.shuffle(stream_x)

            # Stream for w
            sum_w = round_to_nearest_k_over_4(w)
            stream_w = create_stream_4bit(sum_w)

            res = np.zeros(4, dtype=np.int_)
            for q in range(4):

                if (stream_x[q] == 1):
                    res[q] = stream_w[q]
                else:
                    res[q] = 0

            multiply_res.append(res)

        multiply_res = np.asarray(multiply_res)
        add_res = np.sum(multiply_res, axis=0)

        answer = (np.sum(add_res))
        results[k] = answer

    w_avg = np.mean(results)
    return w_avg


def MAC_4bit(weights, inputs, resistance_val, thresholds, iterations):
    N = resistance_val.shape[0]
    n = weights.shape[0]
    results = np.zeros(iterations)

    for k in range(iterations):

        multiply_res = []
        for p in range(n):

            w = weights[p]
            x = inputs[p]

            # Stream for x
            threshold = thresholds[x]
            resistance_arr = resistance_val[((4 * n * k + 4 * p) % N):((4 * n * k + 4 * p + 4) % N)]
            resistance_log = np.log10(resistance_arr)
            stream_x = np.zeros(4, dtype=np.int_)
            for i in range(resistance_arr.shape[0]):
                if (resistance_log[i] > threshold):
                    stream_x[i] = 0
                else:
                    stream_x[i] = 1

            # ones_x = np.array([1 for _ in range(x)], dtype=np.int_)
            # zeros_x = np.array([0 for _ in range(8 - x)], dtype=np.int_)
            # stream_x = np.append(ones_x, zeros_x)
            # np.random.shuffle(stream_x)

            # Stream for w
            sum_w = round_to_nearest_k_over_4(w)
            stream_w = create_stream_4bit(sum_w)

            res = np.zeros(4, dtype=np.int_)
            for q in range(4):

                if (stream_x[q] == 1):
                    res[q] = stream_w[q]
                else:
                    res[q] = 0

            multiply_res.append(res)

        multiply_res = np.asarray(multiply_res)
        add_res = np.sum(multiply_res, axis=0)

        answer = (np.sum(add_res))
        results[k] = answer

    w_avg = np.mean(results)
    return w_avg


def MAC_8bit_software(weights, inputs, iterations):
    n = weights.shape[0]
    results = np.zeros(iterations)

    for k in range(iterations):

        multiply_res = []
        for p in range(n):

            w = weights[p]
            x = inputs[p]

            # Stream for x
            ones_x = np.array([1 for _ in range(x)], dtype=np.int_)
            zeros_x = np.array([0 for _ in range(8 - x)], dtype=np.int_)
            stream_x = np.append(ones_x, zeros_x)
            np.random.shuffle(stream_x)

            # Stream for w
            sum_w = round_to_nearest_k_over_8(w)
            stream_w = create_stream(sum_w)

            res = np.zeros(8, dtype=np.int_)
            for q in range(8):

                if (stream_x[q] == 1):
                    res[q] = stream_w[q]
                else:
                    res[q] = 0

            multiply_res.append(res)

        multiply_res = np.asarray(multiply_res)
        add_res = np.sum(multiply_res, axis=0)

        answer = (np.sum(add_res))
        results[k] = answer

    w_avg = np.mean(results)
    return w_avg


def MAC_8bit(weights, inputs, resistance_val, thresholds, iterations):
    N = resistance_val.shape[0]
    n = weights.shape[0]
    results = np.zeros(iterations)

    for k in range(iterations):

        multiply_res = []
        for p in range(n):

            w = weights[p]
            x = inputs[p]

            # Stream for x
            threshold = thresholds[x]
            resistance_arr = resistance_val[((8 * n * k + 8 * p) % N):((8 * n * k + 8 * p + 8) % N)]
            resistance_log = np.log10(resistance_arr)
            stream_x = np.zeros(8, dtype=np.int_)
            for i in range(resistance_arr.shape[0]):
                if (resistance_log[i] > threshold):
                    stream_x[i] = 0
                else:
                    stream_x[i] = 1

            # Stream for w
            sum_w = round_to_nearest_k_over_8(w)
            stream_w = create_stream(sum_w)

            res = np.zeros(8, dtype=np.int_)
            for q in range(8):

                if (stream_x[q] == 1):
                    res[q] = stream_w[q]
                else:
                    res[q] = 0

            multiply_res.append(res)

        multiply_res = np.asarray(multiply_res)
        add_res = np.sum(multiply_res, axis=0)

        answer = (np.sum(add_res))
        results[k] = answer

    w_avg = np.mean(results)
    return w_avg


def MAC_16bit(weights, inputs, resistance_val, thresholds, iterations):
    N = resistance_val.shape[0]
    n = weights.shape[0]
    results = np.zeros(iterations)

    for k in range(iterations):

        multiply_res = []
        for p in range(n):

            w = weights[p]
            x = inputs[p]

            # Stream for x
            threshold = thresholds[x]
            resistance_arr = resistance_val[((16 * n * k + 16 * p) % N):((16 * n * k + 16 * p + 16) % N)]
            resistance_log = np.log10(resistance_arr)
            stream_x = np.zeros(16, dtype=np.int_)
            for i in range(resistance_arr.shape[0]):
                if (resistance_log[i] > threshold):
                    stream_x[i] = 0
                else:
                    stream_x[i] = 1

            # Stream for w
            sum_w = round_to_nearest_k_over_16(w)
            stream_w = create_stream_16bit(sum_w)

            res = np.zeros(16, dtype=np.int_)
            for q in range(16):

                if (stream_x[q] == 1):
                    res[q] = stream_w[q]
                else:
                    res[q] = 0

            multiply_res.append(res)

        multiply_res = np.asarray(multiply_res)
        add_res = np.sum(multiply_res, axis=0)

        answer = (np.sum(add_res))
        results[k] = answer

    w_avg = np.mean(results)
    return w_avg


def sng_histogram(input, resistance_values, thresholds):
    threshold = thresholds[input]
    bit_stream_length = len(resistance_values)
    sng_stream = np.zeros(bit_stream_length, dtype=np.int_)

    resistance_log_values = np.log10(resistance_values)

    for i in range(bit_stream_length):
        if (threshold >= resistance_log_values[i]):
            sng_stream[i] = 1
        else:
            sng_stream[i] = 0

    return np.sum(sng_stream)