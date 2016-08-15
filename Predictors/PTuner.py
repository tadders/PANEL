import math

def _gen_fine_intervals(units, interval_size, num_intervals=8, upper_limit=None, lower_limit=None):
    if isinstance(units, int):
        cast = int
    elif isinstance(units, float):
        cast = float
    else:
        raise TypeError("units must be numeric")
    if int(interval_size) == 0:
        interval_size = 1
    num_new_values = num_intervals - 1
    lower_l = units - interval_size * math.ceil(num_new_values / 2.0)
    lower_limit = max(x for x in [lower_l, lower_limit] if x is not None)
    upper_l = units + interval_size * math.floor(num_new_values / 2.0)
    upper_limit = min(x for x in [upper_l, upper_limit] if x is not None)

    return list(range(int(lower_limit), int(upper_limit), int(interval_size)))