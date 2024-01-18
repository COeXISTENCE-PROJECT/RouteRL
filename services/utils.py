import json
import os
import time


def get_json(file_path):    # Read json file, return as dict
    try:
        with open(file_path, 'r') as file:
            json_data = json.load(file)
    except FileNotFoundError:
        print(f"[EROR] Cannot locate: %s" % (file_path))
        raise
    return json_data



def make_dir(dir1, dir2, filename):    # Make dir if not exists, make full path
    if not os.path.exists(dir1):
        os.mkdir(dir1)
    dir2 = os.path.join(dir1, dir2)
    if not os.path.exists(dir2):
        os.mkdir(dir2)
    path = os.path.join(dir2, filename)
    return path



def progress_bar(start_time, progress, target):    # Just printing progress bar with ETA
    bar_length = 50
    progress_fraction = progress / target
    filled_length = int(bar_length * progress_fraction)
    bar = 'X' * filled_length + '-' * (bar_length - filled_length)
    elapsed_time = time.time() - start_time
    remaining_time = ((elapsed_time / progress_fraction) - elapsed_time) if progress_fraction else 0
    print(f'\rProgress: |%s| %.2f%%, ETA: %.2f seconds' % (bar, progress_fraction * 100, remaining_time), end='')