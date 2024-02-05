import json
import os
import sys
import time

from prettytable import PrettyTable

def confirm_env_variable(env_var, append=None):
    if env_var in os.environ:
        print("[CONFIRMED] Environment variable exists: %s" % env_var)
        if append:
            path = os.path.join(os.environ[env_var], append)
            sys.path.append(path)
            print("[SUCCESS] Added module directory: %s" % path)
    else:
        raise ImportError("Please declare the environment variable '%s'" % env_var)



def get_json(file_path):    # Read json file, return as dict
    try:
        with open(file_path, 'r') as file:
            json_data = json.load(file)
    except FileNotFoundError:
        print(f"[ERROR] Cannot locate: %s" % (file_path))
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



def show_progress_bar(name_of_operation, start_time, progress, target, end_line=''):    # Just printing progress bar with ETA
    bar_length = 50
    progress_fraction = progress / target
    filled_length = int(bar_length * progress_fraction)
    bar = 'X' * filled_length + '-' * (bar_length - filled_length)
    elapsed_time = time.time() - start_time
    remaining_time = ((elapsed_time / progress_fraction) - elapsed_time) if progress_fraction else 0
    remaining_time = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
    print(f'\r[%s PROGRESS]: |%s| %.2f%%, ETA: %s' % (name_of_operation.upper(), bar, progress_fraction * 100, remaining_time), end=end_line)



def show_progress(name_of_operation, start_time, progress, target, end_line=''):    # Just printing progress bar with ETA
    progress_fraction = progress / target
    elapsed_time = time.time() - start_time
    remaining_time = ((elapsed_time / progress_fraction) - elapsed_time) if progress_fraction else 0
    remaining_time = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
    print(f'\r[%s PROGRESS]: %.2f%%, ETA: %s' % (name_of_operation.upper(), progress_fraction * 100, remaining_time), end=end_line)


def remove_double_quotes(text):
    text = str(text).replace('"','')
    return text

def list_to_string(from_list, separator=', '):
    out_str = ""
    first_time = True
    for item in from_list:
        if first_time:
            out_str = str(item)
            first_time = False
        else:
            out_str = "%s%s%s" % (out_str, separator, item)

    return out_str

def df_to_prettytable(df, header_message="DATA"):
    table = PrettyTable()
    table.field_names = df.columns.tolist()
    for _, row in df.iterrows():
        table.add_row(row.tolist())
    print(f"##### {header_message} #####")
    print(table)