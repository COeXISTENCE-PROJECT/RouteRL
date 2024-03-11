import json
import os
import sys
import time

from prettytable import PrettyTable


def confirm_env_variable(env_var, append=None): # RK: describe in doc string variables.
    if env_var in os.environ:
        print("[CONFIRMED] Environment variable exists: %s" % env_var)
        if append:
            path = os.path.join(os.environ[env_var], append)
            sys.path.append(path)
            print("[SUCCESS] Added module directory: %s" % path)
    else:
        raise ImportError("Please declare the environment variable '%s'" % env_var)
    


def get_params(file_path):      # Read params.json, resolve dependencies
    params = read_json(file_path)
    params = resolve_param_dependencies(params)
    return params



def resolve_param_dependencies(params):    # Resolving dependent parameters in params.json
    for category, settings in params.items():
        for key, value in settings.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                path = value[2:-1].split('.')   # Extract the reference path
                ref_value = params
                for step in path:
                    ref_value = ref_value.get(step, {})
                if not isinstance(ref_value, dict):     # Ensure it's not a nested structure
                    params[category][key] = ref_value
    return params



def read_json(file_path):    # Read json file, return as dict
    try:
        with open(file_path, 'r') as file:
            file_data = json.load(file)
    except FileNotFoundError:
        print(f"[ERROR] Cannot locate: %s" % (file_path))
        raise
    return file_data



def make_dir(folders, filename=None):    # Make dir if not exists, make full path
    if not folders:
        print("[ERROR] Expected at least one folder name.")
        raise ValueError
    if not isinstance(folders, list):
        folders = [folders]
    path = str()
    for folder_name in folders:
        path = os.path.join(path, folder_name)
        if not os.path.exists(path):
            os.mkdir(path)
    if filename: path = os.path.join(path, filename)
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


def string_to_list(text, seperator, brackets=False):
    if brackets:
        text = text.strip("[]")
    elements = text.split(seperator)
    return elements



def df_to_prettytable(df, header_message="DATA", print_every=1):
    table = PrettyTable()
    table.field_names = df.columns.tolist()
    for index, row in df.iterrows():
        if not (index % print_every):
            table.add_row(row.tolist())
    print(f"##### {header_message} #####")
    print(table)



def running_average(values, last_n = 0): 
    # last_n -> -1 disables the averaging, 0 averages all, n averages the last n
    if last_n < 0: return values
    if last_n == 0: start_from = 0
    
    running_sum, running_averages = 0, list()
    for idx, _ in enumerate(values):
        if last_n > 0: start_from = max(0, idx - last_n)
        divide_by = idx - start_from + 1
        running_sum = sum(values[start_from:idx+1])
        running_averages.append(running_sum / divide_by)
    return running_averages
    