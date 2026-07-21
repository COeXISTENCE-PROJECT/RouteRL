import json
import os
import requests
import sys
import time

from prettytable import PrettyTable

from routerl.keychain import Keychain as kc


def get_params(file_path, resolve=True, update=None):
    # Read params.json, resolve dependencies
    params = read_json(file_path)
    if (update is not None) and isinstance(update, dict):
        update_params(params, update)
    params = resolve_ods(params)
    if resolve:
        params = resolve_param_dependencies(params)
    return params


def update_params(old_params: dict, new_params: dict):
    for key, value in new_params.items():
        if not (key in old_params):
            raise ValueError(f"Invalid parameter: {key}")
        elif isinstance(old_params.get(key), str) and old_params.get(key).startswith("${"):
            raise ValueError(f"Invalid parameter: {key}")
        elif (not isinstance(value, dict)) and isinstance(old_params.get(key), dict):
            raise ValueError(f"Can't update group parameter {key} with value: {value}")
        elif isinstance(value, dict) and isinstance(old_params.get(key), dict):
            update_params(old_params[key], value)
        else:
            old_params[key] = value


def resolve_ods(params):
    if not (params[kc.SIMULATOR][kc.NETWORK_NAME] in kc.NETWORK_NAMES):
        if params[kc.PATH_GEN][kc.ORIGINS] == "default" or params[kc.PATH_GEN][kc.DESTINATIONS] == "default":
            raise ValueError("Cannot use default origins/destinations with custom network.")
        return params
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    default_od_path = os.path.join(curr_dir, kc.DEFAULT_ODS_PATH)
    default_ods = read_json(default_od_path).get(params[kc.SIMULATOR][kc.NETWORK_NAME], {})
    if params[kc.PATH_GEN][kc.ORIGINS] == "default":
        params[kc.PATH_GEN][kc.ORIGINS] = default_ods[kc.ORIGINS]
    if params[kc.PATH_GEN][kc.DESTINATIONS] == "default":
        params[kc.PATH_GEN][kc.DESTINATIONS] = default_ods[kc.DESTINATIONS]
    if params[kc.SIMULATOR][kc.NETWORK_NAME] == kc.TWO_ROUTE_YIELD:
        params[kc.PATH_GEN][kc.NUMBER_OF_PATHS] = 2
    if params[kc.SIMULATOR][kc.NETWORK_NAME] == kc.MANHATTAN: # manhattan network is big so we store the network files in zenodo
        zenodo_record_id = kc.ZENODO_RECORD_ID
        api_url = f"https://zenodo.org/api/records/{zenodo_record_id}"

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        save_folder = os.path.join(curr_dir, kc.NETWORK_FOLDER).replace("$net$", kc.MANHATTAN)
        
        response = requests.get(api_url)
        data = response.json()

        # Download all files
        for file_info in data["files"]:
            file_url = file_info["links"]["self"]
            file_name = file_info["key"]

            # Construct the full save path
            save_path = os.path.join(save_folder, file_name)
            response = requests.get(file_url, stream=True)
            
            # Write the file to the specified folder
            with open(save_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
    return params


def resolve_param_dependencies(params):  # Resolving dependent parameters in params.json
    for category, settings in params.items():
        for key, value in settings.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                path = value[2:-1].split('.')  # Extract the reference path
                ref_value = params
                for step in path:
                    ref_value = ref_value.get(step, {})
                if not isinstance(ref_value, dict):  # Ensure it's not a nested structure
                    params[category][key] = ref_value
    return params


def confirm_env_variable(env_var, append=None):
    if env_var in os.environ:
        print("[CONFIRMED] Environment variable exists: %s" % env_var)
        if append:
            path = os.path.join(os.environ[env_var], append)
            sys.path.append(path)
            print("[SUCCESS] Added module directory: %s" % path)
    else:
        raise ImportError("Please declare the environment variable '%s'" % env_var)


def read_json(file_path):  # Read json file, return as dict
    try:
        with open(file_path, 'r') as file:
            file_data = json.load(file)
    except FileNotFoundError:
        print(f"[ERROR] Cannot locate: %s" % (file_path))
        raise
    return file_data


def make_dir(folders, filename=None):  # Make dir if not exists, make full path
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


def show_progress_bar(message, start_time, progress, target, end_line=''):  # Just printing progress bar with ETA
    bar_length = 50
    progress_fraction = progress / target
    filled_length = int(bar_length * progress_fraction)
    bar = 'X' * filled_length + '-' * (bar_length - filled_length)
    elapsed_time = time.time() - start_time
    remaining_time = ((elapsed_time / progress_fraction) - elapsed_time) if progress_fraction else 0
    remaining_time = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
    print(f'\r[%s]: |%s| %.2f%%, ETA: %s' % (message.upper(),
                                             bar,
                                             progress_fraction * 100,
                                             remaining_time),
          end=end_line)


def show_progress(name_of_operation, start_time, progress, target, end_line=''):  # Just printing progress bar with ETA
    progress_fraction = progress / target
    elapsed_time = time.time() - start_time
    remaining_time = ((elapsed_time / progress_fraction) - elapsed_time) if progress_fraction else 0
    remaining_time = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
    print(f'\r[%s PROGRESS]: %.2f%%, ETA: %s' % (name_of_operation.upper(),
                                                 progress_fraction * 100,
                                                 remaining_time),
          end=end_line)


def remove_double_quotes(text):
    text = str(text).replace('"', '')
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


def df_to_prettytable(df, header_message=None, print_every=1):
    table = PrettyTable()
    table.field_names = df.columns.tolist()
    for index, row in df.iterrows():
        if not (index % print_every):
            table.add_row(row.tolist())
    if header_message:
        print(f"##### {header_message} #####")
    print(table)


def running_average(values, last_n=0):
    # last_n -> -1 disables the averaging, 0 averages all, n averages the last n
    if last_n < 0: return values

    running_sum, running_averages = 0, list()
    for idx, _ in enumerate(values):
        start_from = max(0, idx - last_n) if last_n > 0 else 0
        divide_by = idx - start_from + 1
        running_sum = sum(values[start_from:idx + 1])
        running_averages.append(running_sum / divide_by)
    return running_averages
