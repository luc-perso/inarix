import os


def build_data_paths(data_path, grain_types, devices):
    data_paths = []
    hot_encoded_label = []
    n = len(grain_types)
    for num, type in enumerate(grain_types):
        hot_encoded_num = [0] * n
        hot_encoded_num[num] = 1
        for device in devices:
            path = os.path.join(data_path, type + "_" + device)
            if not os.path.exists(path):
                continue
            files = [f for f in os.listdir(path) if f[-4:] == ".jpg"]
            if files != []:
                data_paths.append(path)
                hot_encoded_label.append(hot_encoded_num)

    return data_paths, hot_encoded_label
