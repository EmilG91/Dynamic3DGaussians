import json
import os
import math
import numpy as np
import argparse


# Convert NumPy arrays from lists
def convert_to_numpy(data):
    if isinstance(data, dict):
        return {k: convert_to_numpy(v) for k, v in data.items()}
    elif isinstance(data, list):
        try:
            return np.array(data)
        except:
            return [convert_to_numpy(item) for item in data]
    else:
        return data


def multiply_nested_data(data, multiplier):
    if isinstance(data, dict):
        return {k: multiply_nested_data(v, multiplier) for k, v in data.items()}
    elif isinstance(data, list):
        return [multiply_nested_data(item, multiplier) for item in data]
    elif isinstance(data, np.ndarray):
        return (
            data * multiplier
        ).tolist()  # Convert back to list for JSON serialization
    else:
        return data


def args_parser():
    parser = argparse.ArgumentParser(description="Multiply JSON data by a float value.")
    parser.add_argument("-i", help="Path to the input JSON file")
    parser.add_argument("-o", default=None, help="Path to the output JSON file")

    return parser.parse_args()


if __name__ == "__main__":

    args = args_parser()

    pathToJson = args.i
    savePath = args.o
    
    if savePath is None:
        directory, filename = os.path.split(pathToJson)
        name, extension = os.path.splitext(filename)
        new_name = f"{name}_edited{extension}"
        savePath = os.path.join(directory, new_name)

    md = json.load(open(pathToJson, "r"))

    old_width = md["w"]
    old_height = md["h"]
    new_width = 2560
    new_height = 1440

    width_ratio = float(new_width / old_width)
    height_ratio = float(new_height / old_height)

    if not math.isclose(width_ratio, height_ratio):
        print(
            f"Error width_ratio = {width_ratio} is not equal to height_ratio = height_ratio = {height_ratio}"
        )
        exit()

    old_k_data = convert_to_numpy(md["k"])
    new_k_data = multiply_nested_data(old_k_data, width_ratio)

    md["k"] = new_k_data
    md["w"] = new_width
    md["h"] = new_height

    with open(savePath, "w") as file:
        json.dump(md, file)

    print(f"Saved Converted File at {savePath}")
