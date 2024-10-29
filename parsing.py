import os
import csv
import numpy as np
import pandas as pd


def extract_features(data, label):
    data = np.array(data)
    mean_x = np.mean(data[:, 0])
    std_x = np.std(data[:, 0])
    mean_y = np.mean(data[:, 1])
    std_y = np.std(data[:, 1])
    mean_z = np.mean(data[:, 2])
    std_z = np.std(data[:, 2])
    return [
        mean_x,
        std_x,
        mean_y,
        std_y,
        mean_z,
        std_z,
        label,
    ]


def extract_features_extra(data, label):
    data = np.array(data)
    mean_x = np.mean(data[:, 0])
    std_x = np.std(data[:, 0])
    med_x = np.median(data[:, 0])
    mean_y = np.mean(data[:, 1])
    std_y = np.std(data[:, 1])
    med_y = np.median(data[:, 1])
    mean_z = np.mean(data[:, 2])
    std_z = np.std(data[:, 2])
    med_z = np.median(data[:, 2])

    rms_x = np.sqrt(np.mean(data[:, 0] ** 2))
    rms_y = np.sqrt(np.mean(data[:, 1] ** 2))
    rms_z = np.sqrt(np.mean(data[:, 2] ** 2))
    return [
        mean_x,
        std_x,
        med_x,
        rms_x,
        mean_y,
        std_y,
        med_y,
        rms_y,
        mean_z,
        std_z,
        med_z,
        rms_z,
        label,
    ]


def process_file(file_path, label, output_writer, window=1, extra_features=False):
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        # data = [[float(row[3]), float(row[4]), float(row[5])] for row in reader]
        data = []

        next(reader)
        for row in reader:
            if len(row) < 6:
                break
            data.append([float(row[3]), float(row[4]), float(row[5])])
            if len(data) == 100 * window:
                features = (
                    extract_features(data, label)
                    if not extra_features
                    else extract_features_extra(data, label)
                )
                output_writer.writerow(features)
                data = data[100:] if window != 1 else []


def main(output_file="features.csv", extra_features=False, window=1):
    hand_wash_folder = "./hand_wash/"
    no_hand_wash_folder = "./non_hand_wash/"

    with open(output_file, "w", newline="") as file:
        writer = csv.writer(file)
        if not extra_features:
            writer.writerow(
                ["mean_x", "std_x", "mean_y", "std_y", "mean_z", "std_z", "Activity"]
            )
        else:
            writer.writerow(
                [
                    "mean_x",
                    "std_x",
                    "med_x",
                    "rms_x",
                    "mean_y",
                    "std_y",
                    "med_y",
                    "rms_y",
                    "mean_z",
                    "std_z",
                    "med_z",
                    "rms_z",
                    "Activity",
                ]
            )

        for file_name in os.listdir(hand_wash_folder):
            file_path = os.path.join(hand_wash_folder, file_name)
            process_file(
                file_path,
                "hand_wash",
                writer,
                extra_features=extra_features,
                window=window,
            )

        for file_name in os.listdir(no_hand_wash_folder):
            file_path = os.path.join(no_hand_wash_folder, file_name)
            process_file(
                file_path,
                "no_hand_wash",
                writer,
                extra_features=extra_features,
                window=window,
            )


# to be used in the event that files are improperly named again
def rename_file(replacement_string, rename_string):
    # hand_wash_files = os.listdir("./hand_wash/")
    # non_hand_wash_files = os.listdir("./non_hand_wash/")
    files = os.listdir(".")

    for f in files:
        if replacement_string in f:
            os.rename(
                os.path.join("./", f),
                os.path.join("./", f.replace(replacement_string, rename_string)),
            )
    # for f in non_hand_wash_files:
    #     os.rename(
    #         os.path.join("./non_hand_wash/", f),
    #         os.path.join(
    #             "./non_hand_wash/", f.replace(replacement_string, rename_string)
    #         ),
    #     )


if __name__ == "__main__":
    main("features_4s.csv", extra_features=True, window=4)
