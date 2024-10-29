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


def process_file(file_path, label, output_writer):
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        # data = [[float(row[3]), float(row[4]), float(row[5])] for row in reader]
        data = []

        next(reader)
        for row in reader:
            if len(row) < 6:
                break
            data.append([float(row[3]), float(row[4]), float(row[5])])
            if len(data) == 100:  # 1 second of data at 100Hz
                features = extract_features(data, label)
                output_writer.writerow(features)
                data = []


def main():
    hand_wash_folder = "./hand_wash/"
    no_hand_wash_folder = "./non_hand_wash/"
    output_file = "features.csv"

    with open(output_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["mean_x", "std_x", "mean_y", "std_y", "mean_z", "std_z", "Activity"]
        )

        for file_name in os.listdir(hand_wash_folder):
            file_path = os.path.join(hand_wash_folder, file_name)
            process_file(file_path, "hand_wash", writer)

        for file_name in os.listdir(no_hand_wash_folder):
            file_path = os.path.join(no_hand_wash_folder, file_name)
            process_file(file_path, "no_hand_wash", writer)


if __name__ == "__main__":
    main()
