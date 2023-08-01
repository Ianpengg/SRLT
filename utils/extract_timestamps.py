import os
import argparse
import numpy as np


def extract_timestamp(root_path, data_folder):
    with open(os.path.join(root_path, "radar.timestamps"), "w") as f:
        for path in sorted(os.listdir(data_folder)):
            if path.endswith(".png") or path.endswith(".jpeg") or path.endswith(".jpg"):
                timestamp = path.split(".")[0]
                f.write(timestamp + "\n")
    print("timestamp file is saved in {}".format(root_path))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--root_path", type=str, help="specify the path where the root path is"
    )
    argparser.add_argument(
        "--data_folder", type=str, help="specify the path where the radar image locate"
    )

    args = argparser.parse_args()

    extract_timestamp(args.root_path, args.data_folder)
