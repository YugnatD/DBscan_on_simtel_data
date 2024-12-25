import os
import subprocess
import numpy as np

# Define directories and file paths
dataOIdirPreff = "scratch/simtel_data/gamma_diffuse/"
dataCtapipeOIdirPreff = "scratch/ctapipe_data/gamma_diffuse/"

# Define input files
simtelIn = os.path.join(dataOIdirPreff, "data/corsika_run1.simtel.gz")
dl1In = os.path.join(dataCtapipeOIdirPreff, "data/gamma_diffuse_run1.dl1.h5")

# Additional files
pixel_mapping_csv = "pixel_mapping.csv"
isolated_flower_seed_super_flower_csv = "isolated_flower_seed_super_flower.list"
isolated_flower_seed_flower_csv = "isolated_flower_seed_flower.list"
all_seed_flower_csv = "all_seed_flower.list"

# Define parameter ranges to test
eps_xy_values = range(1, 5, 1) # 1, 2, 3
eps_t_values = range(1, 5, 1) # 1, 2
threshold_values = range(3, 15, 1) # 3, 5, 7, 9

# Base output directory
base_output_dir = "scratch/simtel_data/gamma_diffuse/"

# Iterate over parameter combinations
for eps_xy in eps_xy_values:
    for eps_t in eps_t_values:
        for threshold in threshold_values:
            # Create a unique output folder for this combination
            folder_name = f"XY{eps_xy}_T{eps_t}_THRES{threshold}"
            output_dir = os.path.join(base_output_dir, folder_name)
            # check if the folder exists
            if os.path.exists(output_dir):
                # the folder exist, check if corsika_run1.npe.csv exist
                if os.path.exists(os.path.join(output_dir, "corsika_run1.npe.csv")):
                    # the file exist, skip this iteration
                    continue
            os.makedirs(output_dir, exist_ok=True)

            # Define output files
            outpkl = os.path.join(output_dir, "corsika_run1.npe.pkl")
            outcsv = os.path.join(output_dir, "corsika_run1.npe.csv")
            outh5 = os.path.join(output_dir, "corsika_run1.npe.h5")

            # Command to run the Python script
            command = [
                "python3", "DBscan_on_simtel_data_stereo.py",
                "--trg", simtelIn, dl1In, outpkl, outcsv, outh5,
                pixel_mapping_csv, isolated_flower_seed_super_flower_csv,
                isolated_flower_seed_flower_csv, all_seed_flower_csv,
                str(eps_t), str(eps_xy), str(threshold)
            ]

            # Run the command
            try:
                print(f"Running: EPS_XY={eps_xy}, EPS_T={eps_t}, THRESHOLD={threshold}")
                subprocess.run(command, check=True)
                print(f"Completed: {folder_name}")
            except subprocess.CalledProcessError as e:
                print(f"Error while executing the script for {folder_name}: {e}")
