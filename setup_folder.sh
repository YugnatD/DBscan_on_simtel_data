#!/bin/sh
mkdir scratch
mkdir scratch/simtel_data
mkdir scratch/simtel_data/gamma_diffuse/
mkdir scratch/simtel_data/
mkdir scratch/simtel_data/gamma_diffuse/
mkdir scratch/simtel_data/gamma_diffuse/npe
mkdir scratch/ctapipe_data
mkdir scratch/ctapipe_data/gamma_diffuse/
mkdir scratch/ctapipe_data/gamma_diffuse/data
mkdir scratch/simtel_data/gamma_diffuse/data

# check if corsika_run1.simtel.gz exists in scratch/simtel_data/gamma_diffuse/data
if [ ! -f scratch/simtel_data/gamma_diffuse/data/corsika_run1.simtel.gz ]; then
    echo "You must download the corsika_run1.simtel.gz file from the Yggdrasil server"

# check if the gamma_run1.dl1.h5 file exists in scratch/simtel_data/gamma_diffuse/data
if [ ! -f scratch/simtel_data/gamma_diffuse/data/gamma_run1.dl1.h5 ]; then
    cd scratch/simtel_data/gamma_diffuse/data
    echo "You must download the gamma_run1.dl1.h5 file from the Yggdrasil server"

