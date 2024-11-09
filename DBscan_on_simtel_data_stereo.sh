#!/bin/bash

function printHelp {
    echo " --> ERROR in input arguments "
    echo " [0] -d             : trigger"
    echo " [0] --d_mono       : trigger mono"
    echo " [0] -n             : noise"
    echo " [0] --n_mono       : noise for mono (LST1)"
    echo " [0] -m             : getmap"
    echo " [0] --astropytable      : astropytable"
    echo " [0] --astropytable_read : astropytable_read"
    echo " [0] -h                  : print help"
}

if [ $# -eq 0 ] 
then    
    printHelp
else
    if [ "$1" = "-d" ]; then
 	#
	dataOIdirPreff="scratch/simtel_data/gamma_st/"
	dataOI_npe_dirPreff="scratch/simtel_data/gamma_st/npe/"
	#
	#dataOIdirPreff="scratch/simtel_data/proton_st/"
	#dataOI_npe_dirPreff="scratch/simtel_data/proton_st/npe/"
	#
	mkdir -p $dataOI_npe_dirPreff
	simtelIn=$dataOIdirPreff"/data/corsika_run1.simtel.gz"
	outpkl=$dataOI_npe_dirPreff"/corsika_run1.npe.pkl"
	outcsv=$dataOI_npe_dirPreff"/corsika_run1.npe.csv"
	#
	pixel_mapping_csv="pixel_mapping.csv"
        isolated_flower_seed_super_flower_csv="isolated_flower_seed_super_flower.list"
        isolated_flower_seed_flower_csv="isolated_flower_seed_flower.list"
	all_seed_flower_csv="all_seed_flower.list"
	python DBscan_on_simtel_data_stereo.py --trg $simtelIn $outpkl $outcsv $pixel_mapping_csv $isolated_flower_seed_super_flower_csv $isolated_flower_seed_flower_csv $all_seed_flower_csv
    elif [ "$1" = "-n" ]; then
	#
	#dataOIdirPreff="scratch/simtel_data/proton_st/"
	#dataOI_npe_dirPreff="scratch/simtel_data/proton_st/npe/"
	dataOIdirPreff="scratch/simtel_data/gamma_st/"
	dataOI_npe_dirPreff="scratch/simtel_data/gamma_st/npe/"
	#
	mkdir -p $dataOI_npe_dirPreff
	simtelIn=$dataOIdirPreff"/data/corsika_run1.simtel.gz"
	outpkl=$dataOI_npe_dirPreff"/corsika_run1.npe.pkl"
	outcsv=$dataOI_npe_dirPreff"/corsika_run1.npe.csv"
	#
	pixel_mapping_csv="pixel_mapping.csv"
        isolated_flower_seed_super_flower_csv="isolated_flower_seed_super_flower.list"
        isolated_flower_seed_flower_csv="isolated_flower_seed_flower.list"
	all_seed_flower_csv="all_seed_flower.list"
	python DBscan_on_simtel_data_stereo.py --noise $simtelIn $outpkl $outcsv $pixel_mapping_csv $isolated_flower_seed_super_flower_csv $isolated_flower_seed_flower_csv $all_seed_flower_csv
    elif [ "$1" = "--d_mono" ]; then
	#
	dataOIdirPreff="scratch/simtel_data/gamma_diffuse/"
	dataOI_npe_dirPreff="scratch/simtel_data/gamma_diffuse/npe/"
	dataCtapipeOIdirPreff="scratch/ctapipe_data/gamma_diffuse/"
	#
	mkdir -p $dataOI_npe_dirPreff
	simtelIn=$dataOIdirPreff"/data/corsika_run1.simtel.gz"
	dl1In=$dataCtapipeOIdirPreff"/data/gamma_diffuse_run1.dl1.h5"
	outpkl=$dataOI_npe_dirPreff"/corsika_run1.npe.pkl"
	outcsv=$dataOI_npe_dirPreff"/corsika_run1.npe.csv"
	outh5=$dataOI_npe_dirPreff"/corsika_run1.npe.h5"
	#
	pixel_mapping_csv="pixel_mapping.csv"
        isolated_flower_seed_super_flower_csv="isolated_flower_seed_super_flower.list"
        isolated_flower_seed_flower_csv="isolated_flower_seed_flower.list"
	all_seed_flower_csv="all_seed_flower.list"
	python3 DBscan_on_simtel_data_stereo.py --trg $simtelIn $dl1In $outpkl $outcsv $outh5 $pixel_mapping_csv $isolated_flower_seed_super_flower_csv $isolated_flower_seed_flower_csv $all_seed_flower_csv	
    elif [ "$1" = "--n_mono" ]; then
	#
	dataOIdirPreff="scratch/simtel_data/gamma_diffuse/"
	dataOI_npe_dirPreff="scratch/simtel_data/gamma_diffuse/npe/"
	#
	mkdir -p $dataOI_npe_dirPreff
	simtelIn=$dataOIdirPreff"/data/corsika_run1.simtel.gz"
	outpkl=$dataOI_npe_dirPreff"/corsika_run1.npe.pkl"
	outcsv=$dataOI_npe_dirPreff"/corsika_run1.npe.csv"
	#
	pixel_mapping_csv="pixel_mapping.csv"
        isolated_flower_seed_super_flower_csv="isolated_flower_seed_super_flower.list"
        isolated_flower_seed_flower_csv="isolated_flower_seed_flower.list"
	all_seed_flower_csv="all_seed_flower.list"
	python3 DBscan_on_simtel_data_stereo.py --noise $simtelIn $outpkl $outcsv $pixel_mapping_csv $isolated_flower_seed_super_flower_csv $isolated_flower_seed_flower_csv $all_seed_flower_csv
    elif [ "$1" = "-m" ]; then
	simtelIn="scratch/simtel_data/old/proton_st_NSB268MHz/data/corsika_run1.simtel.gz"
	python DBscan_on_simtel_data_stereo.py --getmap $simtelIn
    elif [ "$1" = "--astropytable" ]; then
	simtelIn="scratch/simtel_data/proton_st/data/corsika_run1.simtel.gz"
	python3 DBscan_on_simtel_data_stereo.py --astropytable $simtelIn
    elif [ "$1" = "--astropytable_read" ]; then
	tableIn="testtable.h5"
	python3 DBscan_on_simtel_data_stereo.py --astropytable_read $tableIn
    elif [ "$1" = "-h" ]; then
        printHelp
    else
        printHelp
    fi
fi

#espeak "I have done"
