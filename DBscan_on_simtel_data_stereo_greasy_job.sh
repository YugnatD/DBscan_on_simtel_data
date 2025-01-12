#!/bin/bash -l

function printHelp {
    echo " --> ERROR in input arguments "
    echo " [0] -d         : default"
    echo " [1]            : particle type (g,gd,e,p)"
    echo " [2]            : Nnodes (1-...)"
    echo " [0] -c         : clean"
    echo " [0] -h         : print help"
}

nCPU_per_node=72
nCPU_idle=1
nJOB_per_node=$(echo "$nCPU_per_node - $nCPU_idle" | bc -l)
greasyJobDir="./greasy_job/"
outGreasySbatch_sh="./run_greasy_sbatch.sh"
siffile="/scratch/snx3000/lburmist/singularity/21.10.2024/ctapipe.sif"

if [ $# -eq 0 ]; then
    printHelp
else
    if [ "$1" = "-d" ]; then
        if [ $# -eq 3 ]; then
            if [ "$2" = "g" ]; then
		particletype="gamma"
	    elif [ "$2" = "gd" ]; then
		particletype="gamma_diffuse"
            elif [ "$2" = "e" ]; then
		particletype="electron"
            elif [ "$2" = "p" ]; then
		particletype="proton"
            fi
	    #
	    Nnodes=$3
	    echo "Nnodes        $Nnodes"
	    echo "nCPU_per_node $nCPU_per_node"
	    echo "nCPU_idle     $nCPU_idle"
	    echo "nJOB_per_node $nJOB_per_node"
	    #
	    fileID=1
	    #
	    rm -rf $outGreasySbatch_sh
	    #
	    for nodesID in $(seq 1 $Nnodes)
	    do
		#
		echo "nodesID = $nodesID"
		echo "#!/bin/sh" >> $outGreasySbatch_sh
		#
		outJOBfile="$greasyJobDir/node_$nodesID.job"
		outJOBfileList="$greasyJobDir/node_$nodesID.joblist"
		#
		echo "sbatch $outJOBfile" >> $outGreasySbatch_sh
		#
		rm -rf $outJOBfile
		rm -rf $outJOBfileList
		echo "#!/bin/bash -l" >> $outJOBfile
		echo "#SBATCH --job-name=simtel" >> $outJOBfile
		echo "#SBATCH --output=/scratch/snx3000/lburmist/simtel_data/job_outlog/simtel.%j.out" >> $outJOBfile
		echo "#SBATCH --error=/scratch/snx3000/lburmist/simtel_data/job_error/simtel.%j.err" >> $outJOBfile
		echo "#SBATCH --account=cta03" >> $outJOBfile
		echo "#SBATCH --time=02:00:00" >> $outJOBfile
		echo "#SBATCH --nodes=1" >> $outJOBfile
		echo "#SBATCH --cpus-per-task=1" >> $outJOBfile
		echo "#SBATCH --partition=normal" >> $outJOBfile
		echo "#SBATCH --constraint=mc" >> $outJOBfile
		echo " " >> $outJOBfile
		echo "module load daint-mc" >> $outJOBfile
		echo "module load GREASY" >> $outJOBfile
		echo " " >> $outJOBfile
		echo "greasy $outJOBfileList" >> $outJOBfile
		#
		for jobIT in $(seq 1 $nJOB_per_node)
		do
		    #
		    inFilePref="/scratch/snx3000/lburmist/simtel_data/$particletype/data/"
		    outFilePref="/scratch/snx3000/lburmist/simtel_data/$particletype/npe/"
		    dataCtapipeOIdirPreff="/scratch/snx3000/lburmist/ctapipe_data/$particletype/data/"
		    mkdir -p $outFilePref
		    echo "  fileID      = $fileID"
		    echo "  inFilePref  = $inFilePref"
		    echo "  outFilePref = $outFilePref"		    
		    #
		    #/scratch/snx3000/lburmist/simtel_data/$particletype/data/corsika_run1.simtel.gz
		    in_simtel_file="$inFilePref/corsika_run$fileID.simtel.gz"
		    dl1In=$dataCtapipeOIdirPreff/$particletype"_run"$fileID".dl1.h5"
		    #
		    if [ -f "$in_simtel_file" ]; then
			if [ -f "$dl1In" ]; then
			    out_pkl_file="$outFilePref/corsika_run$fileID.npe.pkl"
			    out_csv_file="$outFilePref/corsika_run$fileID.npe.csv"
			    out_h5_file="$outFilePref/corsika_run$fileID.npe.h5"
			    #
			    rm -rf $out_pkl_file
			    rm -rf $out_csv_file
			    rm -rf $out_h5_file
			    #
			    pixel_mapping_csv="pixel_mapping.csv"
			    isolated_flower_seed_super_flower_csv="isolated_flower_seed_super_flower.list"
			    isolated_flower_seed_flower_csv="isolated_flower_seed_flower.list"
			    all_seed_flower_csv="all_seed_flower.list"
			    #
			    cmd="singularity run -B /scratch/snx3000/lburmist/:/scratch/snx3000/lburmist/ $siffile python3 DBscan_on_simtel_data_stereo.py --trg $in_simtel_file $dl1In $out_pkl_file $out_csv_file $out_h5_file $pixel_mapping_csv $isolated_flower_seed_super_flower_csv $isolated_flower_seed_flower_csv $all_seed_flower_csv"
			    echo "$cmd" >> $outJOBfileList
			fi
		    fi
		    #
		    ((fileID=fileID+1))
		done
	    done
	else
	    printHelp   
	fi      
    elif [ "$1" = "-c" ]; then
	rm -rf $greasyJobDir/*
	rm -rf $outGreasySbatch_sh
    elif [ "$1" = "-h" ]; then
        printHelp
    else
        printHelp
    fi
fi
