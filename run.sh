#!/bin/bash

# Usage: ./run.sh <# GPUs> <output-directory> <problem name> <epochs> <batch size> <ppn>
# ./run.sh "128 64 32 16 8 4 2 1" weak-512-30-1S large 30 512 1
# COMMS=mvapich ./run.sh 2 mva ts 100 512 1
# COMMS=openmpi ./run.sh 32 ompi ts 100 512 1

nodes=$1
outdir=out/$2
prob=$3
iter=$4
mbsize=$5
ppn=$6
usegpu="${USEGPU:-1}"
scale="${SCALE:-weak}"
comms="${COMMS:-mvapich}"

bin="python dlrm_s_pytorch.py"

for n in $nodes; do

    # Set args for problem size
    if [ "$prob" == "small" ]
    then
	# Set mini-batch for weak and strong scaling
	if [ "$scale" == "weak" ]
	then
	    gmbsize=$((n*mbsize))
	elif [ "$scale" == "strong" ]
	then
	    gmbsize=$mbsize
	else
	    echo "SCALE=weak|strong - SCALE=$scale not supported" 
	    exit
	fi
	gargs="--data-generation=random --numpy-rand-seed=727 --mini-batch-size=$gmbsize --num-batches=$iter --print-freq=10 --print-time --enable-profiling"
	args="--arch-mlp-bot=512-512-64 --arch-mlp-top=1024-1024-1024-1 --arch-embedding-size=1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000 --arch-sparse-feature-size=64 --num-indices-per-lookup=100"
    elif [ "$prob" == "ts" ]
    then
	per_gpu_batch_size=$mbsize
	ngpu=$((n * ppn))
	nemb=128
	batchsize=$((ngpu * per_gpu_batch_size))
	pnemb=$((nemb / n))
	embsize="10000"
	embsizes=$embsize
	sparse_feature_size=256
	first_top_mlp_size=$((nemb * sparse_feature_size + sparse_feature_size))
	for ((a=1 ; a<$nemb ; a++)) 
	do
	    embsizes+="-"
	    embsizes+=$embsize
	done
	# echo $embsizes
	gargs="--data-generation=random --numpy-rand-seed=727 --mini-batch-size=${batchsize} --num-batches=$iter --print-freq=10 --print-time --enable-profiling"
	args="--arch-mlp-bot 64-1024-1024-1024-${sparse_feature_size} --arch-mlp-top ${first_top_mlp_size}-1024-1024-1024-1024-1 --arch-embedding-size ${embsizes} --arch-sparse-feature-size=${sparse_feature_size} --num-indices-per-lookup=100"
    else
	echo "Problem size not specified"
	exit
    fi

    outdir=out/$2-$n-$prob-$gmbsize
    mkdir -p $outdir

    if [ "$usegpu" == "1" ]
    then
	args="$args --use-gpu"
    fi

    if [ "$comms" == "openmpi" ]
    then
	if [ "$usegpu" == "0" ]
	then
	    env1="--bind-to socket --report-bindings"
	    env2="--mca orte_base_help_aggregate 0 --mca btl_openib_allow_ib 1 --mca btl_openib_warn_default_gid_prefix 0 --mca mpi_show_mca_params all"

	    export OMP_NUM_THREADS=20
	    export OMP_PLACES=cores
	    export OMP_PROC_BIND=close
	    export OMP_DISPLAY_ENV=true
	    export PYTORCH_SYNC_ALLREDUCE=1
	else
	    env1="--bind-to none --mca orte_base_help_aggegate 0"
	    # env2="--mca pml ucx --mca btl ^openib,tcp"
	    env2="--mca btl openib,self --mca btl_openib_allow_ib 1 --mca btl_openib_warn_default_gid_prefix 0 --mca pml ^ucx --mca btl_openib_want_cuda_gdr 1"
	fi
	echo mpirun -n $n -N $ppn $env1 $env2 -x EXE="$bin" -x ARGS="$args $gargs" \
	     ./command.sh $outdir/out.txt
	echo mpirun -n $n -N $ppn $env1 $env2 -x EXE="$bin" -x ARGS="$args $gargs" \
	     ./command.sh > $outdir/out.txt
	mpirun -n $n -N $ppn $env1 $env2 -x EXE="$bin" -x ARGS="$args $gargs" \
	       ./command.sh 2>&1 >> $outdir/out.txt
    elif [ "$comms" == "mvapich" ]
    then
	env1="-export-all --hostfile hfile"
	env2="MV2_USE_CUDA=1 MV2_USE_GDRCOPY=0 MV2_DEBUG_SHOW_BACKTRACE=1 MV2_ENABLE_AFFINITY=0 MV2_SUPPORT_TENSOR_FLOW=1 MV2_USE_RDMA_CM=0"
	export EXE="$bin"
	export ARGS="$args $gargs"
        export MV2_USE_CUDA=1
        export MV2_USE_GDRCOPY=0
        export MV2_ENABLE_AFFINITY=0
        export MV2_USE_RDMA_CM=0
	export MV2_DEBUG_SHOW_BACKTRACE=1
	export MV2_SUPPORT_TENSOR_FLOW=1
	mpirun -np $n -ppn $ppn hostname > ./hfile	
	echo CUDA_VISIBLE_DEVICES=0 mpirun_rsh -np $n $env1 $env2 EXE="$bin" ARGS="$args $gargs" \
	     ./command.sh 2>&1 | tee $outdir/out.txt
	echo CUDA_VISIBLE_DEVICES=0 mpirun_rsh -np $n $env1 $env2 $bin $args $gargs
	
	# CUDA_VISIBLE_DEVICES=0 mpirun_rsh -np $n $env1 $env2 $bin $args $gargs 2>&1 | tee $outdir/out.txt    
	# EXE="$bin" ARGS="$args $gargs"
    elif [ "$comms" == "gloo" ]
    then
	export MASTER_ADDR='100.96.167.138'
	env1="--bind-to none --mca orte_base_help_aggegate 0"
	env2="--mca btl self,openib --mca btl_openib_allow_ib 1 --mca btl_openib_warn_default_gid_prefix 0 --mca pml ^ucx --mca btl_openib_want_cuda_gdr 1"
    else
	echo "COMMS=$comms not supported"
	exit
    fi
    mv dlrm_s_pytorch_r* $outdir/.
done
