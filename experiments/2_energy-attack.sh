#!/usr/bin

cd ../unified_interface
METHOD=${1}
K=${2}
CVDS=${3}
DATASET="energy"
OPTION="--dataset ${DATASET} --k ${K} --t_start 0 --t_end 50 --no-eval_last  --t_eval_start 0 --t_eval_end 50 --NUM_TEST_BATCH 1 --targeted"

# ITERS=50
# ITERS=20
eps=0.06

OUT_FILE_PATH="../experiments/tmp_output/${DATASET}_${METHOD}_ITERS.txt"
touch "${OUT_FILE_PATH}"

for trial in 0 1 2
do
    # for eps in 0.04 0.05 0.06 0.07 0.08 #0.02 0.04 0.08 0.15 0.30 0.60 
    for ITERS in 1 2 4 8 16 32
    do
        #STEP_SIZE=0.01 #`echo "print(${eps}/${ITERS}*1.5)"|python3`
        STEP_SIZE=`echo "print(${eps}/${ITERS}*1.5)"|python3`
        fr=`CUDA_VISIBLE_DEVICES=${CVDS} python ./online_attack.py --eps ${eps} --step_size ${STEP_SIZE} --iters ${ITERS} --attack ${METHOD} ${OPTION} --trial ${trial} 2>/dev/null |grep "MSE" -A2  ` # 
        echo "${ITERS},${METHOD},$fr" | tee -a ${OUT_FILE_PATH}
    done
done
echo "" | tee -a ${OUT_FILE_PATH}
