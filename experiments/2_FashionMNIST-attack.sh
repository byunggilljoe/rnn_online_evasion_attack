#!/usr/bin

cd ../unified_interface
METHOD=${1}
K=${2}
CVDS=${3}
DATASET="fashion_mnist"
OPTION="--dataset ${DATASET} --k ${K} --t_start 0 --t_end 28 --no-eval_last  --t_eval_start 0 --t_eval_end 28 --no-eval_last --NUM_TEST_BATCH 5 --targeted"

# ITERS=50
ITERS=20

OUT_FILE_PATH="../experiments/tmp_output/${DATASET}_${METHOD}.txt"
touch "${OUT_FILE_PATH}"

for trial in 0 1 2
do
    for eps in 0.3 0.4 0.5 0.6 #0.02 0.04 0.08 0.15 0.30 0.60
    do
        STEP_SIZE=`echo "print(${eps}/${ITERS}*1.5)"|python3`
        fr=`CUDA_VISIBLE_DEVICES=${CVDS} python ./online_attack.py --eps ${eps} --step_size ${STEP_SIZE} --iters ${ITERS} --attack ${METHOD} ${OPTION} --trial ${trial} 2>/dev/null |grep "Fool rate" -A2 ` #  
        echo "${eps},${METHOD},$fr" | tee -a ${OUT_FILE_PATH}
    done
done
echo "" | tee -a ${OUT_FILE_PATH}
