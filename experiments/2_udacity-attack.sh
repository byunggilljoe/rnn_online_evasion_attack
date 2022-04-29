#!/usr/bin
cd ../unified_interface/
METHOD=${1}
K=${2}
CVDS=${3}
DATASET="udacity"
OPTION="--dataset ${DATASET} --k ${K} --t_start 0 --t_end 20 --t_eval_start 0 --t_eval_end 20 --NUM_TEST_BATCH 5 --targeted --batch_size 16"

ITERS=20

OUT_FILE_PATH="../experiments/tmp_output/${DATASET}_${METHOD}.txt"
touch "${OUT_FILE_PATH}"

for trial in 3 4 5 #0 1 2
do 
    for eps in 0.06 0.07 0.08 0.09 0.1 
    do
        STEP_SIZE=`echo "print(${eps}/${ITERS}*1.5)"|python3`
        fr=`CUDA_VISIBLE_DEVICES=${CVDS} python ./online_attack.py --eps ${eps} --step_size ${STEP_SIZE} --iters ${ITERS} --attack ${METHOD} ${OPTION}  --trial ${trial}` #2> /dev/null |grep "MSE" -A2`  #   
        echo "${eps},${METHOD},$fr" | tee -a ${OUT_FILE_PATH} 
    done
done
echo "" | tee -a ${OUT_FILE_PATH} 
