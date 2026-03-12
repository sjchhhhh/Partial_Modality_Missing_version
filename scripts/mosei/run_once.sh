dataset=mosei
model=emt-dlfr
NPROC=${1:-8}

# 多卡并行（DDP），模仿 EMT-DLFR-MGPU-main
torchrun --nproc_per_node=$NPROC --master_port=29510 run_once.py \
   --datasetName $dataset --modelName $model \
   --model_save_dir results/$dataset/$model/run_once/models \
   --res_save_dir results/$dataset/$model/run_once/results \
   --KeyEval 'Loss(pred_m)'

echo "done!"

