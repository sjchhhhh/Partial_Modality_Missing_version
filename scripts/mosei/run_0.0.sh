#!/bin/bash
dataset=mosei
model=emt-dlfr
mr=0.0
cd "$(dirname "$0")/../.." || exit 1
python run_once.py --datasetName $dataset --modelName $model --missing_rates $mr \
   --model_save_dir results/$dataset/$model/run_once/models \
   --res_save_dir results/$dataset/$model/run_once/results \
   --KeyEval 'Loss(pred_m)' \
   --save_model \
   --gpu_ids 0 1 3
echo "done!"
