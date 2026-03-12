dataset=mosi
model=emt-dlfr
# run_once
python run_once.py --datasetName $dataset --modelName $model \
   --model_save_dir results/$dataset/$model/run_once/models \
   --res_save_dir results/$dataset/$model/run_once/results \
   --save_model \
   --exp_name moe-uncertainty \
   --gpu_ids 5
   
echo "done!"

