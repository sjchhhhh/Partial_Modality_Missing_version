# 统一缺失率训练：单次训练中每个样本随机缺失率，一次得到可应对多种缺失率的模型
dataset=mosi
model=emt-dlfr
python run_once.py --datasetName $dataset --modelName $model \
   --model_save_dir results/$dataset/$model/run_once/models \
   --res_save_dir results/$dataset/$model/run_once/results \
   --save_model \
   --unified_missing \
   --exp_name unified-missing \
   --gpu_ids 2

echo "done!"
