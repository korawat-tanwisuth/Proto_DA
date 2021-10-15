seed=2019
bs=96
nav_t=1
gpu_id=0
epoch=70
### A -> D, W
python image_source.py --trte val --output ckps/source/ --da uda --gpu_id 0 --dset office --max_epoch 100 --s 0 --seed $seed  
python image_target.py --cls_par 0.0 --da uda --dset office --gpu_id 0 --s 0 --output_src ckps/source/ --output ckps/target/ --seed $seed --batch_size $bs --max_epoch $epoch --nav_t $nav_t

### D -> A, W
python image_source.py --trte val --output ckps/source/ --da uda --gpu_id 0 --dset office --max_epoch 100 --s 1 --seed $seed 
python image_target.py --cls_par 0.0 --da uda --dset office --gpu_id 0 --s 1 --output_src ckps/source/ --output ckps/target/ --seed $seed --batch_size $bs --max_epoch $epoch --nav_t $nav_t

### W -> A, D
python image_source.py --trte val --output ckps/source/ --da uda --gpu_id 0 --dset office --max_epoch 100 --s 2 --seed $seed
python image_target.py --cls_par 0.0 --da uda --dset office --gpu_id 0 --s 2 --output_src ckps/source/ --output ckps/target/ --seed $seed --batch_size $bs --max_epoch $epoch --nav_t $nav_t




## Office-Home
epoch=100
### A->C,P,R
python image_source.py --trte val --output ckps/source/ --da uda --gpu_id $gpu_id --dset office-home --max_epoch 50 --s 0 --seed $seed 
python image_target.py --cls_par 0.0 --da uda --dset office-home --gpu_id $gpu_id --s 0 --output_src ckps/source_${seed}/ --output ckps/target/ --seed $seed --batch_size $bs --max_epoch $epoch --nav_t $nav_t
### C->A,P,R
python image_source.py --trte val --output ckps/source/ --da uda --gpu_id $gpu_id --dset office-home --max_epoch 50 --s 1 --seed $seed 
python image_target.py --cls_par 0.0 --da uda --dset office-home --gpu_id $gpu_id --s 1 --output_src ckps/source_${seed}/ --output ckps/target/ --seed $seed --batch_size $bs --max_epoch $epoch --nav_t $nav_t

### P->A,P,R
python image_source.py --trte val --output ckps/source/ --da uda --gpu_id $gpu_id --dset office-home --max_epoch 50 --s 2 --seed $seed 
python image_target.py --cls_par 0.0 --da uda --dset office-home --gpu_id $gpu_id --s 2 --output_src ckps/source_${seed}/ --output ckps/target/ --seed $seed --batch_size $bs --max_epoch $epoch --nav_t $nav_t

### R->C,P,A
python image_source.py --trte val --output ckps/source/ --da uda --gpu_id $gpu_id --dset office-home --max_epoch 50 --s 3 --seed $seed 
python image_target.py --cls_par 0.0 --da uda --dset office-home --gpu_id $gpu_id --s 3 --output_src ckps/source_${seed}/ --output ckps/target/ --seed $seed --batch_size $bs --max_epoch $epoch --nav_t $nav_t

