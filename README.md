# unsup3D_reimplementation
### Train Ex : python main.py --type train --name test --gpu_ids 0 --batch_size 64 --dataset CelebA
### dataset in [CelebA, Synface, Cat]

### evaluate Ex : python main.py --type evaluate --gpu_ids 0 --batch_size 64 --dataset Synface --load_dir ./checkpoint/Synface/model_final

### visualize EX : python main.py --type visualize --gpu_ids 0 --batch_size 64 --dataset CelebA --load_dir ./checkpoint/CelebA/model_final
