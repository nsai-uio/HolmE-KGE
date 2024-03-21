# HolmE
HolmE is a general KGE form that is closed under composition.

Download and pre-process the datasets:
```
source datasets/download.sh
python datasets/process.py
```

Analyse the data to get the existing composition patterns.
```
python data_analyse.py
```

To run the code:
```
WN18RR
CUDA_VISIBLE_DEVICES=1 python ../run.py --dataset WN18RR --model HolmE --rank 200 --regularizer N3 --reg 0.0 --optimizer Adam --max_epochs 1000 --patience 100 --valid 5 --batch_size 1000 --neg_sample_size 100 --init_size 0.001 --learning_rate 0.0001 --gamma 0.0 --bias learn --dtype double --double_neg

FB237
CUDA_VISIBLE_DEVICES=1 python learn.py --dataset FB237 --model HolmE --rank 200 --optimizer Adagrad --learning_rate 1e-1 --batch_size 2000 --neg_sample_size 100 --max_epochs 1000 --valid 5 -train 

YAGO3-10
CUDA_VISIBLE_DEVICES=1 python learn.py --dataset YAGO3-10 --model HolmE --rank 200 --optimizer Adagrad --learning_rate 1e-1 --batch_size 1000 --regularizer N3 --reg 5e-3 --max_epochs 200 --valid 5 -train -id 0 -save
```
