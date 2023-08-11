# PART I DLinear

# 注意，seq_len均从Transformer类的96调成了这里的336
# 具体原因论文中有讲；简单来说就是：对LSTF-Linear来说，seq_len越大越好
1.1 ETTh1 & pred_len=96 & seq_len=336
Dlinear: 
# 学习率这里要调成0.005？大了50倍？
python -u run_longExp.py  --is_training 1  --root_path ./dataset/  --data_path ETTh1.csv --model_id ETTh1_336_96  --model DLinear  --data ETTh1  --features M  --seq_len 336  --pred_len 96  --enc_in 7  --des 'Exp'  --itr 1  --batch_size 32  --learning_rate 0.005  --gpu 1 --test_train_num 10
python -u run_longExp.py  --is_training 1  --root_path ./dataset/  --data_path ETTh1.csv --model_id ETTh1_336_96  --model DLinear  --data ETTh1  --features M  --seq_len 336  --pred_len 96  --enc_in 7  --des 'Exp'  --itr 1  --batch_size 32  --learning_rate 0.005  --gpu 1 --test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 2
NLinear:
python -u run_longExp.py  --is_training 1  --root_path ./dataset/  --data_path ETTh1.csv --model_id ETTh1_336_96  --model NLinear  --data ETTh1  --features M  --seq_len 336  --pred_len 96  --enc_in 7  --des 'Exp'  --itr 1  --batch_size 32  --learning_rate 0.005  --gpu 1 --test_train_num 10
Linear:
python -u run_longExp.py  --is_training 1  --root_path ./dataset/  --data_path ETTh1.csv --model_id ETTh1_336_96  --model Linear  --data ETTh1  --features M  --seq_len 336  --pred_len 96  --enc_in 7  --des 'Exp'  --itr 1  --batch_size 32  --learning_rate 0.005  --gpu 1 --test_train_num 10

1.2 ETTh1 & pred_len=192
Dlinear: 
python -u run_longExp.py  --is_training 1  --root_path ./dataset/  --data_path ETTh1.csv --model_id ETTh1_336_192  --model DLinear  --data ETTh1  --features M  --seq_len 336  --pred_len 192  --enc_in 7  --des 'Exp'  --itr 1  --batch_size 32  --learning_rate 0.005  --gpu 1 --test_train_num 10


1.3 ETTh1 & pred_len=24
Dlinear: 
python -u run_longExp.py  --is_training 1  --root_path ./dataset/  --data_path ETTh1.csv --model_id ETTh1_336_24  --model DLinear  --data ETTh1  --features M  --seq_len 336  --pred_len 24  --enc_in 7  --des 'Exp'  --itr 1  --batch_size 32  --learning_rate 0.005  --gpu 1 --test_train_num 10


1.4 ETTh1 & pred_len=720
Dlinear: 
python -u run_longExp.py  --is_training 1  --root_path ./dataset/  --data_path ETTh1.csv --model_id ETTh1_336_720  --model DLinear  --data ETTh1  --features M  --seq_len 336  --pred_len 720  --enc_in 7  --des 'Exp'  --itr 1  --batch_size 32  --learning_rate 0.005  --gpu 1 --test_train_num 10
NLinear: 
python -u run_longExp.py  --is_training 1  --root_path ./dataset/  --data_path ETTh1.csv --model_id ETTh1_336_720  --model NLinear  --data ETTh1  --features M  --seq_len 336  --pred_len 720  --enc_in 7  --des 'Exp'  --itr 1  --batch_size 32  --learning_rate 0.005  --gpu 1 --test_train_num 10


2.1 ETTm1 & pred_len=96 & seq_len=336
DLinear:
python -u run_longExp.py  --is_training 1  --root_path ./dataset/  --data_path ETTm1.csv --model_id ETTm1_336_96  --model DLinear  --data ETTm1  --features M  --seq_len 336  --pred_len 96  --enc_in 7  --des 'Exp'  --itr 1  --batch_size 8  --learning_rate 0.0001  --gpu 1 --test_train_num 10
NLinear:
python -u run_longExp.py  --is_training 1  --root_path ./dataset/  --data_path ETTm1.csv --model_id ETTm1_336_96  --model NLinear  --data ETTm1  --features M  --seq_len 336  --pred_len 96  --enc_in 7  --des 'Exp'  --itr 1  --batch_size 8  --learning_rate 0.0001  --gpu 1 --test_train_num 10


5.1 Electricity & pred_len=96
DLinear:
python -u run_longExp.py  --is_training 1  --root_path ./dataset/  --data_path electricity.csv --model_id Electricity_336_96  --model DLinear  --data custom  --features M  --seq_len 336  --pred_len 96  --enc_in 321  --des 'Exp'  --itr 1  --batch_size 16  --learning_rate 0.001  --gpu 1 --test_train_num 10
NLinear:
python -u run_longExp.py  --is_training 1  --root_path ./dataset/  --data_path electricity.csv --model_id Electricity_336_96  --model NLinear  --data custom  --features M  --seq_len 336  --pred_len 96  --enc_in 321  --des 'Exp'  --itr 1  --batch_size 16  --learning_rate 0.001  --gpu 1 --test_train_num 10


6.1 Traffic & pred_len=96
Linear:
python -u run_longExp.py --is_training 1 --root_path ./dataset/ --data_path traffic.csv --model_id traffic_336_96 --model Linear --data custom --features M --seq_len 336 --pred_len 96 --enc_in 862 --des 'Exp' --itr 1 --batch_size 16 --learning_rate 0.05  --gpu 1 --test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 2
DLinear: