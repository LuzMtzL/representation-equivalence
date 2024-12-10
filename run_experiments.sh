# Generate continuous periodic 30-bandlimited functions
# python data_generation.py --data scripts/data.py \
#     --max_sin 100 --max_freq 30 --max_amp 1 \
#     --num_train 1000 --num_test 300 \
#     --seed 24

# Train with K=30 frame sequences

# CNP
# python run.py --data scripts/data.py \
#     --K 30 --model CNP_MLP_Mean  --from_begin \
#     --sampling uniform --min_samples 5 --num_target 50 --positional_encoding det --r_dim 32 \
#     --input_dim 1 --hidden_size 64 \
#     --epochs 25 --batch_size 50 \
#     --device cuda --optimizer adam \
#     --learning_rate 0.0001 --weight_decay 1e-8 --momentum 0.9 --drop_rate 0.5 \
#     --seed 24 > ./results/CNP_train_out.txt

# BiLSTM
# python run.py --data scripts/data.py \
#     --K 30 --model BiLSTM  --from_begin \
#     --input_dim 1 --hidden_size 64 \
#     --epochs 25 --batch_size 50 \
#     --device cuda --optimizer adam \
#     --learning_rate 0.0001 --weight_decay 1e-8 --momentum 0.9 --drop_rate 0.5 \
#     --seed 24 > ./results/BiLSTM_train_out.txt

# Test with various K values
# for K in {5..100..5}
# do
#     # CNP
#     python run.py --data scripts/data.py \
#         --K $K --model CNP_MLP_Mean --save_pred \
#         --sampling uniform --min_samples 5 --positional_encoding det --r_dim 32 \
#         --input_dim 1 --hidden_size 64 \
#         --epochs 25 --batch_size 50 \
#         --model_name CNP_MLP_Mean_30_uniform_5 \
#         --seed 24 > ./results/CNP_test_${K}_out.txt
    
#     # BiLSTM
#     python run.py --data scripts/data.py \
#         --K $K --model BiLSTM --save_pred \
#         --input_dim 1 --hidden_size 64 \
#         --epochs 25 --batch_size 50 \
#         --model_name BiLSTM_30_uniform_5 \
#         --seed 24 > ./results/BiLSTM_test_${K}_out.txt
# done

# Get data for plot
# python plot_results.py --pred_fldr predictions/CNP_MLP_Mean_30_uniform_5 \
#     --train_K 30 --start_K 5 --last_K 100 --step_K 5
# python plot_results.py --pred_fldr predictions/BiLSTM_30_uniform_5 \
#     --train_K 30 --start_K 5 --last_K 100 --step_K 5

# make plot
python make_plot.py

