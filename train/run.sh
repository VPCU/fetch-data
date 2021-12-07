<< COMMENT
mkdir train_log/none
echo none 1
python lstm.py \
    --emb-model 'none' \
    >> train_log/none/log_1.txt
echo none 2
python lstm.py \
    --emb-model 'none' \
    >> train_log/none/log_2.txt
echo none 3
python lstm.py \
    --emb-model 'none' \
    >> train_log/none/log_3.txt
COMMENT

mkdir train_log/gmoco
echo gmoco 1
python lstm.py \
    --emb-model 'gmoco' \
    >> train_log/gmoco/log_1.txt
echo gmoco 2
python lstm.py \
    --emb-model 'gmoco' \
    >> train_log/gmoco/log_2.txt
echo gmoco 3
python lstm.py \
    --emb-model 'gmoco' \
    >> train_log/gmoco/log_3.txt
