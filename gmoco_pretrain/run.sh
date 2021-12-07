export PYTHONPATH=$PYTHONPATH:/home/haorui/moco-test
python /home/haorui/moco-test/graphsaint/mocodict/pretrain.py \
    --data_prefix /home/haorui/Downloads/fetch-data/data/2mnews \
    --pretrain_config /home/haorui/Downloads/fetch-data/kd-news.yml