language_list=('all')
test_list=('de' 'es' 'fr' 'ru' 'ko' 'ja' 'zh' 'ar')
export CUDA_VISIBLE_DEVICES='7'
for lg in ${language_list[*]}; do
for test_lg in ${test_list[*]}; do
    python3 -m torch.distributed.launch --nproc_per_node=1 --master_port 29501 \
    predict.py \
    --lg $lg \
    --test_lg $test_lg \
    --simclr 0 \
    --dataset_path /home/zhq/Moco/our_dataset/ \
    --quene_length 0 \
    --load_model_path /home/zhq/Moco/result/4-all-32-true-0-0.06-42-100-0.999-0-dev_qq \
    --test_dev 0 \
    --unsupervised 0 \
    --wolinear 0 \
    > log/test-${lg}-${test_lg}-32.log 2>&1
done
done
