nohup python3 open_lm/datapreprocess/ray/tokenize_shuffle.py \
 --input ./data2 \
 --content_key text \
 --output test_minipile2 \
 --tokenizer "meta-llama/Llama-3.2-1B-Instruct" \
 --seqlen 2048