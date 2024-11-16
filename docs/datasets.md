Raw:
https://huggingface.co/datasets/mlfoundations/dclm-pool-400m-1x

PT:
https://huggingface.co/datasets/Zyphra/Zyda-2/
https://huggingface.co/datasets/Zyphra/dclm-dedup
https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus

SFT:
https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
https://huggingface.co/datasets/BAAI/Infinity-Instruct

DPO:
HuggingFaceH4/ultrafeedback_binarized
https://huggingface.co/datasets/Intel/orca_dpo_pairs
https://huggingface.co/datasets/argilla/OpenHermesPreferences


```
# huggingface-cli download  HuggingFaceH4/ultrafeedback_binarized  --repo-type dataset

repos=( 'HuggingFaceH4/ultrachat_200k'
        'BAAI/Infinity-Instruct'
        'HuggingFaceH4/ultrafeedback_binarized'
        'Intel/orca_dpo_pairs'
        'argilla/OpenHermesPreferences' )

echo $repos
for i in $repos; do
    echo $i
    huggingface-cli download  "$i"  --repo-type dataset
done
```