module load nvidia
module load conda

conda activate dl_sys
cd classes/deep_learning_systems/llm-sparsification-alokvk2/transformers/examples

python -m torch.distributed.launch --nproc_per_node 4 pytorch/language-modeling/run_clm.py \
    --model_name_or_path ~/classes/deep_learning_systems/llm-sparsification-alokvk2/src/models/roberta-0.5 \
    --tokenizer_name "roberta-large" \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir test-clm-roberta-0.5-train/

python -m torch.distributed.launch --nproc_per_node 4 pytorch/language-modeling/run_clm.py \
    --model_name_or_path ~/classes/deep_learning_systems/llm-sparsification-alokvk2/src/models/roberta-0.9 \
    --tokenizer_name "roberta-large" \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir test-clm-roberta-0.9-train/

python -m torch.distributed.launch --nproc_per_node 4 pytorch/language-modeling/run_clm.py \
    --model_name_or_path ~/classes/deep_learning_systems/llm-sparsification-alokvk2/src/models/roberta-0.95 \
    --tokenizer_name "roberta-large" \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir test-clm-roberta-0.95-train/

python -m torch.distributed.launch --nproc_per_node 4 pytorch/language-modeling/run_clm.py \
    --model_name_or_path ~/classes/deep_learning_systems/llm-sparsification-alokvk2/src/models/roberta-0.99 \
    --tokenizer_name "roberta-large" \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir test-clm-roberta-0.99-train/

python -m torch.distributed.launch --nproc_per_node 4 pytorch/language-modeling/run_clm.py \
    --model_name_or_path ~/classes/deep_learning_systems/llm-sparsification-alokvk2/src/models/pegasus-0.9 \
    --tokenizer_name "google/pegasus-large" \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir test-clm-pegasus-.9-train/

python -m torch.distributed.launch --nproc_per_node 4 pytorch/language-modeling/run_clm.py \
    --model_name_or_path ~/classes/deep_learning_systems/llm-sparsification-alokvk2/src/models/pegasus-0.95 \
    --tokenizer_name "google/pegasus-large" \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir test-clm-pegasus-.95-train/
    
python -m torch.distributed.launch --nproc_per_node 4 pytorch/language-modeling/run_clm.py \
    --model_name_or_path ~/classes/deep_learning_systems/llm-sparsification-alokvk2/src/models/pegasus-0.99 \
    --tokenizer_name "google/pegasus-large" \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir test-clm-pegasus-.99-train/

# python -m torch.distributed.launch --nproc_per_node 4 pytorch/text-classification/run_glue.py \
#   --model_name_or_path ~/classes/deep_learning_systems/llm-sparsification-alokvk2/src/models/gpt2-0.99 \
#   --tokenizer_name ~/classes/deep_learning_systems/llm-sparsification-alokvk2/src/models/gpt_tokenizer \
#   --task_name mrpc \
#   --do_train \
#   --do_eval \
#   --max_seq_length 128 \
#   --per_device_train_batch_size 32 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 3 \
#   --output_dir test-glue-gpt2-0.99/
