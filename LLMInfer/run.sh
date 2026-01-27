CUDA_VISIBLE_DEVICES=0,1,2,3 \
python Scripts/LLMInfer/infer.py \
    --input_file "Scripts/LLMInfer/test.jsonl" \
    --output_file "Scripts/LLMInfer/test_output.jsonl" \
    --model_dir "xxx" \
    --num_dp 4 \
    --batch_size 2 \
    --max_input_length 1024 \
    --max_output_length 1024 \
    --max_lora_rank 32 \
    --gpu_memory_utilization 0.9 \
    --config_dir "Scripts/LLMInfer/test.jsonl" \
    --model_dtype "float16"