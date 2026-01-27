import argparse
from tqdm import tqdm,trange
from fileio import *
from transformers import AutoTokenizer
import os
import ray
from vllm import LLM,SamplingParams
from vllm.lora.request import LoRARequest
import re

ray.init()

CONFIG={
    "temperature":1.0,
    "top_p":0.7,
    "n":1,
    "max_tokens":2048
}

@ray.remote(num_gpus=1)
def llminfer(args,chunks):
    llm=LLM(
        model=args.model_dir,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.model_dtype,
        enable_lora=True,
        max_lora_rank=args.max_lora_rank,
        max_model_len=args.max_input_length,
        enforce_eager=True
    )
    tokenizer=AutoTokenizer.from_pretrained(
        args.model_dir
    )

    lora_request=LoRARequest(
        lora_name="lora",
        lora_int_id=1,
        lora_path=args.lora_dir
    ) if args.lora_dir is not None else None
    CONFIG["max_tokens"]=args.max_output_length
    sampling_params=SamplingParams(**CONFIG)

    prompts=[]

    for item in chunks:
        pe=get_pe(item)
        pe=tokenizer.apply_chat_template(
            pe,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(pe)
    batch_num=(len(prompts)+args.batch_size-1)//args.batch_size
    outputs=[]
    for i in trange(batch_num):
        chunk=prompts[
            i*args.batch_size:
            min((i+1)*args.batch_size,len(prompts))
        ]
        if(len(chunk)==0):
            continue
        try:
            output=llm.generate(
                chunk,
                sampling_params=sampling_params,
                lora_request=lora_request,
                use_tqdm=False
            )
            outputs.extend(output)
        except Exception as e:
            tqdm.write(f"meet error in batch {i},error: {str(e)}")
            outputs.extend([None for _ in args.batch_size])

    predictions=[
        o.outputs[0].text if o is not None else "" for o in outputs
    ]
    return predictions

### You may need to modify the code here
def get_pe(item):
    pe=f'''你是一个数学专家，你擅长做数学计算。请你计算以下公式，并使用"[]"包裹答案。示例：
问题：5+7=? 
输出：[12]

问题：{item["question"]}
输出：（注意请不要输出其他无关内容！）'''
    return [
        {"role":"user","content":pe}
    ]

### You may need to modify the code here
def post_processing(content:str):
    match = re.search(r'\[(\d+)\]', content)
    if match:
        result = match.group(1)  # 获取第一个捕获组的内容
    else:
        result = None
    return result


def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model_dir",type=str,default=None)
    parser.add_argument("--lora_dir",type=str,default=None)
    parser.add_argument("--input_file",type=str,default=None)
    parser.add_argument("--output_file",type=str,default=None)
    parser.add_argument("--num_dp",type=int,default=2)
    parser.add_argument("--batch_size",type=int,default=4)
    parser.add_argument("--max_input_length",type=int,default=4096)
    parser.add_argument("--max_output_length",type=int,default=256)
    parser.add_argument("--max_lora_rank",type=int,default=32)
    parser.add_argument("--gpu_memory_utilization",type=float,default=0.9)
    parser.add_argument("--config_dir",type=str,default=None)    
    parser.add_argument("--model_dtype",type=str,default="auto")
    parser.add_argument("--debug",action="store_true")
    args=parser.parse_args()
    return args

if __name__=="__main__":
    args=get_args()
    # os.makedirs(args.output_dir,exist_ok=True)
    valid_data=read_jsonl(
        args.input_file
    )

    if args.debug:
        valid_data=valid_data[:10]

    num_dp=args.num_dp

    batch_size=(len(valid_data)+num_dp-1)//num_dp

    results=[]

    for i in range(num_dp):
        st=i*batch_size
        ed=min((i+1)*batch_size,len(valid_data))
        chunk=valid_data[st:ed]
        result=llminfer.remote(
            args,chunk
        )
        results.append(result)

    outputs=[]
    for result in results:
        outputs.extend(ray.get(result))

    for valid,output in zip(valid_data,outputs):
        valid.update({
            "response":post_processing(output),
            "original_content":output
        })
    
    ray.shutdown()
    write_jsonl(valid_data,args.output_file)
    print("finish")