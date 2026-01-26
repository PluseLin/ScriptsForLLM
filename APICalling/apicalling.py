import concurrent.futures
import time
from typing import List, Dict, Any

# 调用DeepSeekV3进行反事实文本改写
from tqdm import tqdm
import random
from openai import OpenAI
import time
import hashlib
import random
import argparse
import re

from fileio import *

def get_md5(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def process_element(args,prompt:str, ori_id:str, index: int) -> Dict:
    """
    处理单个元素的函数
    
    Args:
        prompt: 输入
        index: 线程索引
        
    Returns:
        Dict: 处理结果字典
    """
    client=clients[index]
    while True:
        try:
            response = client.chat.completions.create(
                model=args.model_name,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=args.temperature,
                top_p=args.top_p
            )
            content=response.choices[0].message.content
            response=post_processing(content)
            return {
                "idx":ori_id,
                "response":response,
                "original_content":content
            }
        except KeyboardInterrupt:
            exit(0)
        except Exception as e:
            tqdm.write(f"遇到错误:{str(e)}，休息5s")
            time.sleep(5)
            continue

def process_list_with_threadpool(args, input_list: List[Any]) -> List[Dict]:
    """
    使用线程池处理列表
    
    Args:
        input_list: 输入列表
    
    Returns:
        List[Dict]: 有序的结果列表
    """
    results = []
    with tqdm(total=len(input_list), desc="处理进度", unit="元素") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # 提交所有任务
            future_to_index = {
                executor.submit(process_element, args, element, old_id, index%thread_num): index
                for index, (element,old_id) in enumerate(input_list)
            }
            
            # 按完成顺序收集结果
            for future in concurrent.futures.as_completed(future_to_index):
                result = future.result()
                results.append(result)
                pbar.update(1)
    
    # 按原始索引排序
    return results

###需要自己修改###
def get_pe(item):
    return f'''你是一个数学专家，你擅长做数学计算。请你计算以下公式，并使用"[]"包裹答案。示例：
问题：5+7=? 
输出：[12]

问题：{item["question"]}
输出：（注意请不要输出其他无关内容！）'''

###需要自己修改###
def post_processing(content:str):
    match = re.search(r'\[(\d+)\]', content)
    if match:
        result = match.group(1)  # 获取第一个捕获组的内容
    else:
        result = None
    return result
        

###可能需要修改###
def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input_file",type=str,default=None)
    parser.add_argument("--output_file",type=str,default=None)
    parser.add_argument("--api_key",type=str,default=None)
    parser.add_argument("--model_name",type=str,default=None)
    parser.add_argument("--base_url",type=str,default=None)
    parser.add_argument("--thread_num",type=int,default=5)
    parser.add_argument("--temperature",type=float,default=1.0)
    parser.add_argument("--top_p",type=float,default=0.7)
    parser.add_argument("--debug",action="store_true")    
    args=parser.parse_args()
    return args

if __name__ == "__main__":
    args=get_args()

    thread_num=args.thread_num
    clients=[
        OpenAI(
            api_key=args.api_key,
            base_url=args.base_url,
        )
        for _ in range(thread_num)
    ]

    input_datas=read_jsonl(args.input_file)
    if args.debug:
        input_datas=random.sample(input_datas,20)
    inputs=[
        (
            get_pe(item),
            i
        )
        for i,item in enumerate(input_datas)
    ]

    merged_result=process_list_with_threadpool(args, inputs)
    
    for result in merged_result:
        idx=result["idx"]
        input_datas[idx]["response"]=result["response"]
        input_datas[idx]["original_content"]=result["original_content"]

    write_jsonl(
        input_datas,
        args.output_file
    )
    print("finish")