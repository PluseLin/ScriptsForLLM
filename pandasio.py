'''
pip install pandas
'''
import pandas as pd
import tempfile
import os

def read_csv_pd(src, delimiter='\t'):
    return pd.read_csv(src, delimiter=delimiter, encoding='utf-8')

def write_csv_pd(obj, dst, delimiter='\t'):
    obj.to_csv(dst, sep=delimiter, index=False, encoding='utf-8')

def jsonl2df(obj):
    return pd.DataFrame(obj)

def df2jsonl(obj):
    return obj.to_dict('records')

def show_head(df:pd.DataFrame, n=5):
    return df.head(n)

def show_tail(df:pd.DataFrame, n=5):
    return df.tail(n)

def show_random(df:pd.DataFrame, n=5):
    return df.sample(n)

def example_dataframe_iteration(df):
    """
    举例说明如何遍历pd.DataFrame每一行并操作特定键值对应的元素
    """
    # 方法1: 使用iterrows() - 返回(index, Series)元组
    print("方法1: 使用iterrows()")
    for index, row in df.iterrows():
        # 获取特定键值的元素
        name = row['name']  # 假设列名为'name'
        age = row['age']    # 假设列名为'age'
        # 进行操作
        print(f"行{index}: name={name}, age={age}")
    
    # 方法2: 使用itertuples() - 更快，返回命名元组
    print("\n方法2: 使用itertuples()")
    for row in df.itertuples(index=True, name='Row'):
        # 通过属性访问
        name = row.name  # 假设列名为'name'
        age = row.age    # 假设列名为'age'
        # 进行操作
        print(f"行{row.Index}: name={name}, age={age}")
    
    # 方法3: 使用apply()函数式操作
    print("\n方法3: 使用apply()")
    def process_row(row):
        name = row['name']
        age = row['age']
        # 返回处理后的值或执行操作
        return f"{name} is {age} years old"
    
    results = df.apply(process_row, axis=1)
    for result in results:
        print(result)
    
    # 方法4: 直接向量化操作（推荐，性能最佳）
    print("\n方法4: 向量化操作")
    # 对整列进行操作
    df['age_plus_10'] = df['age'] + 10
    df['name_upper'] = df['name'].str.upper()
    
    return df

def sort_pd(df, column, cmp=None):
    if cmp is None:
        return df.sort_values(by=column)
    else:
        return df.iloc[sorted(range(len(df)), key=lambda i: cmp(df.iloc[i][column]))]

# 创建示例DataFrame
if __name__ == "__main__":
    data = {
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35],
        'city': ['New York', 'London', 'Tokyo']
    }
    df = pd.DataFrame(data)
    
    # 运行示例函数
    result_df = example_dataframe_iteration(df)
    print("\n处理后的DataFrame:")
    print(result_df)