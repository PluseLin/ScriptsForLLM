from fileio import *
from pandasio import *

# 测试用例
def test_json_functions():
    print("测试JSON读写功能...")
    test_data = {"name": "张三", "age": 25, "city": "北京"}
    write_json(test_data, "test.json")
    read_data = read_json("test.json")
    assert test_data == read_data, "JSON读写测试失败"
    print("JSON读写测试通过")

def test_jsonl_functions():
    print("测试JSONL读写功能...")
    test_data = [{"id": 1, "name": "张三"}, {"id": 2, "name": "李四"}]
    write_jsonl(test_data, "test.jsonl")
    read_data = read_jsonl("test.jsonl")
    assert test_data == read_data, "JSONL读写测试失败"
    print("JSONL读写测试通过")

def test_txt_functions():
    print("测试TXT读写功能...")
    test_data = "第一行\n第二行\n第三行"
    write_txt(test_data, "test.txt")
    read_data = read_txt("test.txt")
    assert test_data == read_data, "TXT读写测试失败"
    print("TXT读写测试通过")

def test_pandas_functions():
    # 测试数据
    data = [
        {"name": "张三", "age": 25, "city": "北京"},
        {"name": "李四", "age": 30, "city": "上海"}
    ]
    df = pd.DataFrame(data)
    
    # 测试jsonl2df和df2jsonl
    print("测试DataFrame与JSONL转换...")
    df_from_jsonl = jsonl2df(data)
    assert df.equals(df_from_jsonl), "jsonl2df转换失败"
    
    jsonl_from_df = df2jsonl(df)
    assert data == jsonl_from_df, "df2jsonl转换失败"
    print("DataFrame与JSONL转换测试通过")
    
    # 测试CSV读写
    print("测试CSV读写功能...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # 写入CSV
        write_csv_pd(df, tmp_path, delimiter=',')
        
        # 读取CSV
        df_read = read_csv_pd(tmp_path, delimiter=',')
        
        # 比较数据
        assert df.equals(df_read), "CSV读写测试失败"
        print("CSV读写测试通过")
        
        # 检查是否包含index列
        with open(tmp_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            assert not first_line.startswith('Unnamed: 0,'), "CSV文件包含index列"
        print("CSV无index列测试通过")
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    
    print("所有pandas函数测试完成！")

def test_sort_pd():
    # 创建测试数据
    data = {
        'name': ['Alice', 'Bob', 'Charlie', 'David'],
        'age': [30, 25, 35, 28],
        'score': [85, 92, 78, 88]
    }
    df = pd.DataFrame(data)
    
    print("原始DataFrame:")
    print(df)
    print()
    
    # 测试1: 默认排序（升序）
    print("测试1: 按age列默认排序（升序）")
    sorted_df1 = sort_pd(df, 'age')
    print(sorted_df1)
    assert list(sorted_df1['age']) == [25, 28, 30, 35], "默认排序测试失败"
    print("默认排序测试通过\n")
    
    # 测试2: 自定义比较函数（降序）
    print("测试2: 按age列自定义排序（降序）")
    def descending_cmp(x):
        return -x  # 降序排序
    sorted_df2 = sort_pd(df, 'age', cmp=descending_cmp)
    print(sorted_df2)
    assert list(sorted_df2['age']) == [35, 30, 28, 25], "降序排序测试失败"
    print("降序排序测试通过\n")
    
    # 测试3: 自定义复杂比较函数
    print("测试3: 按score列奇偶排序（偶数在前）")
    def parity_cmp(x):
        # 偶数优先，然后按值大小
        return (x % 2, x)  # 偶数(0) < 奇数(1)，然后按值排序
    sorted_df3 = sort_pd(df, 'score', cmp=parity_cmp)
    print(sorted_df3)
    # 验证：偶数[88, 92, 78]在前，奇数[85]在后
    assert list(sorted_df3['score']) == [78,88,92,85], "奇偶排序测试失败"
    print("奇偶排序测试通过\n")
    
    # 测试4: 按字符串列排序
    print("测试4: 按name列长度排序")
    def name_length_cmp(x):
        return len(x)  # 按名字长度排序
    sorted_df4 = sort_pd(df, 'name', cmp=name_length_cmp)
    print(sorted_df4)
    # 验证名字长度: Alice(5), Bob(3), Charlie(7), David(5)
    # 同长度时保持原始相对顺序
    assert list(sorted_df4['name']) == ['Bob', 'Alice', 'David', 'Charlie'], "名字长度排序测试失败"
    print("名字长度排序测试通过\n")
    
    print("所有排序测试用例通过！")

if __name__ == "__main__":
    test_sort_pd()