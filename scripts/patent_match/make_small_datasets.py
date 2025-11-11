# python
import os
import json
import random

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    random.seed(42)
    base = os.path.join('data', 'patent_match')
    os.makedirs(base, exist_ok=True)

    train_path = os.path.join(base, 'train.json')
    test_path = os.path.join(base, 'test.json')
    out_train = os.path.join(base, 'train_10.json')
    out_test = os.path.join(base, 'test_20.json')

    if not os.path.exists(train_path):
        raise FileNotFoundError(f'未找到训练集: {train_path}')

    train_data = load_json(train_path)

    if os.path.exists(test_path):
        test_data = load_json(test_path)
        small_train = random.sample(train_data, min(10, len(train_data)))
        small_test = random.sample(test_data, min(20, len(test_data)))
    else:
        n = len(train_data)
        need_train, need_test = 10, 20
        if n < 1:
            raise ValueError('训练集为空')
        idx = list(range(n))
        random.shuffle(idx)
        t_n = min(need_train, n)
        r_n = n - t_n
        v_n = min(need_test, max(0, r_n))
        train_idx = idx[:t_n]
        test_idx = idx[t_n:t_n+v_n]
        small_train = [train_data[i] for i in train_idx]
        small_test = [train_data[i] for i in test_idx]

    save_json(out_train, small_train)
    save_json(out_test, small_test)

    print(f'已保存: {out_train} -> {len(small_train)} 条')
    print(f'已保存: {out_test} -> {len(small_test)} 条')

if __name__ == '__main__':
    main()
