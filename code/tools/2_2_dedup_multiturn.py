import json
from glob import glob
import traceback
import os
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


role_list = ['Luna', 'hutao', 'raidenShogun', 'wanderer', 'ayaka', 'zhongli', 'liyunlong', 'wangduoyu', 'weixiaobao', 'jiumozhi', 'wangyuyan', 'Penny', 'zhangwuji', 'zhaomin', 'huangrong', 'guojing', 'wukong', 'HAL 9000', 'Colonel Nathan R. Jessep', 'Antonio Salieri', 'Stifler', 'Paul Vitti', 'Alvy Singer', 'Violet Weston', 'Willie Soke', 'Gaston', 'The Dude', 'Paul Conroy', 'Truman Capote', 'Mater', 'Andrew Detmer', 'Coriolanus', 'John Keating', 'Wade Wilson', 'Jim Morrison', 'Queen Elizabeth I', 'Jeff Spicoli', 'Fred Flintstone', 'Freddy Krueger', 'Tyrion Lannister', 'James Brown', 'Walt Kowalski', 'John Coffey', 'Theodore Twombly', 'Gregory House', 'Sonny', 'Colonel Hans Landa', 'Judge Dredd', 'Juno MacGuff', 'Professor G.H. Dorr', 'Fletcher Reede', 'Abraham Lincoln', 'Frank T.J. Mackey', 'Leonard Shelby', 'Harvey Milk', 'Randle McMurphy', 'Jack Sparrow', 'John Dillinger', 'Lestat de Lioncourt', 'Tyler Hawkins', 'James Carter', 'Jigsaw', 'John Doe', 'Sherlock Holmes', 'Shrek', 'Pat Solitano', 'Karl Childers', 'Bruno Antony', 'Seth', 'Caden Cotard', 'Travis Bickle', 'Stanley Ipkiss', 'Lyn Cassady', 'Michael Scott', 'Robert Angier', 'Dr. Frank-N-Furter', 'Jack Torrance', 'Tom Ripley', 'D_Artagnan', 'Thor', 'James Bond', 'Mark Renton', 'David Aames', 'Rorschach', 'Jordan Belfort', 'Logan', 'Judy Hoops', 'Doctor Who', 'Raylan Givens', 'Mary Sibley', 'Lucifer Morningstar', 'Twilight Sparkle', 'Oliver Queen', 'Klaus Mikaelson', 'Queen Catherine', 'Dr. Hannibal Lecter', 'Coach Eric Taylor', 'yaemiko']
cn_roles = ['hutao', 'raidenShogun', 'wanderer', 'ayaka', 'zhongli', 'liyunlong', 'wangduoyu', 'weixiaobao', 'jiumozhi', 'wangyuyan', 'zhangwuji', 'zhaomin', 'huangrong', 'guojing', 'wukong', 'yaemiko']

# 设定模型
model_en = SentenceTransformer('all-MiniLM-L6-v2')
model_zh = SentenceTransformer('shibing624/text2vec-bge-large-chinese')

for role in tqdm(role_list):
    files = glob(f"create_multiturn_data/1_2_raw_data_generated/{role}/*.json")
    out_path = f"create_multiturn_data/2_2_dedup_multiturn/{role}.json"

    json_list = []
    for file in files:
        try:
            with open(file, 'r') as f:
                js = json.load(f)
        except:
            # if os.path.exists(file[:-6] + "0.json"):
            #     os.remove(file[:-6] + "0.json")
            # traceback.print_exc()
            js = []
        json_list.extend(js)
    # 只保留有'user'的
    json_list = [x for x in json_list if 'dialogue' in x and 'user' in x['dialogue'][0]]
    # print(role, len(json_list))
    # continue
    # 取所有user内容
    user_texts = [x['dialogue'][0]['user'] for x in json_list]
    # 选择模型
    if role in cn_roles:
        model = model_zh
        threshold = 0.85
    else:
        model = model_en
        threshold = 0.9
    # 计算embedding
    embeddings = model.encode(user_texts, convert_to_tensor=True, show_progress_bar=True)
    # 计算相似度矩阵
    cos_scores = util.pytorch_cos_sim(embeddings, embeddings)
    # 去重
    keep_idx = []
    removed = set()
    for i in range(len(json_list)):
        if i in removed:
            continue
        keep_idx.append(i)
        for j in range(i+1, len(json_list)):
            if cos_scores[i][j] > threshold:
                removed.add(j)
    deduped_json_list = [json_list[i] for i in keep_idx]
    print(role, len(json_list), '->', len(deduped_json_list))

    # 保存去重后的结果
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(deduped_json_list, f, ensure_ascii=False, indent=2)