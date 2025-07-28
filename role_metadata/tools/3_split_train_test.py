import json
import os


role_list = ['Luna', 'hutao', 'raidenShogun', 'wanderer', 'ayaka', 'zhongli', 'liyunlong', 'wangduoyu', 'weixiaobao', 'jiumozhi', 'wangyuyan', 'Penny', 'zhangwuji', 'zhaomin', 'huangrong', 'guojing', 'wukong', 'HAL 9000', 'Colonel Nathan R. Jessep', 'Antonio Salieri', 'Stifler', 'Paul Vitti', 'Alvy Singer', 'Violet Weston', 'Willie Soke', 'Gaston', 'The Dude', 'Paul Conroy', 'Truman Capote', 'Mater', 'Andrew Detmer', 'Coriolanus', 'John Keating', 'Wade Wilson', 'Jim Morrison', 'Queen Elizabeth I', 'Jeff Spicoli', 'Fred Flintstone', 'Freddy Krueger', 'Tyrion Lannister', 'James Brown', 'Walt Kowalski', 'John Coffey', 'Theodore Twombly', 'Gregory House', 'Sonny', 'Colonel Hans Landa', 'Judge Dredd', 'Juno MacGuff', 'Professor G.H. Dorr', 'Fletcher Reede', 'Abraham Lincoln', 'Frank T.J. Mackey', 'Leonard Shelby', 'Harvey Milk', 'Randle McMurphy', 'Jack Sparrow', 'John Dillinger', 'Lestat de Lioncourt', 'Tyler Hawkins', 'James Carter', 'Jigsaw', 'John Doe', 'Sherlock Holmes', 'Shrek', 'Pat Solitano', 'Karl Childers', 'Bruno Antony', 'Seth', 'Caden Cotard', 'Travis Bickle', 'Stanley Ipkiss', 'Lyn Cassady', 'Michael Scott', 'Robert Angier', 'Dr. Frank-N-Furter', 'Jack Torrance', 'Tom Ripley', 'D_Artagnan', 'Thor', 'James Bond', 'Mark Renton', 'David Aames', 'Rorschach', 'Jordan Belfort', 'Logan', 'Judy Hoops', 'Doctor Who', 'Raylan Givens', 'Mary Sibley', 'Lucifer Morningstar', 'Twilight Sparkle', 'Oliver Queen', 'Klaus Mikaelson', 'Queen Catherine', 'Dr. Hannibal Lecter', 'Coach Eric Taylor', 'yaemiko']
cn_roles = ['hutao', 'raidenShogun', 'wanderer', 'ayaka', 'zhongli', 'liyunlong', 'wangduoyu', 'weixiaobao', 'jiumozhi', 'wangyuyan', 'zhangwuji', 'zhaomin', 'huangrong', 'guojing', 'wukong', 'yaemiko']


for turn in ["single", "multi"]:
    for role in role_list:
        if turn == "single":
            role_path = f"create_multiturn_data/2_1_dedup_singleturn/{role}.json"
        elif turn == "multi":
            role_path = f"create_multiturn_data/2_2_dedup_multiturn/{role}.json"
        train_save_path = f"create_multiturn_data/train/{turn}/{role}.json"
        test_save_path = f"create_multiturn_data/test/{turn}/{role}.json"

        # 创建文件夹
        os.makedirs(os.path.dirname(train_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(test_save_path), exist_ok=True)

        try:
            with open(role_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"读取{role_path}失败: {e}")
            continue

        if not data:
            print(f"{role_path} 为空，跳过")
            continue

        test_data = data[:2]
        train_data = data[2:] if len(data) > 2 else []

        # 读取角色profile
        profile_path = f"metadata/role_profiles/{role}_profile.txt"
        try:
            with open(profile_path, 'r', encoding='utf-8') as f:
                profile = f.read().strip()
        except Exception as e:
            print(f"读取{profile_path}失败: {e}")
            profile = ""

        # 为每条数据添加system_prompt
        def add_system_prompt(data_list):
            for item in data_list:
                if 'context' in item:
                    system_prompt = f"{profile}\n\nThe character script below contains context that may relate to the questions. Please consider it as a helpful reference when formulating your answers:\n[Character script]\n\n{item['context']}"
                    item['system_prompt'] = system_prompt
            return data_list

        test_data = add_system_prompt(test_data)
        train_data = add_system_prompt(train_data)

        # with open(test_save_path, 'w', encoding='utf-8') as f:
        #     json.dump(test_data, f, ensure_ascii=False, indent=2)
        with open(train_save_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)