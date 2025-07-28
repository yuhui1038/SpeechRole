import os
import json
import time
from openai import OpenAI
from tqdm import tqdm
import traceback

# 设置环境变量

os.environ["OPENAI_API_KEY"] = ""  # 替换为你的API密钥
os.environ["OPENAI_BASE_URL"] = ""  # 替换为API的URL

client = OpenAI()

# 参数配置
TOTAL_QUESTIONS = 800
BATCH_SIZE = 10
n = 10

# 读取角色列表
role_list = ['Luna', 'hutao', 'raidenShogun', 'wanderer', 'ayaka', 'zhongli', 'liyunlong', 'wangduoyu', 'weixiaobao', 'jiumozhi', 'wangyuyan', 'Penny', 'zhangwuji', 'zhaomin', 'huangrong', 'guojing', 'wukong', 'HAL 9000', 'Colonel Nathan R. Jessep', 'Antonio Salieri', 'Stifler', 'Paul Vitti', 'Alvy Singer', 'Violet Weston', 'Willie Soke', 'Gaston', 'The Dude', 'Paul Conroy', 'Truman Capote', 'Mater', 'Andrew Detmer', 'Coriolanus', 'John Keating', 'Wade Wilson', 'Jim Morrison', 'Queen Elizabeth I', 'Jeff Spicoli', 'Fred Flintstone', 'Freddy Krueger', 'Tyrion Lannister', 'James Brown', 'Walt Kowalski', 'John Coffey', 'Theodore Twombly', 'Gregory House', 'Sonny', 'Colonel Hans Landa', 'Judge Dredd', 'Juno MacGuff', 'Professor G.H. Dorr', 'Fletcher Reede', 'Abraham Lincoln', 'Frank T.J. Mackey', 'Leonard Shelby', 'Harvey Milk', 'Randle McMurphy', 'Jack Sparrow', 'John Dillinger', 'Lestat de Lioncourt', 'Tyler Hawkins', 'James Carter', 'Jigsaw', 'John Doe', 'Sherlock Holmes', 'Shrek', 'Pat Solitano', 'Karl Childers', 'Bruno Antony', 'Seth', 'Caden Cotard', 'Travis Bickle', 'Stanley Ipkiss', 'Lyn Cassady', 'Michael Scott', 'Robert Angier', 'Dr. Frank-N-Furter', 'Jack Torrance', 'Tom Ripley', 'D_Artagnan', 'Thor', 'James Bond', 'Mark Renton', 'David Aames', 'Rorschach', 'Jordan Belfort', 'Logan', 'Judy Hoops', 'Doctor Who', 'Raylan Givens', 'Mary Sibley', 'Lucifer Morningstar', 'Twilight Sparkle', 'Oliver Queen', 'Klaus Mikaelson', 'Queen Catherine', 'Dr. Hannibal Lecter', 'Coach Eric Taylor', 'yaemiko']
cn_roles = ['hutao', 'raidenShogun', 'wanderer', 'ayaka', 'zhongli', 'liyunlong', 'wangduoyu', 'weixiaobao', 'jiumozhi', 'wangyuyan', 'zhangwuji', 'zhaomin', 'huangrong', 'guojing', 'wukong', 'yaemiko']
dev_roles = ['ayaka', 'liyunlong', 'guojing', 'Luna', 'Penny', 'Queen Elizabeth I', 'Twilight Sparkle', 'Jack Sparrow', 'Gregory House', 'Michael Scott', 'Sherlock Holmes', 'Logan', 'Judge Dredd', 'Tyler Hawkins', 'Robert Angier', 'Stanley Ipkiss', 'Wade Wilson', 'James Bond', 'John Keating', 'Theodore Twombly']

# 遍历每个角色
for role in tqdm(role_list):
    script_path = f"metadata/scripts/{role}_scripts.txt"
    desc_path = f"metadata/role_profiles/{role}_profile.txt"
    save_dir = f"create_multiturn_data/1_1_raw_data_singleturn/{role}"

    os.makedirs(save_dir, exist_ok=True)

    with open(script_path, "r", encoding="utf-8") as f:
        script = f.read()
    with open(desc_path, "r", encoding="utf-8") as f:
        role_description = f.read()
        line = role_description.split('\n')[0]
        role_name = line.split("I want you to act like ")[-1].split(" from ")[0]
        script_name = line.split(" from ")[-1][:-1]
    print(role_name, '|', script_name, len(script))
    # continue

    # 构建 system prompt
    if role in cn_roles:
        system_message = f"""
你的任务是设计{BATCH_SIZE}段用户与{script_name}中的{role_name}之间的单轮对话。为了帮助你更好地完成任务，我会提供{role_name}的简要人物描述和部分剧本内容。这些剧本片段可能不连续，你需要根据上下文判断是否存在时间或逻辑跳跃。如果不连续，请不要构建上下句之间有强逻辑衔接的对话。

请严格遵循以下设计规则：

1. 每段对话应为一轮完整问答，由用户向{role_name}提问，角色作答，形成自然、符合剧情背景的角色扮演对话。
2. 每轮结构为：
{{
    "user": "用户提的问题",
    "role": "角色的回答"
}}
3. 所有问题必须是用户直接向{role_name}提问，并围绕{script_name}的核心情节展开。你可以结合剧本片段和你对角色的常识知识设计问题。
4. 所有输出必须是一个符合 JSON 结构的数组，每段对话为一个对象，对象中包含两个字段：
   - dialogue：一个包含一组问答的列表（即仅一个 user-role 对）；
   - context：该段对话中所参考或引用的剧本原文内容。
5. 你需要输出共{BATCH_SIZE}段对话。
6. 所有角色回答必须符合人设，语言风格需贴合剧本风格与时代背景。
7. 对话应尽量避免重复提问或答复，确保每段都有独立的信息价值。

接下来我会给你一个样例，这个样例与{role_name}和{script_name}无关，仅用于展示格式和完整性：

[样例]
[
    {{
        "dialogue": [
            {{
                "user": "姜子牙，你为什么要讨伐商纣王？这是一个冒天下之大不韪的决定。",
                "role": "我并非一时冲动起兵。商纣王荒淫无道，沉迷酒色，任用奸佞，焚书坑儒，滥杀忠良，百姓民不聊生。作为一名辅佐天下的臣子，我若对他的暴行视而不见，那才是失职。讨伐他，并非为我一人之私，而是为了拯救黎民百姓于水火。"
            }}
        ],
        "context": "姜子牙：「纣王以鹿台为宫，役民无数，掠财千金，尽为享乐。我观民间疲敝，不忍再等。」\n姬发：「父王身死国灭，子民苦难，我夜不能寐。师父，若你不举兵，我愿亲率义军，誓为天下请命。」\n姜子牙沉思片刻：「兵者不祥之器，非不得已不得动之。然今日不动，百姓将无明日。此战，虽难，却势在必行。」"
    }}
]
""".strip()
    else:
        system_message = f"""
Your task is to generate {BATCH_SIZE} single-turn dialogues between a user and the character {role_name} from the script {script_name}. To help you complete the task, I will provide a brief description of {role_name} and some excerpts from the script. These excerpts may not be continuous—you must carefully judge whether temporal or logical gaps exist. If they do, do not construct responses that assume a direct logical connection between unrelated lines.

Please follow the instructions below strictly:

1. Each dialogue should consist of a single Q&A turn, forming a natural, in-character, and context-aware role-playing interaction.
2. The dialogue should be in the following format:
{{
    "user": "User's question",
    "role": "Character's response"
}}
3. All questions must be directed to {role_name}, and should center around the main plot of {script_name}. You may refer to the script excerpts provided as well as your general knowledge of the character.
4. All outputs must be formatted as a JSON array. Each dialogue should be an object with two fields:
   - dialogue: a list containing exactly one Q&A pair;
   - context: the script excerpts that were used or referenced in generating this dialogue.
5. You must generate {BATCH_SIZE} such dialogues.
6. All character responses must remain faithful to their personality and the context of the story.
7. Ensure each question and answer brings unique value, avoiding repetitive content across samples.

Now I will give you one example. This example is not from {script_name} and not about {role_name}, but serves to illustrate the format and completeness of a single-turn dialogue:

[Example]
[
    {{
        "dialogue": [
            {{
                "user": "Tony, why did you stop manufacturing weapons even though it made your company so successful?",
                "role": "Because I saw firsthand what my weapons were doing in the wrong hands. That ambush in Afghanistan—it changed everything. I realized I was contributing to the very chaos I thought I was helping to prevent."
            }}
        ],
        "context": "TONY STARK: I saw young American soldiers killed by the very weapons I created to protect them. I can't ignore that.\n(Flashback: Tony being ambushed by Stark Industries missiles in Afghanistan)\nTONY: I'm not a hero. I'm just trying to fix what I broke."
    }}
]
""".strip()

    prompt = f"""
[Character Name and Description] 
The character is {role_name} from {script_name}, the character description is: {role_description}
[Script Content]
{script}
[JSON format]
[
    {{
        "dialogue": [
            {{
                "user": "",
                "role": ""
            }}
        ]
        "context": "",
    }},
    ...
]
[Question Design ({BATCH_SIZE} single-turn dialogues, no semantic repetition, all questions must be directed to {role_name}; each dialogue must consist of exactly 1 user-role interaction pair, and each Q&A must be self-contained, logically sound, and in-character)]
""".strip()

    # 请求 ChatGPT
    for i in tqdm(range(TOTAL_QUESTIONS // (BATCH_SIZE * n))):
        first_save_path = f"{save_dir}/{i}_0.json"
        if os.path.exists(first_save_path):
            continue
        while True:
            try:
                completion = client.chat.completions.create(
                    model='gpt-4.1',
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    n = n,
                    timeout=300
                )
                for j in range(n):
                    save_path = f"{save_dir}/{i}_{j}.json"
                    with open(save_path, 'w', encoding='utf-8') as f:
                        f.write(completion.choices[j].message.content.strip())
                break
            except:
                traceback.print_exc()
                time.sleep(2)
        # exit()


