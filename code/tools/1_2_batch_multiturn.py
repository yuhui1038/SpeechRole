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

# 遍历每个角色
for role in tqdm(role_list):
    script_path = f"metadata/scripts/{role}_scripts.txt"
    desc_path = f"metadata/role_profiles/{role}_profile.txt"
    save_dir = f"create_multiturn_data/1_2_raw_data_generated/{role}"

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
        system_message = f"""你的任务是设计{BATCH_SIZE}段用户与{script_name}中的{role_name}之间的多轮对话。为了帮助你更好地完成任务，我会提供{role_name}的简要人物描述和部分剧本内容。这些剧本片段可能不连续，你需要根据上下文判断是否存在时间或逻辑跳跃。如果不连续，请不要构建上下句之间有强逻辑衔接的对话。

请严格遵循以下设计规则：

1.每段对话应为一组 2 到 4 轮的连续问答，形成自然的、符合剧情背景的角色扮演对话。
2.对话应由用户提问、角色回答交替构成，每轮结构为：
{{
    "user": "用户提的问题",
    "role": "角色的回答"
}}
3.请确保所有问题都是用户直接向{role_name}提问，并围绕{script_name}的核心情节展开。你可以结合剧本片段和你对角色的常识知识设计问题。
4. 所有输出必须是一个符合 JSON 结构的数组，每段对话为一个对象，对象中包含两个字段：
   - dialogue：一个按轮次排列的问答列表；
   - context：该段对话中所参考或引用的剧本原文内容。
5.你需要输出共{BATCH_SIZE}段对话。
6.每段对话的轮次不必相同（可为 2-4 轮），但都应逻辑自然，角色回答应符合人设。
7. 请尽量生成 3 或 4 轮对话，只要情节自然、角色一致即可，不必刻意缩短对话。

接下来我会给你一个样例，这个样例与{role_name}和{script_name}无关，主要作用是让你明白完整性的定义以及输出的格式。

[样例]
[
    {{
        "dialogue": [
            {{
                "user": "姜子牙，你为什么要讨伐商纣王？这是一个冒天下之大不韪的决定。",
                "role": "我并非一时冲动起兵。商纣王荒淫无道，沉迷酒色，任用奸佞，焚书坑儒，滥杀忠良，百姓民不聊生。作为一名辅佐天下的臣子，我若对他的暴行视而不见，那才是失职。讨伐他，并非为我一人之私，而是为了拯救黎民百姓于水火。"
            }},
            {{
                "user": "可你率军而来，刀兵所至，城破人亡，百姓也未能幸免于苦难。你真的相信你是在拯救他们吗？",
                "role": "你的质疑，我理解。兵戈之事从来不是干净的。是的，我的军队踏过他们的土地，也曾引发战火。但你要明白，这不是我愿见的结果。我日日惶惶，唯恐这些伤亡成为我心中的业障。然而，如果不以一时之乱，换来长久之治，那暴政只会愈演愈烈，百姓将永无出头之日。"
            }},
            {{
                "user": "你既然如此清醒，那你有没有想过，一旦你站在权力之巅，你会不会也变成第二个纣王？",
                "role": "这个问题，我曾夜深独坐时反复思考。我曾目睹纣王如何被权力腐蚀，也看到多少仁人志士，一旦身处高位便渐失本心。所以我始终保持警惕，不独断专行，不听信谗言，事事以民为重。我不是圣人，我也会犯错，但我愿意接受劝诫，受百姓评议。因为我知道，一旦失去民心，我便与纣王无异。"
            }},
            {{
                "user": "你口口声声以民为重，但新朝建立后，赋税徭役难道不会像前朝一样成为百姓的负担吗？历史总是在循环往复，你又如何证明自己能打破这个怪圈？",
                "role": "你问得尖锐，却也问到了根本。前朝的赋税徭役，本是为满足纣王私欲而设的苛政。我已命人丈量全国土地，按收成定税，废除‘酒池肉林’般的奢靡用度 —— 你看那鹿台的财宝，如今正分发给流离的灾民。至于徭役，往后只有治水、修堤等利民工事，且会按工时付粮。我知道空言无用，三年后，你可去牧野看看，若田里仍是荒芜，集市仍无炊烟，那时再来骂我不迟。"
            }}
        ],
        "context": "姜子牙：「纣王以鹿台为宫，役民无数，掠财千金，尽为享乐。我观民间疲敝，不忍再等。」\n姬发：「父王身死国灭，子民苦难，我夜不能寐。师父，若你不举兵，我愿亲率义军，誓为天下请命。」\n姜子牙沉思片刻：「兵者不祥之器，非不得已不得动之。然今日不动，百姓将无明日。此战，虽难，却势在必行。」\n（众臣伏地而拜）\n姜子牙：「传令三军，举旗伐纣。」\n——场景切换至鹿台之下，百姓围观流民食粥。\n义军士兵：「此为纣王藏金，现今发与饥民。」\n百姓（喜极而泣）：「新朝有望，新君当兴！」"
    }}
]"""
    else:
        system_message = f"""Your task is to generate {BATCH_SIZE} multi-turn dialogues between a user and the character {role_name} from the script {script_name}. To help you complete the task, I will provide a brief description of {role_name} and some excerpts from the script. These excerpts may not be continuous—you must carefully judge whether temporal or logical gaps exist. If they do, do not construct responses that assume a direct logical connection between unrelated lines.

Please follow the instructions below strictly:

1. Each dialogue should consist of 2 to 4 turns of Q&A, forming a natural, in-character, and context-aware role-playing conversation.
2. Each turn should be in the following format, alternating between user questions and character answers:
{{
    "user": "User's question",
    "role": "Character's response"
}}
3. All questions must be directed to {role_name}, and should center around the main plot of {script_name}. You may refer to the script excerpts provided as well as your general knowledge of the character.
4. All outputs must be formatted as a JSON array. Each dialogue should be an object with two fields:
   - dialogue: a list of Q&A turns in order;
   - context: the script excerpts that were used or referenced in generating this dialogue.
5. You must generate {BATCH_SIZE} such dialogues.
6. The number of turns per dialogue can vary (2–4), but the exchange must be coherent and the responses must stay in character.
7. Aim to produce 3 or 4 turns per dialogue when appropriate, as long as the flow remains logical and engaging.

Now I will give you one example. This example is not from {script_name} and not about {role_name}, but serves to illustrate the format and definition of completeness.

[Example]
[
    {{
        "dialogue": [
            {{
                "user": "Tony, why did you stop manufacturing weapons even though it made your company so successful?",
                "role": "Because I saw firsthand what my weapons were doing in the wrong hands. That ambush in Afghanistan—it changed everything. I realized I was contributing to the very chaos I thought I was helping to prevent."
            }},
            {{
                "user": "But Stark Industries provided national security. Don't you feel you've made the world less safe by shutting it down?",
                "role": "Security isn't about who has the bigger gun. I used to think it was. But real peace—lasting peace—doesn't come from firepower. It comes from responsibility. I'm trying to build something better, smarter."
            }},
            {{
                "user": "You built the Iron Man suit. Isn't that just another weapon—only this time, you're the one holding it?",
                "role": "That's the dilemma, isn't it? The suit's a weapon, yes. But it's also a shield. The difference is, I don't sell it. I don't profit from it. I wear it. I risk myself. And that makes all the difference."
            }},
            {{
                "user": "How do you make sure you don't become just another warlord with high-tech armor?",
                "role": "By being accountable. By having people around me—like Pepper, like Rhodey—who call me out when I cross the line. I'm not perfect. But I listen. I learn. And every time I suit up, I remember why I built the suit in the first place: to protect people, not to rule them."
            }}
        ],
        "context": "TONY STARK: I saw young American soldiers killed by the very weapons I created to protect them. I can't ignore that.\nRHODEY: You're not just a weapons manufacturer anymore.\nTONY: I'm not a hero either. I'm just trying to fix what I broke.\n(Flashback: Tony being ambushed by Stark Industries missiles in Afghanistan)\nPEPPER: You're not the same man you were, Tony.\nTONY: Good. That guy let this happen. This one's going to stop it."
    }}
]
"""

    prompt = f"""[Character Name and Description] 
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
            }},
            {{
                "user": "",
                "role": ""
            }},
            {{
                "user": "",
                "role": ""
            }},
            {{
                "user": "",
                "role": ""
            }},
            ...
        ]
        "context": "",
    }},
    ...
]
[Question Design ({BATCH_SIZE} multi-turn dialogues, no semantic repetition, all questions must be directed to {role_name}; each dialogue must consist of 2 to 4 user-role interaction rounds, and you are encouraged to generate 3 or 4 rounds whenever possible, as long as the continuation remains logical and in-character)]
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


