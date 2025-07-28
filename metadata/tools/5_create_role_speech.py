import json
import base64
import os
import sys
import argparse
sys.path.append('CosyVoice/third_party/Matcha-TTS')
sys.path.append('CosyVoice')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from tqdm import tqdm
import torchaudio


def wav_to_base64(wav_path: str) -> str:
    with open(wav_path, "rb") as audio_file:
        base64_audio = base64.b64encode(audio_file.read()).decode('utf-8')
    return "data:audio/wav;base64," + base64_audio


role_list = ['Luna', 'hutao', 'raidenShogun', 'wanderer', 'ayaka', 'zhongli', 'liyunlong', 'wangduoyu', 'weixiaobao', 'jiumozhi', 'wangyuyan', 'Penny', 'zhangwuji', 'zhaomin', 'huangrong', 'guojing', 'wukong', 'HAL 9000', 'Colonel Nathan R. Jessep', 'Antonio Salieri', 'Stifler', 'Paul Vitti', 'Alvy Singer', 'Violet Weston', 'Willie Soke', 'Gaston', 'The Dude', 'Paul Conroy', 'Truman Capote', 'Mater', 'Andrew Detmer', 'Coriolanus', 'John Keating', 'Wade Wilson', 'Jim Morrison', 'Queen Elizabeth I', 'Jeff Spicoli', 'Fred Flintstone', 'Freddy Krueger', 'Tyrion Lannister', 'James Brown', 'Walt Kowalski', 'John Coffey', 'Theodore Twombly', 'Gregory House', 'Sonny', 'Colonel Hans Landa', 'Judge Dredd', 'Juno MacGuff', 'Professor G.H. Dorr', 'Fletcher Reede', 'Abraham Lincoln', 'Frank T.J. Mackey', 'Leonard Shelby', 'Harvey Milk', 'Randle McMurphy', 'Jack Sparrow', 'John Dillinger', 'Lestat de Lioncourt', 'Tyler Hawkins', 'James Carter', 'Jigsaw', 'John Doe', 'Sherlock Holmes', 'Shrek', 'Pat Solitano', 'Karl Childers', 'Bruno Antony', 'Seth', 'Caden Cotard', 'Travis Bickle', 'Stanley Ipkiss', 'Lyn Cassady', 'Michael Scott', 'Robert Angier', 'Dr. Frank-N-Furter', 'Jack Torrance', 'Tom Ripley', 'D_Artagnan', 'Thor', 'James Bond', 'Mark Renton', 'David Aames', 'Rorschach', 'Jordan Belfort', 'Logan', 'Judy Hoops', 'Doctor Who', 'Raylan Givens', 'Mary Sibley', 'Lucifer Morningstar', 'Twilight Sparkle', 'Oliver Queen', 'Klaus Mikaelson', 'Queen Catherine', 'Dr. Hannibal Lecter', 'Coach Eric Taylor', 'yaemiko']

# 读取角色音频文本
with open('metadata/role_voices/audios_text.json', 'r', encoding='utf-8') as f:
    audios_text = json.load(f)

# 初始化 cosyvoice2
cosyvoice = CosyVoice2('CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False)

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_part', type=int, default=1, help='总分片数')
    parser.add_argument('--current_part', type=int, default=1, help='当前分片编号从1开始')
    parser.add_argument('--mode', type=str, default='test', choices=['test', 'train'], help='处理test还是train数据')
    return parser.parse_args()

args = parse_args()
total_part = args.total_part
current_part = args.current_part
mode = args.mode
role_list = role_list[current_part-1::total_part]

if mode == 'test':
    data_base_path = 'create_multiturn_data/test'
else:
    data_base_path = 'create_multiturn_data/train'

for role in tqdm(role_list, desc='角色进度'):
    role_speech_text = audios_text.get(role, "")
    for turn in ['single', 'multi']:
        data_path = f"{data_base_path}/{turn}/{role}.json"
        output_base_path = f"create_multiturn_data/{mode}/{turn}_speech"
        if not os.path.exists(data_path):
            continue
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for i, d in enumerate(data):
            # 假设每条数据有多轮对话
            for j, turn_data in enumerate(d['dialogue']):
                text = turn_data['role']
                role_audio_path = f"metadata/role_voices/{role}.wav"
                role_audio_text = audios_text[role]
                prompt_speech_16k = load_wav(role_audio_path, 16000)
                output_dir = f"{output_base_path}/{role}"
                os.makedirs(output_dir, exist_ok=True)
                output_path = f"{output_dir}/{role}_{i}_{j}_role.wav"
                for k, result in enumerate(cosyvoice.inference_zero_shot(text, role_audio_text, prompt_speech_16k, stream=False)):
                    torchaudio.save(output_path, result['tts_speech'], cosyvoice.sample_rate)
                    break  # 只取第一个结果
                base64_str = wav_to_base64(output_path)
                turn_data['role_speech_path'] = output_path
                turn_data['role_speech_base64'] = base64_str

        # 保存
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    