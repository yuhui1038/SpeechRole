# coding=utf-8
'''
requires Python 3.6 or later
pip install requests tqdm
'''

import base64
import json
import uuid
import requests
import os
from pathlib import Path
from tqdm import tqdm
import traceback
import glob

# 配置参数
uid = ""
appid = ""
access_token = ""
cluster = "volcano_tts"  # 火山引擎集群
voice_type = "BV702_streaming"  # 使用的语音模型
host = "openspeech.bytedance.com"
api_url = f"https://{host}/api/v1/tts"

# 文件路径配置
test_dir = "create_multiturn_data/test"
output_base_dir = "create_multiturn_data/test"

# 构造请求头
headers = {
    "Authorization": f"Bearer;{access_token}",
    "Content-Type": "application/json"
}

def wav_to_base64(wav_path: str) -> str:
    with open(wav_path, "rb") as audio_file:
        base64_audio = base64.b64encode(audio_file.read()).decode('utf-8')
    return "data:audio/wav;base64," + base64_audio

def tts_request(text, output_path):
    """
    向火山引擎 TTS 接口发送请求，生成语音并保存
    :param text: 要合成的文本
    :param output_path: 输出文件路径
    :return: 成功与否
    """
    try:
        req_json = {
            "app": {
                "appid": appid,
                "token": access_token,
                "cluster": cluster
            },
            "user": {
                "uid": uid
            },
            "audio": {
                "voice_type": voice_type,
                "encoding": "wav",
                "speed_ratio": 1.0,
                "volume_ratio": 1.0,
                "pitch_ratio": 1.0,
            },
            "request": {
                "reqid": str(uuid.uuid4()),
                "text": text,
                "text_type": "plain",
                "operation": "query",
                "with_frontend": 1,
                "frontend_type": "unitTson"
            }
        }

        resp = requests.post(api_url, json.dumps(req_json), headers=headers, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if "data" in data:
                audio_data = base64.b64decode(data["data"])
                with open(output_path, "wb") as f:
                    f.write(audio_data)
                return True
            else:
                print(f"TTS 响应无 data 字段: {data}")
                return False
        else:
            print(f"TTS 请求失败，状态码: {resp.status_code}, 响应: {resp.text}")
            return False

    except Exception as e:
        print(f"TTS 请求异常: {str(e)}")
        traceback.print_exc()
        return False

def process_test_data():
    """处理test数据，为每个user生成语音"""
    total_files = 0
    total_success = 0
    total_fail = 0
    
    # 先创建文件夹
    print("开始创建文件夹...")
    for turn_type in ["single", "multi"]:
        turn_dir = os.path.join(test_dir, turn_type)
        if not os.path.exists(turn_dir):
            print(f"目录不存在: {turn_dir}")
            continue
            
        json_files = glob.glob(os.path.join(turn_dir, "*.json"))
        
        for json_file in json_files:
            role = os.path.basename(json_file).replace('.json', '')
            speech_output_dir = os.path.join(output_base_dir, f"{turn_type}_speech", role)
            Path(speech_output_dir).mkdir(parents=True, exist_ok=True)
            print(f"创建文件夹: {speech_output_dir}")
    
    print("文件夹创建完成！\n")
    
    # 处理single和multi两种类型
    for turn_type in ["single", "multi"]:
        turn_dir = os.path.join(test_dir, turn_type)
        if not os.path.exists(turn_dir):
            print(f"目录不存在: {turn_dir}")
            continue
            
        # 获取所有json文件
        json_files = glob.glob(os.path.join(turn_dir, "*.json"))
        total_files += len(json_files)
        
        print(f"\n处理 {turn_type} 类型数据，共 {len(json_files)} 个文件")
        
        for json_file in tqdm(json_files, desc=f"处理{turn_type}数据"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if not data:
                    continue
                    
                # 获取角色名（从文件名）
                role = os.path.basename(json_file).replace('.json', '')
                
                # 为每个角色创建对应的语音输出目录
                speech_output_dir = os.path.join(output_base_dir, f"{turn_type}_speech", role)
                Path(speech_output_dir).mkdir(parents=True, exist_ok=True)
                
                # 处理每条数据
                for i, item in enumerate(data):
                    if 'dialogue' not in item:
                        continue
                        
                    dialogue = item['dialogue']
                    if not isinstance(dialogue, list):
                        continue
                    
                    # 为每个对话轮次生成语音
                    for j, turn in enumerate(dialogue):
                        if 'user' in turn and turn['user'].strip():
                            user_text = turn['user'].strip()
                            
                            # 生成音频文件名
                            audio_filename = f"{role}_{i}_{j}_user.wav"
                            audio_path = os.path.join(speech_output_dir, audio_filename)
                            
                            # 如果音频文件不存在，则生成
                            if not os.path.exists(audio_path):
                                success = tts_request(user_text, audio_path)
                                if success:
                                    total_success += 1
                                else:
                                    total_fail += 1
                            else:
                                total_success += 1
                            
                            # 添加user_speech_path字段
                            turn['user_speech_path'] = audio_path
                            
                            # 检查音频文件是否存在，如果存在则生成base64编码
                            if os.path.exists(audio_path):
                                try:
                                    base64_audio = wav_to_base64(audio_path)
                                    turn['user_speech_base64'] = base64_audio
                                except Exception as e:
                                    print(f"生成base64编码失败 {audio_path}: {str(e)}")
                
                # 保存修改后的数据
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                    
            except Exception as e:
                print(f"处理文件 {json_file} 时出错: {str(e)}")
                total_fail += 1
                continue
    
    print(f"\n处理完成！")
    print(f"总文件数: {total_files}")
    print(f"成功生成语音: {total_success} 个")
    print(f"失败: {total_fail} 个")

if __name__ == "__main__":
    process_test_data()