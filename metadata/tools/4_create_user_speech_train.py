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
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 配置参数
appid = ""
access_token = ""
cluster = "volcano_tts"  # 火山引擎集群
voice_type = "BV702_streaming"  # 使用的语音模型
host = "openspeech.bytedance.com"
api_url = f"https://{host}/api/v1/tts"

# 文件路径配置
train_dir = "create_multiturn_data/train"
output_base_dir = "create_multiturn_data/train"

# 构造请求头
headers = {
    "Authorization": f"Bearer;{access_token}",
    "Content-Type": "application/json"
}

# 线程锁，用于保护计数器
counter_lock = threading.Lock()

def wav_to_base64(wav_path: str) -> str:
    with open(wav_path, "rb") as audio_file:
        base64_audio = base64.b64encode(audio_file.read()).decode('utf-8')
    return "data:audio/wav;base64," + base64_audio

def tts_request(text, output_path):
    return False
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
                "uid": "388808087185088"
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

def process_single_audio(args):
    """处理单个音频生成任务"""
    user_text, audio_path, turn = args
    
    # 如果音频文件不存在，则生成
    if not os.path.exists(audio_path):
        success = tts_request(user_text, audio_path)
        if success:
            # 生成base64编码
            try:
                base64_audio = wav_to_base64(audio_path)
                turn['user_speech_base64'] = base64_audio
            except Exception as e:
                print(f"生成base64编码失败 {audio_path}: {str(e)}")
        return success
    else:
        # 如果文件已存在，生成base64编码
        try:
            base64_audio = wav_to_base64(audio_path)
            turn['user_speech_base64'] = base64_audio
        except Exception as e:
            print(f"生成base64编码失败 {audio_path}: {str(e)}")
        return True

def process_train_data():
    """处理train数据，为每个user生成语音"""
    total_files = 0
    total_success = 0
    total_fail = 0
    
    # 先创建文件夹
    print("开始创建文件夹...")
    for turn_type in ["single", "multi"]:
        turn_dir = os.path.join(train_dir, turn_type)
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
        turn_dir = os.path.join(train_dir, turn_type)
        if not os.path.exists(turn_dir):
            print(f"目录不存在: {turn_dir}")
            continue
            
        # 获取所有json文件
        json_files = glob.glob(os.path.join(turn_dir, "*.json"))
        # 只处理前20个角色
        json_files = json_files
        total_files += len(json_files)
        
        print(f"\n处理 {turn_type} 类型数据，共 {len(json_files)} 个文件")
        
        for json_file in tqdm(json_files, desc=f"处理{turn_type}数据"):
            if not 'Tom Ripley' in json_file:
                continue
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
                
                # 收集所有需要处理的音频任务
                audio_tasks = []
                
                # 处理每条数据
                for i, item in enumerate(tqdm(data, desc=f"处理{role}数据", leave=False)):
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
                            
                            # 添加user_speech_path字段
                            turn['user_speech_path'] = audio_path
                            
                            # 添加到任务列表
                            audio_tasks.append((user_text, audio_path, turn))
                
                # 使用线程池并发处理音频生成
                with ThreadPoolExecutor(max_workers=40) as executor:
                    # 提交所有任务
                    future_to_task = {executor.submit(process_single_audio, task): task for task in audio_tasks}
                    
                    # 处理完成的任务
                    for future in tqdm(as_completed(future_to_task), total=len(audio_tasks), desc=f"生成{role}语音"):
                        try:
                            success = future.result()
                            with counter_lock:
                                if success:
                                    total_success += 1
                                else:
                                    total_fail += 1
                        except Exception as e:
                            print(f"处理音频任务时出错: {str(e)}")
                            with counter_lock:
                                total_fail += 1
                
                # 保存修改后的数据
                # with open(json_file, 'w', encoding='utf-8') as f:
                #     json.dump(data, f, ensure_ascii=False, indent=2)
                    
            except Exception as e:
                print(f"处理文件 {json_file} 时出错: {str(e)}")
                total_fail += 1
                continue
    
    print(f"\n处理完成！")
    print(f"总文件数: {total_files}")
    print(f"成功生成语音: {total_success} 个")
    print(f"失败: {total_fail} 个")

if __name__ == "__main__":
    process_train_data()