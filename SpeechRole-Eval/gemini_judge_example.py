from openai import OpenAI
import base64
import os
import json
import re
import argparse
from tqdm import tqdm
import time
import traceback
import torchaudio
import tempfile
import torch


model="gemini-2.5-pro-preview-06-05"

api_keys = [
    "",
    ]
current_key_index = 0

def get_client():
    """获取当前API key的客户端"""
    global current_key_index
    return OpenAI(
        api_key=api_keys[current_key_index],
        base_url=""
    )

def switch_to_next_key():
    """切换到下一个API key"""
    global current_key_index
    current_key_index = (current_key_index + 1) % len(api_keys)
    print(f"切换到API key {current_key_index + 1}/{len(api_keys)}: {api_keys[current_key_index][:20]}...")
    return get_client()

# 初始化客户端
client = get_client()

def wav_to_base64(wav_path: str) -> str:
    with open(wav_path, "rb") as audio_file:
        base64_audio = base64.b64encode(audio_file.read()).decode('utf-8')
    return base64_audio
    # return "data:audio/wav;base64," + base64_audio

# def wav_to_base64(wav_path: str) -> str:
#     TARGET_SR = 16000
#     waveform, sample_rate = torchaudio.load(wav_path)
#     # 如果不是单声道，转为单声道
#     if waveform.shape[0] > 1:
#         waveform = torch.mean(waveform, dim=0, keepdim=True)
#     # 重采样
#     if sample_rate != TARGET_SR:
#         resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=TARGET_SR)
#         waveform = resampler(waveform)
#         sample_rate = TARGET_SR
#     # 保存到临时文件
#     with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpfile:
#         torchaudio.save(tmpfile.name, waveform, sample_rate)
#         tmpfile_path = tmpfile.name
#     # 读取并编码
#     with open(tmpfile_path, "rb") as audio_file:
#         base64_audio = base64.b64encode(audio_file.read()).decode('utf-8')
#     os.remove(tmpfile_path)
#     return base64_audio

def parse_gemini_result(result_text: str) -> dict:
    """解析Gemini的结果，提取reason和score"""
    # 查找分数模式，如 [Scores]: (6, 9) 或 [Scores]: (6,9)
    score_pattern = r'\[Scores\]:\s*\((\d+),\s*(\d+)\)'
    score_match = re.search(score_pattern, result_text)
    
    score_a = int(score_match.group(1))
    score_b = int(score_match.group(2))
    score = (score_a, score_b)
    
    return {
        "reason": result_text.strip(),
        "score": score
    }


metrics = {
    "Instruction_Adherence": "Instruction Adherence: Do the spoken responses strictly follow the task instruction, remaining fully in character without any out-of-role explanations or assistant-like meta-comments?",
    "Speech_Fluency": "Speech Fluency: Are the responses delivered fluently, with smooth articulation, appropriate pacing, and minimal disfluencies such as stuttering or unnatural pauses?",
    "Conversational_Coherence": "Conversational Coherence: Do the responses maintain logical consistency within the dialogue, aligning with previous content without contradictions or abrupt topic shifts?",
    "Speech_Naturalness": "Speech Naturalness: Do the responses sound natural, human-like, and free from noticeable artifacts or robotic effects typically associated with synthetic speech?",
    "Prosodic_Consistency": "Prosodic Consistency: Does the prosody, including pitch, stress, and intonation, align with the character's intended speaking style and remain consistent across the discourse?",
    "Emotion_Appropriateness": "Emotion Appropriateness: Are emotional cues in the speech (e.g., anger, joy, sadness) well-aligned with the dialogue context and the character's emotional state?",
    "Personality_Consistency": "Personality Consistency: Do the responses consistently reflect the character's personality traits, such as optimism, sarcasm, or authority?",
    "Knowledge_Consistency": "Knowledge Consistency: Are the responses grounded in the character's established background, knowledge, and relationships, without fabricating out-of-character facts?"
}

roles = ['Coriolanus']

def main(mode, test_model):
    global client  # 声明client为全局变量
    
    for role in tqdm(roles):
        save_path = f"test_results/{test_model}/{mode}/{role}.json"
        if os.path.exists(save_path):
            print(f"{save_path} 已存在，跳过该角色。")
            continue
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        gt_data_path = f"test_data/{mode}_turn/{role}.json"
        with open(gt_data_path, 'r') as f:
            gt_data = json.load(f)

        line = gt_data[0]['system_prompt'].split('\n')[0]
        role_name = line.split("I want you to act like ")[-1][:-1]

        test_data_path = f"model_output/{test_model}/{mode}_result/json/{role}.json"
        with open(test_data_path, 'r') as f:
            test_data = json.load(f)

        output_list = []
        for i in range(len(gt_data)):
            question = gt_data[i]['system_prompt'] + "There are multiple rounds of questions, divided by ###:\n" + "\n###\n".join(gt_data[i]['dialogue'][j]['user'] for j in range(len(gt_data[i]['dialogue'])))
            
            # 处理test_data的不同结构
            if 'dialogue' in test_data:
                # 旧格式：单个dialogue数组
                dialogue_key = 'dialogue'
                dialogue_data = test_data['dialogue']
            else:
                # 新格式：多个dialogue（dialogue_0, dialogue_1等）
                dialogue_key = f'dialogue_{i}'
                if dialogue_key not in test_data:
                    print(f"警告：在test_data中找不到{dialogue_key}")
                    continue
                dialogue_data = test_data[dialogue_key]
            
            base64_audio1 = [
                {
                    "type": "input_audio",
                    "input_audio": {
                    "data": "data:audio/wav;base64," + wav_to_base64(dialogue_data[j]['model_output']['audio_path']),
                    "format": "wav"
                    }
                }
                for j in range(len(gt_data[i]['dialogue']))
            ]
            base64_audio2 = [
                {
                    "type": "input_audio",
                    "input_audio": {
                    "data": "data:audio/wav;base64," + wav_to_base64(gt_data[i]['dialogue'][j]['role_speech_path']),
                    "format": "wav"
                    }
                }
                for j in range(len(gt_data[i]['dialogue']))
            ]

            output_list.append({'idx': i})
            for metric in tqdm(metrics):
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "text": f"## **[Question Start]**\n\n{question}\n\n## **[Question End]**\n\n\n## **[Model A's response start, with one answer audio in each round]**\n\n"
                        }]
                        + base64_audio1 +
                        [{
                            "type": "text",
                            "text": f"\n\n## **[Model A's Response End]**\n\n\n## **[Model B's response start, with one answer audio in each round]**\n\n"
                        }]
                        + base64_audio2 +
                        [{
                            "type": "text",
                            "text": f"\n\n## **[Model B's Response End]**\n\n\n## **[Instruction]**\n\nThe task instruction of the two models is to directly role-play as {role_name}.\n\nPlease evaluate the following aspect of each model's response:\n{metrics[metric]}\n\n" + "Please provide a brief qualitative evaluation for the relative performance of the two models, followed by paired quantitative scores from 1 to 10, where 1 indicates poor performance and 10 indicates excellent performance.\n\nThe output should be in the following format:\n{Qualitative Evaluation}, [Scores]: ({the score of Model A}, {the score of Model B})\n\nPlease ensure that your evaluations are unbiased and that the order in which the responses were presented does not affect your judgment."
                        }]
                    }
                ]
                # continue
                # print(messages)
                while True:
                    try:
                        response = client.chat.completions.create(
                            model=model,
                            n=1,    # 返回一个候选回答
                            messages=messages,
                            timeout=300,
                        )
                        raw_result = response.choices[0].message.content
                        parsed_result = parse_gemini_result(raw_result)
                        output_list[-1][f"{metric}_results"] = parsed_result
                        break
                    except Exception as e:
                        error_msg = str(e)
                        print(f"API调用错误: {error_msg}")
                        
                        # 检查是否是余额不足
                        if "该令牌额度已用尽" in error_msg:
                            print("检测到余额不足，尝试切换API key...")
                            try:
                                client = switch_to_next_key()
                                # 继续尝试，不增加sleep时间
                                continue
                            except Exception as switch_error:
                                print(f"切换API key失败: {switch_error}")
                                # 如果所有key都用完了，退出
                                if current_key_index == len(api_keys) - 1:
                                    print("所有API key都已用完，退出程序")
                                    raise e
                                else:
                                    # 继续尝试下一个key
                                    continue
                        else:
                            # 其他错误，等待后重试
                            traceback.print_exc()
                            time.sleep(10)
                            continue
                # break
        with open(save_path, 'w', encoding='utf-8') as json_file:
            json.dump(output_list, json_file, indent=4, ensure_ascii=False)
        # exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Gemini Judge for Audio LLM Evaluation')
    parser.add_argument('--mode', type=str, required=True,
                        help='Mode: multi or single')
    parser.add_argument('--test_model', type=str, required=True,
                        help='Test model: ali_cloud or cascade')
    
    args = parser.parse_args()
    main(args.mode, args.test_model)
