#!/bin/bash

# Gemini Judge 批量运行脚本
# 并行运行不同的 mode 和 test_model 组合

# 设置工作目录
cd benchmark

# 创建日志目录
mkdir -p logs

# 定义配置组合
modes=("multi" "single")
test_models=("example_model")

# 记录开始时间
echo "开始运行 Gemini Judge 评估..."
echo "开始时间: $(date)"
echo "=================================="

# 存储后台进程的PID
pids=()

# 遍历所有配置组合，启动并行进程
for mode in "${modes[@]}"; do
    for test_model in "${test_models[@]}"; do
        echo "启动进程: mode=$mode, test_model=$test_model"
        
        # 创建日志文件名
        log_file="logs/${test_model}_${mode}_gemini_judge.log"
        
        # 在后台运行 Python 脚本，输出到对应的日志文件
        (
            echo "=== 开始运行: mode=$mode, test_model=$test_model ==="
            echo "开始时间: $(date)"
            echo "=================================="
            
            python gemini_judge.py --mode "$mode" --test_model "$test_model"
            
            exit_code=$?
            echo "=================================="
            echo "结束时间: $(date)"
            
            if [ $exit_code -eq 0 ]; then
                echo "✅ 成功完成: mode=$mode, test_model=$test_model"
            else
                echo "❌ 运行失败: mode=$mode, test_model=$test_model (退出码: $exit_code)"
            fi
        ) > "$log_file" 2>&1 &
        
        # 保存进程PID
        pids+=($!)
        
        echo "进程已启动，PID: $!, 日志文件: $log_file"
        echo "----------------------------------"
        
        # 短暂延迟，避免同时启动太多进程
        sleep 20
    done
done

echo "所有进程已启动，等待完成..."
echo "=================================="

# 等待所有后台进程完成
for pid in "${pids[@]}"; do
    echo "等待进程 PID: $pid 完成..."
    wait $pid
    if [ $? -eq 0 ]; then
        echo "✅ 进程 PID: $pid 成功完成"
    else
        echo "❌ 进程 PID: $pid 失败"
    fi
done

# 记录结束时间
echo "=================================="
echo "所有任务完成！"
echo "结束时间: $(date)"
echo "日志文件位置: logs/"
