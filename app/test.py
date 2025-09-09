#!/usr/bin/env python3
import asyncio
import aiohttp
import json
import time
from datetime import datetime

# 学者ID列表
SCHOLAR_IDS = ["11822654", "11822655", "11822656", "11822680", "11822681"]
BASE_URL = "http://localhost:8009/chat_stream"

async def process_stream(response, scholar_id):
    """处理SSE流响应"""
    full_answer = ""
    start_time = time.time()
    
    # 读取SSE响应流
    async for line in response.content:
        line = line.decode('utf-8').strip()
        if not line or not line.startswith('data: '):
            continue
        
        # 解析SSE数据
        data = line[6:]  # 移除 'data: ' 前缀
        try:
            event_data = json.loads(data)
            event_type = event_data.get("event")
            
            if event_type == "data":
                # 累积部分回答
                token = event_data.get("data", {}).get("answer", "")
                full_answer += token
                
            elif event_type == "end":
                # 完成事件，包含完整回答
                complete_answer = event_data.get("data", {}).get("answer", "")
                if complete_answer and not full_answer:
                    full_answer = complete_answer
                break
                
            elif event_type == "error":
                # 错误事件
                error_msg = event_data.get("data", {}).get("error", "未知错误")
                return f"错误: {error_msg}", time.time() - start_time
                
        except json.JSONDecodeError:
            print(f"无法解析JSON: {data}")
            continue
            
    return full_answer, time.time() - start_time

async def test_scholar(session, scholar_id):
    """测试单个学者的响应"""
    print(f"开始请求学者 {scholar_id}...")
    start_time = time.time()
    
    try:
        async with session.post(
            BASE_URL,
            json={
                "scholar_open_id": scholar_id,
                "message": "你是谁？介绍一下你自己",
                "history": []
            },
            headers={
                "Accept": "text/event-stream",
                "Content-Type": "application/json"
            }
        ) as response:
            if response.status != 200:
                return f"请求失败，状态码: {response.status}", time.time() - start_time
            
            answer, process_time = await process_stream(response, scholar_id)
            return answer, process_time
            
    except Exception as e:
        return f"请求异常: {str(e)}", time.time() - start_time

async def main():
    """主函数：并发测试多个学者"""
    print(f"开始并发测试 {len(SCHOLAR_IDS)} 个学者的响应...\n")
    
    async with aiohttp.ClientSession() as session:
        tasks = [test_scholar(session, scholar_id) for scholar_id in SCHOLAR_IDS]
        results = await asyncio.gather(*tasks)
        
        total_time = 0
        
        # 打印结果
        for i, scholar_id in enumerate(SCHOLAR_IDS):
            answer, process_time = results[i]
            total_time += process_time
            
            print("=" * 80)
            print(f"学者ID: {scholar_id}")
            print(f"用时: {process_time:.2f}秒")
            print("-" * 40)
            
            # 显示回答，如果太长则截断
            if len(answer) > 1000:
                print(f"{answer[:1000]}...(已截断，共{len(answer)}字符)")
            else:
                print(answer)
            print("=" * 80)
            print()
        
        avg_time = total_time / len(SCHOLAR_IDS)
        print(f"测试完成，平均响应时间: {avg_time:.2f}秒")

if __name__ == "__main__":
    start = time.time()
    print(f"测试开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 运行异步主函数
    asyncio.run(main())
    
    print(f"测试结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总测试时间: {time.time() - start:.2f}秒")