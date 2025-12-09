import json
from openai import OpenAI
import re
import time
import numpy as np
from scipy.spatial.distance import jensenshannon
from sentence_transformers import SentenceTransformer

client = OpenAI(
    api_key="sk-VZKKr3kd4p59mP3TGWF3dxdS9U9uWTbSNiSeSNWuV6QPLMyq",
    base_url="https://api.nuwaapi.com/v1"
)


def prompt_responses():
    prompt_response = []
    data_path = '../dataset/ultrafeedback_curriculum_dpo_pairs.json'
    with open(data_path, 'r', encoding='utf-8') as file:
        datas = json.load(file)
    for data in datas:
        responses = []
        prompt = data['easy']['conversation'][0]['content']
        responses.append(data['easy']['conversation'][1]['chosen_content'])
        for key, value in data.items():
            responses.append(value['conversation'][1]['rejected_content'])
        result = {'prompt': prompt, 'responses': responses}
        prompt_response.append(result)
    return prompt_response


def gpt_4_api(messages: list, max_retries: int = 3, delay: int = 5):
    """调用 GPT-4.1 API，带错误重试和正则抽取分数"""
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="gpt-4.1",
                messages=messages
            )
            content = completion.choices[0].message.content.strip()

            match = re.search(r'\[\[\s*(\d+(\.\d+)?)\s*\]\]', content)
            if match:
                number = float(match.group(1))
                print(f"匹配到的评分: {number}")
                return number
            else:
                print("⚠️ 未找到匹配数字，原始输出:", content)
                return 0.0

        except Exception as e:
            print(f"❌ 调用出错 (第 {attempt + 1} 次): {e}")
            if attempt < max_retries - 1:
                print(f"⏳ {delay} 秒后重试...")
                time.sleep(delay)
            else:
                print("🚨 已达最大重试次数，返回 0")
                return 0.0


def Calculate_PC():
    # 计算每个prompt的PC值
    with open('../dataset/ultrafeedback_curriculum_dpo_pairs.json', 'r', encoding='utf-8') as file:
        datas = json.load(file)
    result_scores = []
    for data in datas:
        prompt = data['easy']['conversation'][0]['content']
        messages = [
            {
                'role': 'system',
                'content': f"""You are an expert evaluator measuring the MODEL-UNDERSTANDING DIFFICULTY of a single English prompt. 
            Your task: Read the given prompt and produce a single numeric score between 1.0 (very easy for GPT-like models) and 10.0 (extremely difficult). 
            The score may include one decimal place. Output ONLY the final numeric score.

            Internal evaluation procedure (do not output these details):
            1. Read the prompt between <<<PROMPT>>> and <<<END>>>.
            2. For each of the following 5 dimensions, assign a sub-score (1.0–10.0):
               - Clarity of intent (25%): How clear and unambiguous the task is.
               - Reasoning depth (25%): How many reasoning steps or chains of logic are required.
               - Domain knowledge requirement (20%): How much specialized knowledge is needed.
               - Constraints & formatting (15%): How strict or numerous the instructions/format rules are.
               - Ambiguity / multiple interpretations (15%): Degree to which the prompt allows several different valid readings.
            3. Compute the weighted average of these sub-scores.
            4. Round to one decimal place.
            5. Output ONLY the final numeric score:[[rating]], please strictly keep this format.(e.g.,[[6.7]]).""",
            },
            {
                'role': 'user',
                'content': f"""
                        Input prompt:
                        <<<PROMPT>>>
                        {prompt}
                        <<<END>>>
                        """
            }, ]

        gpt_score = gpt_4_api(messages)
        result_score = {
            "prompt": prompt,
            "PC_score": gpt_score/10
        }
        result_scores.append(result_score)
    return result_scores


def PC_result():
    file_path = '../dataset/PC_gpt_results.json'
    with open(file_path, 'r', encoding='utf-8') as file:
        datas = json.load(file)
    return datas


def embedding_to_distribution(embedding, tau=1.0):
    # 数值稳定的 softmax
    emb = np.array(embedding) / tau
    emb = emb - np.max(emb)
    exp_vals = np.exp(emb)
    return exp_vals / np.sum(exp_vals)


def Calculte_PPD(embedding_model_path):
    prompt_response = prompt_responses()

    # 将prompt以及responses构建三元组
    triples = []
    for data in prompt_response:
        prompt = data['prompt']
        responses = data['responses']
        pairs = [(0, 1)]
        for pair in pairs:
            triples.append((prompt, responses[pair[0]], responses[pair[1]]))
    print("len(triples):", len(triples))

    model_path = embedding_model_path
    model = SentenceTransformer(model_path)
    results = []

    for num, triple in enumerate(triples):
        print("sample_number: ", num)

        # response -> embedding
        embedding1 = model.encode(triple[1], normalize_embeddings=False)
        embedding2 = model.encode(triple[2], normalize_embeddings=False)

        dist1 = embedding_to_distribution(embedding1)
        dist2 = embedding_to_distribution(embedding2)

        js_div = jensenshannon(dist1, dist2, base=2) ** 2
        print("JS_score:", js_div)

        results.append({
            'prompt': triple[0],
            'responses': [triple[1], triple[2]],
            'JS_score': js_div
        })

    return results


def PPD_Normalization(embedding_model_path):

    PPD_result = Calculte_PPD(embedding_model_path)
    scores = []
    min_score = 1
    for data in PPD_result:
        scores.append(data['JS_score'])
        if data['JS_score'] != 0 and data['JS_score'] < min_score:
            min_score = data['JS_score']

    # ϵ
    epsilon = min_score / 10
    scores = np.array(scores)

    results = []
    for data in PPD_result:
        ppd = (np.log(data['JS_score'] + epsilon) - (np.log(scores + epsilon)).min()) / (
                (np.log(scores + epsilon)).max() - (np.log(scores + epsilon)).min())
        result = {
            "prompt": data['prompt'],
            "responses": data['responses'],
            "normalization_score": ppd
        }
        results.append(result)

    return results


def linear_single_pair(ppd_datas, pc_datas, alpha):
    results = {}
    # 初始化
    for ppd_data in ppd_datas:
        prompt = ppd_data['prompt']
        results[prompt] = {}
        results[prompt]['pairs'] = [(0, 1)]
        results[prompt]['sft'] = []
        results[prompt]['difficulty'] = []
        results[prompt]['responses'] = []

    prompt_response = prompt_responses()

    for num, data in enumerate(ppd_datas):
        print("当前进程：", num)
        prompt = data['prompt']
        ppd_score = data['normalization_score']
        print("ppd_score: ", ppd_score)
        for response in prompt_response:
            if prompt == response['prompt']:
                results[prompt]['responses'] = response['responses']
                results[prompt]['sft'].append(response['responses'][0])
        for pc_data in pc_datas:
            if pc_data['prompt'] == prompt:
                pc_score = pc_data['PC_gpt_score']
                print("pc_score: ", pc_score)
                difficulty_score = alpha * ppd_score + (1 - alpha) * pc_score
                print("difficulty_score: ", difficulty_score)
                results[prompt]['difficulty'].append(difficulty_score)

    # 保存三元组样本以及对应的难度分数
    with open("../dataset/Single-Pair/ultrafeedback(alpha={}).json".format(alpha), 'w',
              encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    difficulty_score_list = []
    for prompt, data in results.items():
        for difficulty in data['difficulty']:
            difficulty_score_list.append(difficulty)

    difficulty_score_list = sorted(difficulty_score_list)

    with open("../dataset/Single-Pair/ultrafeedback_score_list(alpha={}).json".format(alpha), 'w',
              encoding='utf-8') as f:
        json.dump(difficulty_score_list, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    alpha = 0.5
    embedding_model_path = '../models/bge-m3'

    pc_datas = PC_result()
    ppd_datas = PPD_Normalization(embedding_model_path)
    linear_single_pair(ppd_datas, pc_datas, alpha)

