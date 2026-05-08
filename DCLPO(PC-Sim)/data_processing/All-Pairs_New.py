import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import json
from sentence_transformers import SentenceTransformer


def PC_result():
    file_path = '../dataset/PC_gpt_results.json'
    with open(file_path, 'r', encoding='utf-8') as file:
        datas = json.load(file)
    return datas


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


def compute_S_cos(embedding_model_path, prompt_datas):
    model = SentenceTransformer(embedding_model_path)
    sim_datas = []
    # 收集responses用于编码
    for item in prompt_datas:
        prompt = item["prompt"]
        responses = item["responses"]
        pairs = [(0, 1),(0, 2),(0, 3),(1, 2),(1, 3),(2, 3)]
        for pair in pairs:
            embeddings_1 = model.encode(responses[pair[0]], convert_to_tensor=True, normalize_embeddings=True)
            embeddings_2 = model.encode(responses[pair[1]], convert_to_tensor=True, normalize_embeddings=True)

            if embeddings_1.dim() == 1:
                embeddings_1 = embeddings_1.unsqueeze(0)
            if embeddings_2.dim() == 1:
                embeddings_2 = embeddings_2.unsqueeze(0)

            embeddings_1 = F.normalize(embeddings_1, p=2, dim=1).squeeze()
            embeddings_2 = F.normalize(embeddings_2, p=2, dim=1).squeeze()
            cos_sim = torch.dot(embeddings_1,embeddings_2).item()

            sim_data = {
                "prompt": prompt,
                "response1": responses[pair[0]],
                "response2": responses[pair[1]],
                "sim_score": cos_sim
            }
            sim_datas.append(sim_data)
    print("len(sim_datas): ", len(sim_datas))
    return sim_datas


def main(sim_datas, pc_datas, w1, w2):
    results = {}
    # 初始化
    for sim_data in sim_datas:
        prompt = sim_data['prompt']
        results[prompt] = {}
        results[prompt]['pairs'] = [(0, 1),(0, 2),(0, 3),(1, 2),(1, 3),(2, 3)]
        results[prompt]['sft'] = []
        results[prompt]['difficulty'] = []
        results[prompt]['responses'] = []

    prompt_response = prompt_responses()

    for num, data in enumerate(sim_datas):
        print("当前进程：", num)
        prompt = data['prompt']
        sim_score = data['sim_score']
        print("sim_score: ", sim_score)
        for response in prompt_response:
            if prompt == response['prompt']:
                results[prompt]['responses'] = response['responses']
                results[prompt]['sft'] = response['responses'][0]
        for pc_data in pc_datas:
            if pc_data['prompt'] == prompt:
                pc_score = pc_data['PC_gpt_score']
                print("pc_score: ", pc_score)
                difficulty_score = w1 * pc_score + w2 * sim_score
                print("difficulty_score: ", difficulty_score)
                results[prompt]['difficulty'].append(difficulty_score)

    # 保存三元组样本以及对应的难度分数
    with open("../dataset/All-Pairs/ultrafeedback(w1={},w2={}).json".format(w1,w2), 'w',
              encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    difficulty_score_list = []
    for prompt, data in results.items():
        for difficulty in data['difficulty']:
            difficulty_score_list.append(difficulty)
    
    # 對文件進行中的分數進行排序
    difficulty_score_list = sorted(difficulty_score_list)

    with open("../dataset/All-Pairs/ultrafeedback_score_list(w1={},w2={}).json".format(w1,w2), 'w',
              encoding='utf-8') as f:
        json.dump(difficulty_score_list, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    w1=0.7
    w2=0.3
    w3=0

    print("w1: ", w1)
    print("w2: ", w2)

    embedding_model_path = '/hpc2hdd/home/fye374/models/bge-m3'  ## m3e-base(768维）//////bge-m3（1024维） //////bce-embedding-base_v1（768维）

    pc_datas = PC_result()
    prompt_datas = []
    prompt_datas = prompt_responses()
    sim_datas = compute_S_cos(embedding_model_path, prompt_datas)
    main(sim_datas, pc_datas, w1, w2)

    



