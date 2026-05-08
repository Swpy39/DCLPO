import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import json
from sentence_transformers import SentenceTransformer

def build_anchors(embeddings: torch.Tensor, K=128, seed=42):
    assert embeddings.dim() == 2, "embeddings are (N, D)"
    device = embeddings.device
    print("k:",K)
    embeddings = F.normalize(embeddings, p=2, dim=1).cpu()

    kmeans = MiniBatchKMeans(n_clusters=K, batch_size=2048, random_state=seed)
    kmeans.fit(embeddings.numpy())

    anchors = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    anchors = F.normalize(anchors, p=2, dim=1).to(device)
    return anchors

def get_anchor_distribution(v: torch.Tensor, anchors: torch.Tensor, tau=0.07):
    v = F.normalize(v, p=2, dim=1)
    anchors = F.normalize(anchors, p=2, dim=1)
    sim = torch.matmul(v, anchors.T) / tau
    P = F.softmax(sim, dim=-1)
    return P

def js_divergence(P: torch.Tensor, Q: torch.Tensor, eps=1e-12):
    M = 0.5 * (P + Q)
    P_log = torch.log(P + eps)
    Q_log = torch.log(Q + eps)
    M_log = torch.log(M + eps)
    kl_pm = torch.sum(P * (P_log - M_log), dim=-1)
    kl_qm = torch.sum(Q * (Q_log - M_log), dim=-1)
    js = 0.5 * (kl_pm + kl_qm)
    return js

def compute_S_div(v_plus, v_minus, anchors, tau=0.07, eps=1e-12):
    if v_plus.dim() == 1:
        v_plus = v_plus.unsqueeze(0)
    if v_minus.dim() == 1:
        v_minus = v_minus.unsqueeze(0)

    P_plus = get_anchor_distribution(v_plus, anchors, tau=tau)
    P_minus = get_anchor_distribution(v_minus, anchors, tau=tau)
    D_js = js_divergence(P_plus, P_minus, eps=eps)

    return D_js.item()

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


def compute_S_cos(embeddings_1: torch.Tensor, embeddings_2: torch.Tensor):
    if embeddings_1.dim() == 1:
        embeddings_1 = embeddings_1.unsqueeze(0)
    if embeddings_2.dim() == 1:
        embeddings_2 = embeddings_2.unsqueeze(0)

    embeddings_1 = F.normalize(embeddings_1, p=2, dim=1).squeeze()
    embeddings_2 = F.normalize(embeddings_2, p=2, dim=1).squeeze()
    cos_sim = torch.dot(embeddings_1,embeddings_2).item()
    return cos_sim

def PC_result():
    file_path = '../dataset/PC_gpt_results.json'
    with open(file_path, 'r', encoding='utf-8') as file:
        datas = json.load(file)
    return datas

if __name__ == "__main__":
    w1=1.0
    w2=0.0

    torch.manual_seed(0)
    embedding_model_path = '/hpc2hdd/home/fye374/models/bge-m3'
    model = SentenceTransformer(embedding_model_path)

    prompt_data = prompt_responses()

    all_responses = []
    for item in prompt_data:
        all_responses.extend(item["responses"])

    embeddings = model.encode(all_responses, convert_to_tensor=True, normalize_embeddings=True)
    print("Embeddings shape:", embeddings.shape)

    anchors = build_anchors(embeddings, K=128)
    print("Anchors shape:", anchors.shape)

    results = {}
    global_idx = 0

    for item in prompt_data:
        prompt = item['prompt']
        results[prompt] = {}
        results[prompt]['pairs'] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        results[prompt]['sft'] = item['responses'][0]
        results[prompt]['difficulty'] = []
        results[prompt]['responses'] = item['responses']

    JS= []
    min_js_score = 1
    for item in prompt_data:
        prompt = item['prompt']
        n = len(item["responses"])
        local_embeddings = embeddings[global_idx:global_idx + n]
        global_idx += n

        for i in range(n):
            for j in range(i + 1, n):
                D_js = compute_S_div(local_embeddings[i], local_embeddings[j], anchors, tau=0.07)
                if D_js < min_js_score and D_js > 0:
                    min_js_score = D_js
                JS.append(D_js)

    min_js_score = min(JS)
    max_js_score = max(JS)
    print("min_js_score:", min_js_score)
    print("max_js_score:", max_js_score)
    
    epsilon = min_js_score / 10
    if epsilon < 0:
        epsilon=0
    global_idx = 0

    pc_datas = PC_result()

    for item in prompt_data:
        prompt = item['prompt']
        n = len(item["responses"])
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        local_embeddings = embeddings[global_idx:global_idx + n]
        global_idx += n

        for pair in pairs:
            D_js = compute_S_div(local_embeddings[pair[0]], local_embeddings[pair[1]], anchors, tau=0.07)
            S_js = (D_js - min_js_score)/(max_js_score - min_js_score)
            
            for pc_data in pc_datas:
                if pc_data['prompt'] == prompt:
                    pc_score = pc_data['Score']
                    break

            difficulty_score = float(w1 * pc_score + w2 * S_js)
            
            print("difficulty_score: ", difficulty_score)

            results[prompt]['difficulty'].append(difficulty_score)

    print(f"include {len(results)} Responses-JS")

    with open("../dataset/All-Pairs/ultrafeedback(w1={},w2={}).json".format(w1,w2), 'w',encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print("save ../dataset/All-Pairs/ultrafeedback(w1={},w2={}).json".format(w1,w2))

    difficulty_score_list = []
    for prompt, data in results.items():
        for difficulty in data['difficulty']:
            difficulty_score_list.append(difficulty)
    
    difficulty_score_list = sorted(difficulty_score_list)

    with open("../dataset/All-Pairs/ultrafeedback_score_list(w1={},w2={}).json".format(w1,w2), 'w',encoding='utf-8') as f:
        json.dump(difficulty_score_list, f, ensure_ascii=False, indent=4)

    



