# Modeling Structural Difficulty in Preference Data for Large Language Model Alignment

This is the open code of the paper ``Modeling Structural Difficulty in Preference Data for Large Language Model Alignment''.

### Environment
```
conda create -n dpo_env python=3.12
conda activate dpo_env
pip install -r requirements.txt
```

### Dataset
We utilized the open-source dataset [ServiceNow-AI/Curriculum_DPO_preferences](https://huggingface.co/datasets/ServiceNow-AI/Curriculum_DPO_preferences) and [nvidia/HelpSteer](https://huggingface.co/datasets/nvidia/HelpSteer) in this project to support model training and evaluation. We would like to acknowledge and thank the authors and contributors of this dataset for their valuable work and contributions to the open-source community.

### Model
During the experimental phase, we conducted evaluations using the [HuggingFaceH4/zephyr-7b-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) and [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) models. The [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) model was utilized for embedding conversion.
>All the models employed in this project are open-source, and we sincerely appreciate the efforts of their respective developers and contributors to the open-source community.

### Data_Processing
In this paper, we employed the Multi-Dimensional Curriculum Learning (Multi-Dimensional CL) approach for data processing.
We designed three distinct curriculum learning strategies: Single-Pair, N-Pairs, and All-Pairs.
Before training the model, the following data preparation steps are required:
Prior to running the code, the α (alpha) parameter should be initialized (ranging from 0.0 to 1.0) to ensure a proper linear relationship between PC and PPD.
##### Single-Pair
```
python Single-Pair.py
```
##### N-Pairs
```
python N-Pairs.py
```
##### All-Pairs
```
python All-Pairs.py
```
After executing the code, you can verify whether the data has been successfully processed in the following directories:
- dataset/Single-Pair
- dataset/N-Pairs
- dataset/All-Pairs

### Train
After completing the data processing phase, the next step is the model training stage.
```
python -u train.py model=Mistral-7B-Instruct-v0.3 datasets=[curri_dpo] loss=dpo loss.beta=0.1 exp_name=Mistral-7B-Instruct-v0.3_dpo_train gradient_accumulation_steps=1 batch_size=8 eval_batch_size=1 trainer=BasicTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 optimizer=adamW alpha=0.5 curriculum_type=All-Pairs
```
In this stage, the model type must be explicitly specified. For example, when using the Zephyr-7b-beta model, set the corresponding command parameter as follows:
```
model=zephyr-7b-beta
exp_name=zephyr-7b-beta_dpo_train
```
This parameter should remain consistent with the configuration defined in the `config` file.
Furthermore, ensure that all training parameters are aligned with those used during data processing. For instance:
- If `alpha=0.6` was used in data processing, the same value should be set in the training command (`alpha=0.6`).
- If the `Single-Pair` curriculum learning strategy was applied during data processing, set `curriculum_type=Single-Pair` in the training command as well.

Maintaining consistency between data preparation and training configurations is essential for ensuring model stability and reproducibility.


### Evaluation
In this paper, we used MT-Bench, SHP-2, WizardLM, and UltraFeedback as evaluation datasets.
The evaluation process was conducted using GPT-4.1 as the scoring and analysis model.

The prompt used in MT-Bench is shown below:
```
[System]
    As an impartial judge, you need to evaluate the overall performance of the AI assistant in the following two rounds of dialogue. Please analyze according to the following formatting criteria.
    [Evaluation Dimensions]Contextual coherence: whether the second round of answers effectively takes into account the content of the first round, and whether the dialogue logic is natural and smooth. Comprehensive quality: aspects include practicality, accuracy, depth, and creativity. Gradual improvement: whether the second round of answers discusses details in depth based on the first round.
    [Evaluation Process]Briefly analyze, point out the good results of the two rounds of answers and finally give an overall score. The score range is 1-10, Keep two decimal places.
    [Output Format]Comments: [Analysis Content], Comprehensive Score: [[rating]], please strictly keep this format. For example: "Rating: [[5.00]]"
    
    [Question1]
    {question1}
    
    [Answer to Question1]
    {answer1}
    
    [Question2]
    {question2}
    
    [Answer to Question2]
    {answer2}
    
    [The End of dialogue]
```
For more details on the experimental setup and implementation, please refer to the paper "Modeling Structural Difficulty in Preference Data for Large Language Model Alignment".





