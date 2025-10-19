# DiffCurri-DPO: Difficulty-Aware Curriculum for Direct Preference Optimization

This is the open code of the paper ``DiffCurri-DPO: Difficulty-Aware Curriculum for Direct Preference Optimization''.

### Environment
```
conda create -n dpo_env python=3.12
conda activate dpo_env
pip install -r requirements.txt
```

### Model
在实验阶段，我们分别使用了Zephyr-7b-beta和Mistral-7B-Instruct-v0.3模型进行测试，其中采用bge-m3模型进行embedding转换。

### Data_Processing
在论文中，我们使用Multi-Dimensional CL对数据进行处理，其中我们设置了三种不同的课程学习训练方法：Single-Pair, Three-Pairs and All-Pairs. 下面是在训练模型前对数据进行的数据准备：
在运行下述代码前，需要对代码中的alpha进行初始化（0.0~1.0），保证PC和PPD之间构成的线性关系，
##### Single-Pair
```
python Single-Pair_linear.py
```
##### Three-Pairs
```
python Three-Pairs_linear.py
```
##### All-Pairs
```
python All-Pairs_linear.py
```
运行上述代码后，可以在dataset/Single-Pair、dataset/Three-Pairs、dataset/All-Pairs中检查数据是否处理成功。

### Train
上述的数据处理阶段完成后，接下来进行model训练阶段：
```
python -u train.py model=Mistral-7B-Instruct-v0.3 datasets=[curri_dpo] loss=dpo loss.beta=0.1 exp_name=Mistral-7B-Instruct-v0.3_dpo_train gradient_accumulation_steps=1 batch_size=8 eval_batch_size=1 trainer=BasicTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 optimizer=adamW alpha=0.5 curriculum_type=All-Pairs
```
其中model需要进行指定，如果使用Zephyr-7b-beta，则将上述指令设置为model=zephyr-7b-beta，与config中的配置保持一致。
请保证训练数据与上述指令的一致性，比如在数据处理中的alpha=0.6，那么上述指令需设置为alpha=0.6；在数据处理中采用Single-Pair的形式，那么上述的指令也需设置为curriculum_type=Single-Pair。

### Evaluation
在论文中，我们采用了MT-Bench、SHP-2、WizardLM和UltraFeedback作为测试数据，并采用gpt-4.1进行评估。
其中，MT-Bench的prompt如下：
```
\textbf{[System]}\\
    As an impartial judge, you need to evaluate the overall performance of the AI assistant in the following two rounds of dialogue. Please analyze according to the following formatting criteria.
    \textbf{[Evaluation Dimensions]}Contextual coherence: whether the second round of answers effectively takes into account the content of the first round, and whether the dialogue logic is natural and smooth. Comprehensive quality: aspects include practicality, accuracy, depth, and creativity. Gradual improvement: whether the second round of answers discusses details in depth based on the first round.
    \textbf{[Evaluation Process]}Briefly analyze, point out the good results of the two rounds of answers and finally give an overall score. The score range is 1-10, Keep two decimal places.
    \textbf{[Output Format]}Comments: [Analysis Content], Comprehensive Score: [[rating]], please strictly keep this format. For example: "Rating: [[5.00]]"
    
    \medskip
    \textbf{[Question1]}{question1}
    
    \textbf{[Answer to Question1]}\\
    {answer1}
    
    \textbf{[Question2]}{question2}
    
    \textbf{[Answer to Question2]}\\
    {answer2}
    
    \textbf{[The End of dialogue]}
```
其余具体细节请参考论文``DiffCurri-DPO: Difficulty-Aware Curriculum for Direct Preference Optimization''。


