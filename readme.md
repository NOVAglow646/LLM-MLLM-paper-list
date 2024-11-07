* [LLMs/Large Multi-modal models](#llmslarge-multi-modal-models) 
* [Prompt Learning](#prompt-learning)
* ⭐[In-Context Learning](#in-context-learning)
* [ICL Theories](#icl-theories)



## LLMs (Large language models)

### 2024

1. **Model Editing with Canonical Examples** [[paper]](http://arxiv.org/abs/2402.06155) 提出了一个新任务：让模型学习几个特定的文本例子，以实现某些纠正，同时还不能让模型改变很多。
1. **Evaluating Large Language Models at Evaluating Instruction Following** [[paper]](https://openreview.net/forum?id=tr0KidwPLc) (ICLR 2024) 
1. **Not all Layers of LLMs are Necessary during Inference** (Arxiv April 2024) 训练一个对LLM中间层feature的分类器判断是否应该早停来获取早停层数，来加速LLM推理。还发现中间层预测的top prob和top prob-second top prob在各个任务上都呈现出随着层数加深而增加并逐渐稳定的趋势（但在不同任务上层数不一样）。[[paper]](http://arxiv.org/abs/2403.02181)
1. **Demonstrating Mutual Reinforcement Effect through Information Flow** (Arxiv March 2024) [[paper]](https://arxiv.org/pdf/2403.02902) 研究了同时进行word分类和text分类的MRE（Mutual Reinforcement Effect）任务，也观察到了anchor那篇中的三种attention activation随layer的分布趋势。
1. **A Theoretical Understanding of Self-Correction through In-context Alignment** (Arxiv May 2024) [[paper]](http://arxiv.org/abs/2405.18634) 理论分析transformer中的各个模块在self-correction中发挥的作用
1. **Mechanics of Next Token Prediction with Self-Attention** (AISTATS 2024) [[paper]](https://proceedings.mlr.press/v238/li24f.html) 构造了一个graph来描述next token prediction任务，在简化setting下理论分析出last token更倾向于给更经常作为label的token分配更高的attention。
1. **The pitfalls of next-token prediction** (Arxiv April 2024) [[paper]](http://arxiv.org/abs/2403.06963) 指出了自回归模型的缺陷：错误滚雪球效应和在一个单一token路径上只能学出一个类似induction head的shortcut模型
1. **A Law of Next-Token Prediction in Large Language Models** (Arxiv August 2024) [[paper]](https://arxiv.org/pdf/2408.13442v1)
1. **The Unlocking Spell on Base LLMs: Rethinking Alignment via In-Context Learning** (ICLR 2024) [[paper]](https://openreview.net/forum?id=wxJ0eXwwda) 通过ICL，添加system prompt和风格化的输出，实现只用很少的样本（3个）来提升LLM alignment。

### 2023

1. **Instruction-following Evaluation through Verbalizer Manipulation** (Arxiv July 2023) [[paper]](http://arxiv.org/abs/2307.10558) 发现LLM遵循flipped-label instructions的能力很差，说明ICL可能只是直接利用了预训练语料的知识，而不是学习了context。即使是强如GPT-4的模型也不能很好地遵循flipped-label instructions。
2. **Reasoning or Reciting? Exploring the Capabilities and Limitations of Language Models Through Counterfactual Tasks** (Arxiv Aug 2023) [[paper]](http://arxiv.org/abs/2307.02477) 一些主要发现：①模型在counterfactual的setting中性能会变差，且setting和常见的、符合事实的setting相差越远，性能越差，说明了模型可能的记忆现象。②在算术任务上，ICL能提升counterfactual（不同进制的计算）性能，但和default setting的差距难以抹平。
3. **Can the Inference Logic of Large Language Models be Disentangled into Symbolic Concepts?** (Arxiv Apr 2023) [[paper]](https://arxiv.org/abs/2304.01083) 提出了一种empirical的指标来衡量输入句子里的某些词和词组对某一特定输出的决定程度。
4. **Contrastive Chain-of-Thought Prompting** (Arxiv Nov 2023) [[paper]](http://arxiv.org/abs/2311.09277) 使用对比CoT，即一个正确CoT搭配一个错误CoT能相比常规的CoT带来提升.
5. 

### 2022

1. **Same Pre-training Loss, Better Downstream: Implicit Bias Matters for Language Models** [[paper]](https://arxiv.org/pdf/2210.14199.pdf)

2. 

### 2021

**LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS** 将对模型权重矩阵的更新限制为低秩矩阵乘积$BA$的形式，极大减少了pre-trained model迁移到新任务的代价（不用fine-tune所有参数） [[paper]](https://arxiv.org/abs/2106.09685)

### 2019

1. **Are Sixteen Heads Really Better than One?** (NeurIPS 2019) [[paper]](https://proceedings.neurips.cc/paper_files/paper/2019/hash/2c601ad9d2ff9bc8b282670cdd54f69f-Abstract.html) 在某些层上，只用一个head性能也能保持不变。同时提出了使用attention梯度来衡量head的重要性，提出了剪枝策略。



## MLLMs (Multimodal LLMs)

### 2024

1. **Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation** (Arxiv Oct 2024) [[paper]](http://arxiv.org/abs/2410.13848) 用一个自回归transformer统一实现多模态的理解和生成任务
2. **DOES SPATIAL COGNITION EMERGE IN FRONTIER MODELS? **(Arxiv Oct 2024) [[paper]](http://arxiv.org/abs/2410.06468) 空间理解任务 SPACE benchmark
3. **Task Navigator: Decomposing Complex Tasks for Multimodal Large Language Models** (CVPR 2024) [[paper]](https://openaccess.thecvf.com/content/CVPR2024W/MAR/papers/Ma_Task_Navigator_Decomposing_Complex_Tasks_for_Multimodal_Large_Language_Models_CVPRW_2024_paper.pdf) 工程文章，借助LLM分解任务，提升MLLM完成复杂视觉理解任务的能力
4. **Decomposing Complex Visual Comprehension into Atomic Visual Skills for Vision Language Models** [[paper]](https://openreview.net/pdf?id=nFU4xCyoe0) 原子视觉任务benchmark Atomic Visual Skills Benchmark (AVSBench) 



## VLMs (Vision-language models)

### 2024

1. **VisionLLaMA: A Unified LLaMA Interface for Vision Tasks** (Arxiv Mar 2024) [[paper]](https://arxiv.org/pdf/2403.00522) Vision LLaMa
2. **Is A Picture Worth A Thousand Words? Delving Into Spatial Reasoning for Vision Language Models** (NeurIPS 2024) [[paper]](http://arxiv.org/abs/2406.14852) 
   在三个合成的空间理解任务上评测LLM和LVM，主要发现：1）该任务的总体表现并不好 2）对于VLM而言，更依赖于语言信息而不是视觉信息做决策，去掉/扰乱视觉信息甚至会有提升 3）VLM中的language encoder比同样的单独LLM性能更好，说明多模态pretrain对于language有用
3. **Can Vision Language Models Learn from Visual Demonstrations of Ambiguous Spatial Reasoning?** (Arxiv Sep 2024) [[paper]](https://arxiv.org/abs/2406.02537) 
4. **TOPVIEWRS: Vision-Language Models as Top-View Spatial Reasoners** (Arxiv June 2024) [[paper]](http://arxiv.org/abs/2406.02537) 提了一个新的俯视图理解的数据集，发现VLM的俯视图理解能力仍然很差
5. **Emu3: Next-Token Prediction is All You Need** (Arxiv September 2024) [[paper]](http://arxiv.org/abs/2409.18869) 将文本、图片、视频都转化为token，进行next-token prediction的预训练。能同时做图片视频的生成、视觉-语言理解。

### 2022

1. **Data Determines Distributional Robustness in Contrastive Language Image Pre-training (CLIP)** (Arxiv 2022) [[paper]](https://arxiv.org/abs/2205.01397)
2. 





## Prompt Learning

### Prompt learning：

1. **Conditional Prompt Learning for Vision-Language Models** (CoCoOp, CVPR2022) 将图片特征直接加到context token上，获得sample-wise的prompt，以实现instance的generalization。其实就是希望通过引入图像信息来使得prompt描述得更贴切。不过感觉还是有点怪，因为所有class都加上了同样的可学习prefix，为什么能提高预测为正确类的概率？
2. MaPLe: Multi-modal Prompt Learning, CVPR2023 
3. Prompt-aligned Gradient for Prompt Tuning, ICCV2023
4. Compound Text-Guided Prompt Tuning via Image-Adaptive Cues, AAAI2024
5. MmAP : Multi-modal Alignment Prompt for Cross-domain Multi-task Learning, AAAI2024
6. **Improving Zero-Shot Generalization for CLIP with Synthesized Prompts** (ICCV 2023)

### For DA:

1. Domain Adaptation via Prompt Learning, arxiv 2022
2. AD-CLIP: Adapting Domains in Prompt Space Using CLIP, ICCV2023
3. Multi-Prompt Alignment for Multi-Source Unsupervised Domain Adaptation, NIPS2023
4. Prompt-based Distribution Alignment for Unsupervised Domain Adaptation, AAAI2024

### For DG:

1. StyLIP: Multi-Scale Style-Conditioned Prompt Learning for CLIP-based Domain Generalization, arxiv2023





## In-Context Learning

### 2024

1. **Explore Spurious Correlations at the Concept Level in Language Models for Text Classification** (Arxiv Jan 2024) [[paper]](http://arxiv.org/abs/2311.08648) 发现了LLM在文本分类中会依赖的concept-label spurious correlation，提出使用ChatGPT来扩充数据来消除虚假关联。

2. **Positional Information Matters for Invariant In-Context Learning: A Case Study of Simple Function Classes** (ongoing work) [[paper]](Positional Information Matters for Invariant In-Context Learning: A Case Study of Simple Function Classes) 发现模型对于demonstration的permutation invariance或许是ICL OOD的关键。提出使用相同的positional encoding来提升ICL OOD性能。

3. **Simple synthetic data reduces sycophancy in large language models** (Arxiv Feb 2024) [[paper]](http://arxiv.org/abs/2308.03958) LLMs会迎合提问者的观点而罔顾事实。提出合成一些用户的观点和正确性无关的新prompt，然后在这些数据上fine-tune来解决sycophancy问题。

4. **Understanding In-Context Learning in Transformers and LLMs by Learning to Learn Discrete Functions** (ICLR 2024 Oral) [[paper]](https://arxiv.org/abs/2310.03016) 探究transformer在一系列离散任务上的能力。特别地，发现经过预训练的模型相比随机初始化的模型获得了更强的最近邻、disjunction和conjunction的能力。

5. **Batch-ICL: Effective, Efficient, and Order-Agnostic In-Context Learning**  (Arxiv Jan 2024) 发现使用batch ICL，将N个example设置为N个one-shot inference，再把每个inference得到的token做平均，替换到query sample做aggregation最终再预测能带来提升。一个奇特的发现是做aggregation时从某一层往后做性能会突增，在那之前性能接近零。对此解释是transformer的低层是在学语义信息。

6. **RefuteBench: Evaluating Refuting Instruction-Following for Large Language Models** (Arxiv Feb 2024) [[paper]](http://arxiv.org/abs/2402.13463) 评估模型的改变它们的原始输出并遵循和一开始相违背的指令的能力。主要观察：1)大部分模型都会倾向于遵守它们的预训练知识 2)模型很难根据人类后续的反馈泛化到新的问题 3)所有模型都会逐步忘记人类反馈并落回到它们的内部知识里 4)模型是不是第一时间遵守了人类的反馈，对于后续的行为起到关键作用

7. **Function Vectors in Large Language Models** (ICLR 2024) [[paper]](http://arxiv.org/abs/2310.15213) 发现context prompt的最后一个token的隐层表示encode了这个任务的信息，称为function vector（FV）。将其加到zero-shot的prompt上，发现有显著提升。

8. **A Data Generation Perspective to the Mechanism of In-Context Learning** (Arxiv Feb 2024) [[paper]](http://arxiv.org/abs/2402.02212) 有关task recognition和task learning的综述

9. **Identifying and Analyzing Task-Encoding Tokens in Large Language Models** (Arxiv Feb 2024) [[paper]](http://arxiv.org/abs/2401.11323) 探究了context中的template词（"data:","answer:"）/stopword（标点、连词等无意义词）/content对performance的意义。结果发现template词对ICL性能提升最有用，content反而没什么用；还探究了template词的什么特征使得它有别于context中的其他成分，结果发现template词本身的语义、其重复性、其分隔x和y的格式作用这三者都对ICL性能有显著的作用。

10. **Whispers that Shake Foundations: Analyzing and Mitigating False Premise Hallucinations in Large Language Models** (Arxiv Feb 2024) [[paper]](http://arxiv.org/abs/2402.19103) 发现，问题中的错误前提而导致的回答中的幻觉是由于模型中特定的head的激活所引起的。提出了一种强行消除这些head对于问题中的错误前提对应的token的attention的方法。

11. **In-context Vectors: Making In Context Learning More Effective and Controllable Through Latent Space Steering** (Arxiv Feb 2024) [[paper]](https://arxiv.org/abs/2311.06668) 提出用context的第L层表示构造一个表征任务信息的vector（ICV），然后再加到query时的第L层所有token的表示上。

12. **The mechanistic basis of data dependence and abrupt learning in an in-context classification task** (ICLR 2024 Oral) [[paper]](https://arxiv.org/abs/2312.03002) 有关transformer 的IWL（in-weights learning）和ICL学习过程的实验性分析。在一个两层toy transformer中揭示了induction head学习机制。

13. **Understanding In-context Learning From Repetitions** (ICLR 2024) [[paper]](https://openreview.net/forum?id=bGGYcvw8mp) 揭示了context中重复出现的pattern会导致模型更倾向于输出这个pattern的现象。

14. **In-context Learning Learns Label Relationships but is not Conventional Learning** (ICLR 2024) [[paper]](https://openreview.net/pdf?id=YPIA7bgd5y) 以更大的模型和更长的context重新审视以往的ICL讨论，并得出了以下三个结论：1)ICL会学x-y映射，正确的label是有用的，且模型越大这一效应越明显 2)ICL能学预训练时没见过的新任务 3)即使context很长，ICL也不能彻底覆盖预训练获得的preference 4)LLM更关注更靠近query的example

15. **How do Large Language Models Learn In-Context? Query and Key Matrices of In-Context Heads are Two Towers for Metric Learning** (Arxiv Feb 2024) [[paper]](http://arxiv.org/abs/2402.02872) 在简单的word classification任务上，首先按照类似Function Vector的做法，提取出对输出正确预测贡献最大的head。然后分析这些head并发现了如下机制：label的V encode了label的特征，label的K encode了demonstration的特征；last token的Q encode了query的特征；last token query和正确label的K的attention score比其他head的显著大；last token Q与在context中出现更多的label/更靠近query的label的K的attention score更大。

16. **Locating Factual Knowledge in Large Language Models: Exploring the Residual Stream and Analyzing Subvalues in Vocabulary Space** (Arxiv Jan 2024) [[paper]](http://arxiv.org/abs/2312.12141) 提出了一种定位transformer中对输出某一label贡献最大的attention或FFN layer（或其subvalue）的方法。

17. **In-Context Learning State Vector with Inner and Momentum Optimization** (NeurIPS 2024) [[paper]](http://arxiv.org/abs/2404.11225) 提了一种新的用vector压缩信息的技术（State Vector SV）：是将前L层的每层的attention输出concat起来。然后提了三种技术（aggregate每一个example的SV、用momentum、分组提取SV再聚合）来进一步优化SV，取得了一些性能提升。

18. **GNNavi: Navigating the Information Flow in Large Language Models by Graph Neural Network** (Arxiv Feb 2024)  [[paper]](http://arxiv.org/abs/2402.11709) 提出将GNN插在LLM的某一层后面，强行使得information flow（token representation就是node representation）是从x->y和y->:连边，然后得到的node representation输给LLM的下一层（每个token的都保留着，因为GNN的输出也是所有node的输出）。最后只在ICL数据集上微调GNN，能够实现和lora媲美的速度和更好的acc。

19. **Decomposing Label Space, Format and Discrimination: Rethinking How LLMs Respond and Solve Tasks via In-Context Learning** (Arxiv April 2024) [[paper]](http://arxiv.org/abs/2404.07546) 将ICL能力分成1)正则化输出的label space、2)正则化输出的label format，和3)提升label space/format分布内的判别能力三个方面。结论：ICL的能力主要来自前两者。同时也在实验上间接证明了ICL会倾向于预测出context和test更像的样本的label。

20. **The Evolution of Statistical Induction Heads: In-Context Learning Markov Chains** (Arxiv Feb 2024) [[paper]](http://arxiv.org/abs/2402.11004) 在预测Markov序列任务上，揭示了存在一个学习出从简单到复杂function的过程（uniform -> unigram -> bigrams (optimal)）。此外，也验证了类似retrieval（n-gram），即找最相似的context token然后取它后面的token作为预测的机制 

21. **In-Context Language Learning: Architectures and Algorithms** (Arxiv Jan 2024) [[paper]](http://arxiv.org/abs/2401.12973) 构造了一个模拟的language token ICL任务，给了一系列实验证据说明transformer实现了和n-gram类似的retrieval过程

22. **Trusting Your Evidence: Hallucinate Less with Context-aware Decoding** (Arxiv May 2024) [[paper]](http://arxiv.org/abs/2305.14739) 为了增强对context的关注能力，提出在推理时加权以context为条件的预测和不含context的预测：$y=\text{softmax}((1+\alpha) p_\theta(y|c,x)-\alpha p_\theta(y|x))$​ 。背后的理论基础是朴素贝叶斯 [[blog]](https://spaces.ac.cn/archives/9617)

23. **How In-Context Learning Emerges from Training on Unstructured Data: On the Role of Co-Occurrence, Positional Information, and Noise Structures** (Arxiv Jun 2024) [[paper]](http://arxiv.org/abs/2406.00131) 在非ICL格式的数据上训练，探究了“国家-首都”类任务（预训练常见）和输出首字母任务（不常见），发现pattern在训练数据里的重复性和位置信息分别是这两种任务的关键。

24. **Benefits of Transformer: In-Context Learning in Linear Regression Tasks with Unstructured Data** (Arxiv Feb 2024) [[paper]](http://arxiv.org/abs/2402.00743) 分析多层、PE、multi head等模块对于提升ICL在线性回归任务上性能的作用。

25. **Do pretrained Transformers Learn In-Context by Gradient Descent?** (ICML 2024) [[paper]](http://arxiv.org/abs/2310.08540) 讨论了一下目前ICL工作的不切实际的setting，从一些实验指标上说明了ICL和GD有显著不同。

26. **Rectifying Demonstration Shortcut in In-Context Learning** (NAACL 2024) [[paper]](http://arxiv.org/abs/2403.09488) 发现context单词的字面意思会影响ICL分类的结果（一种shortcut）。提出了一种calibration的策略。

27. **Investigating the Pre-Training Dynamics of In-Context Learning: Task Recognition vs. Task Learning** (Arxiv June 2024) [[paper]](http://arxiv.org/abs/2406.14022) 训练过程中task learning和task recognition存在竞争现象

28. **Transformers Can Perform Distributionally-robust Optimisation through In-context Learning** (ICML 2024 workshop on ICL) [[paper]](https://openreview.net/pdf?id=MOgg2cEms5) ICL有一定的DRO的能力 

29. **How Do In-Context Examples Affect Compositional Generalization?** (ACL 2024) [[paper]](http://arxiv.org/abs/2305.04835) 发现context example对于组合泛化能力影响显著。具体来说，context example和query越像、example越多样、每个样本越简单，泛化能力越好。

30. **What Do Language Models Learn in Context? The Structured Task Hypothesis** (ACL 2024) [[paper]](http://arxiv.org/abs/2406.04216) 通过实验验证了ICL能够对预训练见过的任务进行复合的假设，否定了ICL仅仅能够进行分布内任务的试别以及ICL能够泛化到某些训练时没见过的任务的假设。

31. **What needs to go right for an induction head? A mechanistic study of in-context learning circuits and their formation** (ICML 2024) [[paper]](http://arxiv.org/abs/2404.07129) 识别了transformer在解决ICL的copy-and-paste任务中存在的三种circuit

32. **In-Context Learning of Energy Functions** (ICML 2024 ICL workshop) [[paper]](http://arxiv.org/abs/2406.12785) 提出了将next-token的条件分布建模为能量函数的形式，发现transformer也能在这种形式下展现出ICL能力

33. **From Words to Numbers: Your Large Language Model Is Secretly A Capable Regressor When Given In-Context Examples** (Arxiv April 2024) [[paper]](http://arxiv.org/abs/2404.07544) 发现诸如GPT-4，Claude-3之类的LLM能够在不重新训练的情况下做linear和non-linear regression，甚至有时能超过supervised training的方法（但仅限于很大的LLM）。

34. **Disentangling Latent Shifts of In-Context Learning Through Self-Training** (Arxiv Oct 2024) [[paper]](http://arxiv.org/abs/2410.01508) 针对ICL不稳定的问题，提出为student LLM训练一个adapter用来从teacher LLM那里获取context的知识。【insight】认为之前的vector系列工作只考虑attn head，不够全面。

35. **Learning Task Representations from In-Context Learning** (ICML 2024 ICL workshop) [[paper]](Learning Task Representations from In-Context Learning) 提出learnable task vector（LTV），为所有head增加可学习的权重，然后加权组合每一个head的activation来得到每一层function vector。发现其可以增强ICL的长度泛化能力。

36. **Task Diversity Shortens the ICL Plateau** (Arxiv Oct 2024) [[paper]](http://arxiv.org/abs/2410.05448) synthetic setting，在更多的function class上训练可以加快收敛。发现A任务训练到loss正在逃离plateau的checkpoint在B任务上继续训，可以加快B的训练，说明不同任务之间有一些common structure，提供了为什么多任务训练能更快收敛的一个解释。

37. **Many-Shot In-Context Learning** (ICML 2024 ICL workshop) [[paper]](http://arxiv.org/abs/2404.11018) ICL的潜力被few-shot限制了

38. **Out-of-distribution generalization via composition: a lens through induction heads in Transformers** (Arxiv Aug 2024) [[papaer]](http://arxiv.org/abs/2408.09503) 在OOD的copy任务上，发现了OOD性能源于执行不同功能层的composition（并没有测复杂的组合泛化任务）。还发现了induction head和previous token head的各自内部的表示的相似性。

39. **Context-Scaling versus Task-Scaling in In-Context Learning** (Arxiv Oct 2024) [[paper]](https://arxiv.org/pdf/2410.12783) 核心发现：kernel smoothing的特征映射是能够进行context scaling的关键

40. **Bayesian scaling laws for in-context learning** (Arxiv Oct 2024) [[paper]](http://arxiv.org/abs/2410.16531) 推导了一种基于贝叶斯的scaling law。在模拟数据集上效果比exponetial scaling law好，在真实LLM和数据集上效果还行。

    



### 2023

1. **Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?** [[paper]](https://arxiv.org/abs/2202.12837) 做了一系列消融实验来对ICL进行解释。主要结论：即使input和label不是一一对应，只要label的分布合理，那么ICL同样能给出较为正确的答案.
2. **Symbol tuning improves in-context learning in language models** (EMNLP 2023) [[paper]](http://arxiv.org/abs/2305.08298) 将demonstration的label换为无意义的symbol，然后微调，以此强迫模型学习input-label mapping。
3. **In-context Learning Generalizes, But Not Always Robustly: The Case of Syntax** (Arxiv Nov 2023) [[paper] ](In-context Learning Generalizes, But Not Always Robustly: The Case of Syntax) 本文通过构建一些语法任务来测试模型对于句子结构的理解能力，以及OOD泛化性能。总的说来，LLM还是会用到一些spurious correlation。
4. **A Closer Look at In-Context Learning under Distribution Shifts** (Arxiv May 2023) [[paper]](http://arxiv.org/abs/2305.16704) 在一定的分布偏移下，transformer比set-based MLP的性能好；在严重的分布偏移下，两种模型的ICL能力都丧失了。
5. **Few-shot Fine-tuning vs. In-context Learning: A Fair Comparison and Evaluation** (Arxiv May 2023) [[paper]](https://www.lsv.uni-saarland.de/wp-content/uploads/2023/07/Few-shot-Fine-tuning-vs.-In-context-Learning.pdf) 在参数量相当的情况下，ICL的OOD不如FT。30B的ICL跟6.7B的FT性能相当。大部分情况下ICL不如FT。
6. **Instruction-following Evaluation through Verbalizer Manipulation** (Arxiv July 2023) [[paper]](http://arxiv.org/abs/2307.10558) 发现LLM遵循flipped-label instructions的能力很差，说明ICL可能只是直接利用了预训练语料的知识，而不是学习了context。即使是强如GPT-4的模型也不能很好地遵循flipped-label instructions。
7. **Reasoning or Reciting? Exploring the Capabilities and Limitations of Language Models Through Counterfactual Tasks** (Arxiv Aug 2023) [[paper]](http://arxiv.org/abs/2307.02477) 一些主要发现：①模型在counterfactual的setting中性能会变差，且setting和常见的、符合事实的setting相差越远，性能越差，说明了模型可能的记忆现象。②在算术任务上，ICL能提升counterfactual（不同进制的计算）性能，但和default setting的差距难以抹平。
8. **What In-Context Learning "Learns" In-Context: Disentangling Task Recognition and Task Learning** (Findings of ACL 2023) [[paper]](http://arxiv.org/abs/2305.09731) 分别用随机label（x-y映射关系被破坏）和非自然语言label（x-y映射关系保留）来检验模型的从预训练知识中识别任务和从context中学习input-label映射关系的能力，发现：这两种能力同时存在；任务识别能力基本不随模型规模变化；in-context学习能力会随模型变大而上升。
9. **Larger language models do in-context learning differently** (Arxiv Mar 2023) [[paper]](http://arxiv.org/abs/2303.03846) 和disentanglement TR and TL 那篇差不多，发现了：小模型会倾向于用prior，随着模型增大，覆盖prior而从context学习映射关系的能力会越来越强。
10. **In-Context Learning Creates Task Vectors** (Arxiv Oct 2023) [[paper]](http://arxiv.org/abs/2310.15916) 同样发现context的最后一个token的表示encode了该任务的信息。通过实验发现ICL近似是在实现如下过程：1)从context学出一个映射函数 2)将这个映射函数用到query上来预测。一个重要观察是：说明模型更倾向于使用vector里的信息，而不是原始context
11. **Label Words are Anchors: An Information Flow Perspective for Understanding In-Context Learning** (EMNLP 2023) [[paper]](http://arxiv.org/abs/2305.14160) 浅层网络从text到label聚合信息，深层网络从label到last token聚合信息。
12. **Pretraining Data Mixtures Enable Narrow Model Selection Capabilities in Transformer Models** (Arxiv Nov 2023) [[paper]](http://arxiv.org/abs/2311.00871) 发现ICL在测试和预训练任务不相同时，性能不好。
13. **Pretraining task diversity and the emergence of non-Bayesian in-context learning for regression** (NeurIPS 2023) [[paper]](https://arxiv.org/abs/2306.15063) 发现预训练学习的任务越多，ICL在新任务上的泛化越强（不同任务：不同线性回归的W）
14. **The Transient Nature of Emergent In-Context Learning in Transformers** (NeurIPS 2023) [[paper]](http://arxiv.org/abs/2311.08360) 训练任务：每个序列的token都有一个label。该任务既可以用ICL解决也可以用In-weights Learning (IWL)解决。实验发现随着训练epoch增加，ICL性能先上升再下降，而IWL能力逐渐上升。
15. **THE EFFECTS OF PRETRAINING TASK DIVERSITY ON IN-CONTEXT LEARNING OF RIDGE REGRESSION** (ICLR 2023 workshop) [[paper]](https://openreview.net/pdf?id=EshX_qlA3o) 随着预训练时见到的线性回归w（都来自同一分布）越来越多，ICL表现逐渐从MMSE（预训练w的加权组合）变为岭回归（test理论最优）。
16. **Birth of a Transformer: A Memory Viewpoint** (NeurIPS 2023) [[paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/0561738a239a995c8cd2ef0e50cfa4fd-Paper-Conference.pdf) 构建了一个bigram任务，在简化setting下推导出了两层transformer要解决这个任务所应具备的参数闭式解，以此计算模型参数和最优解的差距来分析训练过程中的ICL能力的变化



### 2022

1. **What Can Transformers Learn In-Context? A Case Study of Simple Function Classes** (NeurIPS 2022) [[paper]](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c529dba08a146ea8d6cf715ae8930cbe-Abstract-Conference.html) 实验发现：1)linear function是能通过transformer学到的（性能能逼近最小二乘估计）2)ICL有一定的OOD泛化能力（train -> test, context -> test）3)ICL也能学到更复杂的函数，比如sparse linear functions、ReLU NNs、decision trees。
2. **Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?** (EMNLP 2022) [[paper]](http://arxiv.org/abs/2202.12837) 探究ICL work的因素。
3. **On the Compositional Generalization Gap of In-Context Learning** (Arxiv 2022) [[paper]](http://arxiv.org/abs/2211.08473) 在CFQ等组合泛化任务上测，发现大模型的OOD（query和context不一致）和ID之间的组合泛化能力的gap相比小模型更小。



## ICL Theories

### 2024

1. **How do Transformers perform In-Context Autoregressive Learning?** (Arxiv Feb 2024) [[paper]](http://arxiv.org/abs/2402.05787) 在限定linear attention、diagonal weight matrix等条件下，对于序列预测任务$s_{T+1}=Ws_T$（文章考虑的$W$是酉矩阵和正交矩阵两种情况），从理论上给出了取到全局最优解时，transformer 参数所应满足的性质。

2. **On Mesa-Optimization in Autoregressively Trained Transformers: Emergence and Capability** (Arxiv May 2024) [[paper]](http://arxiv.org/abs/2405.16845) 理论证明了，不同于直接在ICL目标上进行预训练，经过自回归预训练的one-layer linear attention不能在简单如服从高斯分布的序列上实现ICL。

3. **How Do Nonlinear Transformers Learn and Generalize in In-Context Learning?** (ICML 2024) [[paper]](http://arxiv.org/abs/2402.15607) 在进行ICL预训练的情况下，给出了非线性attention的ID和OOD的泛化保证

4. **Why Larger Language Models Do In-context Learning Differently?** (ICML 2024) [[paper]](http://arxiv.org/abs/2405.19592) 本文对于更大的模型更容易在flipped label任务上失败给了理论解释：大模型更容易受到prompt中noise的影响，而小模型只会关注更重要的feature所以不容易受到noise影响，进而使pretrain feature发挥更大的作用。

5. **Dual Operating Modes of In-Context Learning** (ICML 2024) [[paper]](http://arxiv.org/abs/2402.18819) 理论setting：在混合高斯的线性回归上预训练，分析了给定test context时的后验概率，解释了task recognition和task learning：发现context较短时以task recognition（调整后验的混合高斯的各分量的权重）为主。context变长之后以task learning为主。

6. **In-Context Learning with Transformers: Softmax Attention Adapts to Function Lipschitzness** (Arxiv May 2024) [[paper]](http://arxiv.org/abs/2402.11639) softmax能adaptively学一个attention window来实现将context $y_i$ 进行插值作为预测，将分类任务中见到的retrieval机制拓展到了回归任务上。

7. **Towards Better Understanding of In-Context Learning Ability from In-Context Uncertainty Quantification** (Arxiv May 2024) [[paper]](http://arxiv.org/abs/2405.15115) 理论，多头SoftMax attention，任务是估计p(y|x)和Var(y|x)，给出了分布内泛化error bound。

8. **An Information-Theoretic Analysis of In-Context Learning** (Arxiv Jan 2024) [[paper]](http://arxiv.org/abs/2401.15530) 在信息论视角下，将ICL泛化误差拆解为多项。

### 2023

1. **What learning algorithm is in-context learning? Investigations with linear models** (ICLR 2023) [[paper]](http://arxiv.org/abs/2211.15661) 还没看，理论理解ICL机制的文章，linear regression任务，但它的理论设定是模型要在ICL任务上预训练，与实际的Auto Regressive预训练有较大gap。它的证明思路也是通过网络参数构造解，和A Theoretical Understanding of Self-Correction through In-context Alignment这篇类似。
2. **Transformers as Algorithms: Generalization and Stability in In-context Learning** (ICML 2023) [[paper]](http://arxiv.org/abs/2301.07067) 考虑了context为一系列独立pair和前后样本有关联两种模式，在进行ICL预训练的条件下，给了一个non-linear transformer的excess risk的upper bound
3. **In-Context Convergence of Transformers** (Arxiv Oct 2023) [[paper]](http://arxiv.org/abs/2310.05249) linear regression任务，需要预训练，一层非线性attention，但是做了其他简化使得transforer就是在根据x之间的attention weight来加权组合各个context y作为最终预测。
4. **Trained Transformers Learn Linear Models In-Context** (Arxiv Oct 2023) [[paper]](http://arxiv.org/abs/2306.09927) linear regression任务，需要预训练，一层线性attention。证明了预训练loss收敛到全局最优解时，当训练和测试context足够长时，能学到测试prompt上的正确解W。
5. **What and How Does In-Context Learning Learn? Bayesian Model Averaging, Parameterization, and Generalization** (Arxiv Oct 2023) [[paper]](arXiv:2305.19420) 理论文章，还没看
6. **Transformers as Statisticians: Provable In-Context Learning with In-Context Algorithm Selection** (NeurIPS 2023) [[paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/b2e63e36c57e153b9015fece2352a9f9-Paper-Conference.pdf) 证明了存在一个L-层线性transformer在线性回归、lasso、ridge问题上error有上界。同时在理论和实验上发现了会自动选择最优预训练知识的现象。
7. **The Learnability of In-Context Learning** (NeurIPS 2023) [[paper]](https://openreview.net/forum?id=f3JNQd7CHM) 证明了当预训练分布包含下游任务的分布的mixuture，ICL能逼近下游任务上的贝叶斯最优分类器。



## Multimodal ICL

### 2024

1. **What Makes Multimodal In-Context Learning Work?** (CVPR 2024 Workshop on Prompting in Vision) [[paper]](https://arxiv.org/abs/2404.15736) 对Multimodal ICL的实验性分析
2. **Link-Context Learning for Multimodal LLMs** (CVPR 2024) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/html/Tai_Link-Context_Learning_for_Multimodal_LLMs_CVPR_2024_paper.html) 提出一种新的fine-tune MLLM的方法：让context和query具有一定的causal联系，发现能提升模型通过context学习新概念的能力
3. **Can Vision Language Models Learn from Visual Demonstrations of Ambiguous Spatial Reasoning?** (Arxiv Sep 2024) 
4. **Finding Visual Task Vectors** (ECCV 2024) [[paper]](https://arxiv.org/pdf/2404.05729) 

### 2023

1. **What Makes Good Examples for Visual In-Context Learning?** [[paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/398ae57ed4fda79d0781c65c926d667b-Paper-Conference.pdf) 纯vision ICL。找和query最相近的样本来做ICL，类似Link-context learning。

