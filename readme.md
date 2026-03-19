## Preface

本仓库记录关于LLM (large language models)和Multimodal LLM的文章。看过的文章会至少用一句话概括内容，有些还会有notes。只有标题的就是还没看过的，只是先存档到这里。

有关OOD generalization的paper list请移步（OOD list 已停止维护）：[link](https://github.com/NOVAglow646/OOD-Generalization-Paper-Reading-Notes)

###  🔥 Updates

- 2025-11 接下来主要关注agentic MLLM，latent visual reasoning，unified model，world model等。
- 2025-03 接下来主要关注MLLM的reasoning和perception的问题，以及LLM的reasoning、test-time scaling。
- 2024-12 接下来主要关注VLM的hallucination、reasoning问题。同时也会follow ICL的最新进展。
- 2024-05 接下来主要关注探究ICL机制的相关工作

## Directory

* [MLLM](#mllm)
  
  * [Evaluation and understandings of multimodal reasoning](#evaluation-and-understandings-of-multimodal-reasoning)
  * ⭐[Think with Images](#think-with-images)
  * [Latent Reasoning](#latent-reasoning)
  * ⭐[Improving Multimodal Reasoning](#improving-multimodal-reasoning)
  * ⭐[Improving Perception/Mitigating Hallucination](#improving-perception-mitigating-hallucination)
  * [Video models](#video-models)
  * [Vision-language Alignment](#vision-language-alignment)
  * [Interpretability and Understanding](#interpretability-and-understanding)
  * [Unifying Understanding and Generation](#unifying-understanding-and-generation)
  * [Multimodal ICL](#multimodal-icl)
  * [Reward Model](#reward-model)
  * [Prompt Learning](#prompt-learning)
  
* [LLM](#llm) 
  
  * ⭐[In-Context Learning](#in-context-learning)
  * [ICL Theories](#icl-theories)
  * ⭐[Reasoning and Test-time compute](#reasoning-and-test-time-compute)
  * [Distillation](#distillation)
  * [Alignment](#alignment)
  * [Interpretability](#interpretability)
  * [Other](#other)
  
* [Agents](#agents)



# MLLM

## Survey

### 2025

1.**Mind with Eyes: from Language Reasoning to Multimodal Reasoning** [[paper]](https://arxiv.org/pdf/2503.18071) 多模态推理综述

### 2024 

1. **A Survey on Multimodal Large Language Models** [[paper]](https://arxiv.org/pdf/2306.13549) 综述



## Benchmarks and Evaluation of Multimodal Reasoning

### 2025

1. **Can MLLMs Reason in Multimodality? EMMA: An Enhanced MultiModal ReAsoning Benchmark** (Arxiv Jan 2025) [[paper]](http://arxiv.org/abs/2501.05444) 一个比较全面的涵盖数学、物理、化学、代码的视觉推理任务的benchmark。发现文本CoT很难提升2D变换这种需要空间想象的任务的性能。
2. **Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?** (Arxiv 2025.04) [[paper]](http://arxiv.org/abs/2504.13837) RL相比base model只是增加了 k较小时候的pass@k acc。当k足够大，base model会反超RL model。在数学、code、visual reasoning任务上都验证了这一现象。
3. **MPBench: A Comprehensive Multimodal Reasoning Benchmark for Process Errors Identification** (Arxiv 2025.03) [[paper]](http://arxiv.org/abs/2503.12505)  从三个角度评测多模态PRM：1）评估单步正确性的能力 2）从多条推理路径中选出最优的能力 3）从某一步的多个candidate中选出最优的能力
4. **Multimodal RewardBench: Holistic Evaluation of Reward Models for Vision Language Models** [[paper]](https://arxiv.org/pdf/2502.14191) 所标注的数据为(prompt, chosen response, rejected response)三元组，但标注是trajectory-level的。用来测RM的preference是否准确。
5. **L-RewardBench: A Challenging Benchmark for Vision-Language Generative Reward Models** [[paper]](https://arxiv.org/pdf/2411.17451)
6. **VisuLogic: A Benchmark for Evaluating Visual Reasoning in Multi-modal Large Language Models** [[paper]](http://arxiv.org/abs/2504.15279) 类似公务员题的图形推理benchmark
7. **GeoLaux: A Benchmark for Evaluating MLLMs’ Geometry Performance**
   **on Long-Step Problems Requiring Auxiliary Lines** [[paper]](https://arxiv.org/pdf/2508.06226v1) 几何题benchmark，平均所需推理步数为6.51。包含41.8%的需要辅助线才能做的题。
8. **MM-CoT:A Benchmark for Probing Visual Chain-of-Thought Reasoning in Multimodal Models** (Arxiv 2025.12) [[paper]](http://arxiv.org/abs/2512.08228) 任务是让模型选出视觉正确、逻辑连贯的cot。发现主要错误类型为（比例从高到低）：重复已有context内容而无法做出实质性的下一步推理、被其他视觉信息干扰、依赖文本先验而没有正确利用视觉信息
9. **SpatialTree: How Spatial Abilities Branch Out in MLLMs** (Arxiv 2025.12) [[paper]](https://arxiv.org/abs/2512.20617) 将MLLM的能力划分为perception、mental mapping（与语言对齐）、mental simulation（推理和规划）、agentic（根据上一步状态产生下一步动作）。低难度正交，但对难度大的任务有用；简单任务上RL会overthinking，导致简单任务上提升不大；auto think（自适应RL长度）有用。

### 2024

1. **Is A Picture Worth A Thousand Words? Delving Into Spatial Reasoning for Vision Language Models** (NeurIPS 2024) [[paper]](http://arxiv.org/abs/2406.14852) 
   在三个合成的空间理解任务上评测LLM和LVM，主要发现：1）该任务的总体表现并不好 2）对于VLM而言，更依赖于语言信息而不是视觉信息做决策，去掉/扰乱视觉信息甚至会有提升 3）VLM中的language encoder比同样的单独LLM性能更好，说明多模态pretrain对于language有用。【insight】现有的将视觉信息转化到language space再进行推理的范式不够好。
2. **Can Vision Language Models Learn from Visual Demonstrations of Ambiguous Spatial Reasoning?** (Arxiv Sep 2024) [[paper]](https://arxiv.org/abs/2406.02537) 
3. **TOPVIEWRS: Vision-Language Models as Top-View Spatial Reasoners** (Arxiv June 2024) [[paper]](http://arxiv.org/abs/2406.02537) 提了一个新的俯视图理解的数据集，发现VLM的俯视图理解能力仍然很差
4. **Decomposing Complex Visual Comprehension into Atomic Visual Skills for Vision Language Models** [[paper]](https://openreview.net/pdf?id=nFU4xCyoe0) 原子视觉任务benchmark Atomic Visual Skills Benchmark (AVSBench) 
5. **DOES SPATIAL COGNITION EMERGE IN FRONTIER MODELS? ** (Arxiv Oct 2024) [[paper]](http://arxiv.org/abs/2410.06468) 提出了空间理解任务 SPACE benchmark。发现目前最强的模型在简单的空间任务上性能很差
6. **Towards Interpreting Visual Information Processing in Vision-Language Models** (ICLR 2025 886) 检查物体信息是否编码在了特定的vision token里。发现object token去掉之后模型掉点最严重。高gradient token影响也挺大。
7. **Zero-Shot Visual Reasoning by Vision-Language Models: Benchmarking and Analysis**



## Latent Multimodal Reasoning

### 2026

1. **【🔧SFT】Forest Before Trees: Latent Superposition for Efficient Visual Reasoning** [[paper]](http://arxiv.org/abs/2601.06803) (Arxiv 2026.01) 方法很简洁：将SFT的next-token label（比如位置t）替换为soft label（位置t开始到结尾T的每个位置的logits的沿窗口的softmax）。

### 2025

1. **【🔧SFT+🚀RL】Machine Mental Imagery: Empower Multimodal Reasoning with Latent Visual Tokens** (Arxiv 2025.07) [[paper]](http://arxiv.org/abs/2506.17218) 让模型生成latent token辅助推理。两阶段SFT+RL。SFT阶段一对齐MLLM生成的latent和gt helper image；SFT阶段二将生成的latent作为input，进行SFT。RL为GRPO，loss只加在text上（因为生成的latent
2. **【🔧SFT+🚀RL】Latent Visual Reasoning** (Arxiv 2025.10) [[paper]](https://openreview.net/forum?id=j84WR5ORsC) 只在visual cot（带crop图）上SFT + GRPO，SFT阶段对齐latent和gt img embedding。
3. **【🔧SFT】Chain-of-Visual-Thought: Teaching VLMs to See and Think Better with Continuous Visual Tokens **(Arxiv 2025.11) [[paper]](http://arxiv.org/abs/2511.19418) 思路：借助视觉模型（SAM、DepthAnything、PIDINet和DINO）提供监督来让模型生成visual token。训练过程循序渐进，分为四阶段：理解visual token、生成、用visual token推理、随机drop一些种类的visual token用于增强对所有token的利用。
4. **【🔧SFT+🚀RL】Monet: Reasoning in Latent Visual Space Beyond Images and Language** (Arxiv 2025.11) [[paper]](https://arxiv.org/pdf/2511.21395) 提出了一种新的latent visual reasoning SFT方法，和一种针对latent thinking的强化学习算法VLPO。在分布内和分布外任务上取得了提升。
5. **Mull-Tokens: Modality-Agnostic Latent Thinking** (Arxiv 2025.12)
6. **【Test-time training】Reasoning Within the Mind: Dynamic Multimodal Interleaving in Latent Space ** (Arxiv 2025.12) [[page]](https://mllm-dmlr.github.io/) [[paper]](https://arxiv.org/pdf/2512.12623) 用confidence作为奖励信号，对latent进行test-time梯度更新。性能提升一般。
7. **【🔧SFT】Interleaved Latent Visual Reasoning with Selective Perceptual Modeling** (Arxiv 2025.12) [[paper]](http://arxiv.org/abs/2512.05665) 两阶段SFT。第一阶段用一个额外的MLLM从aux img中选出部分emb用于和latent对齐；第二阶段纯文本CE loss。
8. **VisMem: Latent Vision Memory Unlocks Potential of Vision-Language Models** (Arxiv 2025.12) [[paper]](https://www.alphaxiv.org/abs/2511.11007) 增加了一个查询生成器（输入context输出query）用于生成记忆query Q，然后将Q与context X、可学习的memory token M 一起送入记忆生成器（长期和短期各一个，分别attach在vision encoder和LLM上）来生成最终的latent token。实验比较硬核，测的benchmark和复现的baseline很多。
9. **Latent Implicit Visual Reasoning** (Arxiv 2025.12) [[paper]](https://www.alphaxiv.org/abs/2512.21218) 两阶段SFT。第一阶段用了一个visual bottleneck机制：让answer token只能看到latent而看不到原始输入图像。第二阶段用正常attention。和Monet提出的机制类似。



## ⭐Improving Multimodal Reasoning

### 2025

1. **Imagine while Reasoning in Space: Multimodal Visualization-of-Thought** (Arxiv 2025.01) [[paper]](10.48550/arXiv.2501.07542) 利用Anole-7b这种能同时生成图片和文字的模型，每一步生成图片和文本，构成Multimodal Visualization-of-Thought，提升空间推理能力。只在2d网格视觉任务进行了测试。
1. **Boosting Multimodal Reasoning with MCTS-Automated Structured Thinking** (Arxiv 2025.02) [[paper]](http://arxiv.org/abs/2502.02339) training-free。定义一个动作空间（Visual Parsing、CoT、divide-and-conquer等）在一个500样本的小数据集上产生reasoning path，为每个问题进行MCTS：每一步从动作空间选择一个动作。为每个问题得到最优推理路径后，为每个路径计算Problem Condition Complexity (PCC)，每个问题-路径-PCC称为一个card。测试时，计算测试问题的PCC，并找出与之PCC最接近的card，让其按照这个card的每一步的action选择进行推理。这样避免了测试时进行复杂的搜索。
1. **Virgo: A Preliminary Exploration on Reproducing o1-like MLLM** (Arxiv 2025.02) [[paper]](http://arxiv.org/abs/2501.01904) 用少量（5k）纯文本的long thought数据训练MLLM就能带来显著提升
1. **URSA: Understanding and Verifying Chain-of-thought Reasoning in Multimodal Mathematics** (Arxiv 2025.02) [[paper]](http://arxiv.org/abs/2501.04686) 借助Gemini合成CoT做fine-tune。对于verifier的训练：逻辑正确性和perception正确性两种监督信号。逻辑正确性：用二分查找的方式获取中间步的correctness label：先找到导向错误的链，从逻辑链的中点开始做MCTS，如果导向错误，则说明错误在前半段，否则在后半段。perception正确性：prompt一个LLM把正确路径上的步骤改错，然后继续
1. **Introducing Visual Perception Token into Multimodal Large Language Model** (Arxiv 2025.02) [[paper]](http://arxiv.org/abs/2502.17425) 提了两种方法。方法一：fine-tune MLLM使其学会什么时候该输出一个“visual perception token”，其中包含图像关键区域的坐标信息，然后把这部分图片裁下来重新输进去；方法二：fine-tune MLLM使其学会什么时候该输出“re-encode token”，re-encode token是一个hidden rep，不需要要求其有可解码的意义。然后将训练MLLM根据re-encode token预测答案，同时利用re-encode token来筛选DINO的特征作为辅助信息输入MLLM。
1. **Visual-RFT: Visual Reinforcement Fine-Tuning** (Arxiv 2025.03) [[paper]](http://arxiv.org/abs/2503.01785) 借鉴deepseek-r1的思想，使用RL+verifiable reward来增强MLLM在物体检测和分类上的性能
1. **Visual Agents as Fast and Slow Thinkers** (ICLR 2025) [[paper]](http://arxiv.org/abs/2408.08862) 让switch adapter（其实是一个MLLM）来判断是否启动对视觉信息的进一步考察。若启动，则switch adapter会输出missing object信息和初步文本clue，输给一个proposal adapter（MLLM）根据missing object信息输出bounding box，或让一个SAM根据missing object信息进一步输出bounding box。最终将原图+初步clue+bounding box或分割的mask一起输给MLLM得到最终回答。
1. **【🚀RL】MM-Eureka: Exploring Visual Aha Moment with Rule-based Large-scale Reinforcement Learning** (Arxiv 2025.03) [[paper]](http://arxiv.org/abs/2503.07365) 在多模态推理上复现R1，rule-based RL（用的RLOO，和GRPO基本差不多），对internVL-2.5-instruct-8B和internVL-2.5-pretrained-38B做的RL。任务主要是数学视觉推理。
1. **【🚀RL】R1-Zero's "Aha Moment" in Visual Reasoning on a 2B Non-SFT Model** (Arxiv 2025.02) [[paper]](http://arxiv.org/abs/2503.05132) 对qwen-2-vl-2B做的GRPO。任务主要是空间推理。
1. **【🚀RL】Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models** [[paper]](https://arxiv.org/pdf/2503.06749) motivation：sec 3.1发现，直接用随便收集的10k开源数据进行GRPO不work。总体思路：
   1. （针对多模态感知的优化）先用fig2的框架prompt DS-R1来为现有的多模态问题生成高质量cot以及正确答案，得到vision-R1-cold数据集。
   2. 然后（sec 3.2.2）在这个数据集上SFT一个qwen2.5VL，但是发现会overthinking（输出很长但是错误的推理过程）。
   3. 为了解决overthinking，提出PTST（fig4），分成多阶段训练，每一阶段限制输出长度为L_s。
1. **VisualPRM: An Effective Process Reward Model for Multimodal Reasoning** [[paper]](http://arxiv.org/abs/2503.10291) [[project page]](https://internvl.github.io/blog/2025-03-13-VisualPRM/) 先通过MC采样得到step-wise分数，然后训一个PRM。并且构建了一个基于MC采样的具有process得分的数据集VisualPRM400K
1. **【🚀RL】R1-VL: Learning to Reason with Multimodal Large Language Models via Step-wise Group Relative Policy Optimization** (Arxiv 2025.03) [[paper]](http://arxiv.org/abs/2503.12937) 训练时其实还是正常的GRPO，只不过每个回答的reward计算时用到了对于每个step的评估。方法：1）先进行CoT sft warm up； 2）step-wise acc reward：eq2，注意，**是分配给整个solution的，只是这个reward用到了对于每个step的评估，所以称为step-wise。下面的validity reward同理**。当solution包含答案时才给分（正确为1+$\alpha k$，错误为$\alpha*k$），否则为0。k为该链中步骤和关键推理步骤（让GPT4从数据集中的每个cot中提取）；3）step-wise validity reward：包括完整性和逻辑性两个准则。完整性：回答必须包含背景、推理、答案三部分；逻辑性：背景必须在推理步之前，答案必须在推理步之后。同时满足完整性和逻辑性的solution才得到reward 1.
1. **Visual-o1: Understanding ambiguous instructions via multi-modal multi-turn chain-of-thoughts reasoning** (ICLR 2025) [[paper]](https://openreview.net/pdf/e4711feed2e5512d1ff80753981a2c637d597fc7.pdf) training-free, prompt工程，多轮CoT
1. **AtomThink: A Slow Thinking Framework for Multimodal Mathematical Reasoning** (CVPR 2025) [[paper]](http://arxiv.org/abs/2411.11930) 通过prompt限制每一步可能的action：一步推理/验证/得出结论，让LLM自己选；每个问题只产生一个探索路径；
1. **【🚀RL】OThink-MR1: Stimulating multimodal generalized reasoning capabilities via dynamic reinforcement learning** [[paper]](http://arxiv.org/abs/2503.16081) 提出根据training step来动态调整KL散度的权重
1. **【🚀RL】Boosting the Generalization and Reasoning of Vision Language Models with Curriculum Reinforcement Learning**  (Arxiv 2025.04) [[paper]](http://arxiv.org/abs/2503.07065) 三阶段从简到难的GRPO训练：判断题、多选题、open-ended generation。
1. **Benchmarking Multimodal CoT Reward Model Stepwise by Visual Program** (Arxiv 2025.04) [[http://arxiv.org/abs/2504.06606]] 利用visual programming技术，让code generation model生成解决问题的代码块，其优势在于可验证对错。然后利用MLLM将代码块和运行结果（作为step-wise annotation）转化为COT。以此生成的COT具有step-wise的多角度的annotation，用来训练一个RM（但没讲清楚RM的具体结构）。
1. **【🔧SFT】CogCoM: A Visual Language Model with Chain-of-Manipulations Reasoning** (ICLR 2025) [[paper]](http://arxiv.org/abs/2402.04236) 让GPT4生成针对多模态问题的工具调用链，然后将其转为多轮的VQA链，每轮包含子图片、子问题和答案，用这些数据对MLLM做SFT
1. **【🚀RL】SoTA with Less: MCTS-Guided Sample Selection for Data-Efficient Visual Reasoning Self-Improvement** [[paper]](http://arxiv.org/abs/2504.07934) 用MCTS筛选出更难的（至少迭代5次才做对的，以及迭代50次都没做对的）样本用来GRPO。是在qwen2.5VL-7B-instruct上做的RL。
1. **【🔧SFT】Do we Really Need Visual Instructions? Towards Visual Instruction-Free Fine-tuning for Large Vision-Language Models** (Arxiv 2025.02) [[paper]](http://arxiv.org/abs/2502.11427) 作者认为任务解决能力和感知能力应该是分开的两种能力，分别做纯文本和VL的sft，推理时混合这两种vector。
1. **【🚀RL】VL-Rethinker: Incentivizing Self-Reflection of Vision-Language Models with Reinforcement Learning** (Arxiv 2025.05) [[paper]](http://arxiv.org/abs/2504.08837) 提了两个技术：1）保存一些（问题，回答，advantage）对，将adv的数值作为概率重新sample，来强调非常对或者非常错的样本；2）Forecd rethinking: 由于发现常规的RL不一定能带来rethinking的pattern，提出在RL rollout时强者让模型进行self-verification/self-correction/self-questioning
1. **【❄training-free】VisuoThink: Empowering LVLM Reasoning with Multimodal Tree Search** (Arxiv 2025.05)  [[paper]](http://arxiv.org/abs/2504.09130) tree-search + vision-text interleaved reasoning。需要借助外部工具来获得视觉辅助信息，所以最终预测由majority vote得出。
1. 【**🚀RL**】**SophiaVL-R1: Reinforcing MLLMs Reasoning with Thinking Reward** (Arxiv 2025.05) [[paper]](http://arxiv.org/abs/2505.17018) 除了GRPO之外，还训练了一个3B的reward model（训练数据来自QwenVL-72B对于QwenVL-7B的rollout数据的打分），用来作为thinking的reward（但是并不是step-wise的，而是对整个thinking的reward）。最终reward是outcome reward和thinking reward的和。
1. **【🔧SFT+🚀RL】Pixel Reasoner: Incentivizing Pixel-Space Reasoning with Curiosity-Driven Reinforcement Learning** (Arxiv 2025.05) [[paper]](http://arxiv.org/abs/2505.15966) 两阶段训练，第一阶段通过SFT让MLLM初步具备输出bounding box的能力（训练数据构建：自带visual cues的数据集，或者是gpt4o生成）；第二阶段curiosity-driven RL，强制模型用bounding box辅助推理的比例不能低于某个阈值
1. **【🔧SFT】Don't Look Only Once: Towards Multimodal Interactive Reasoning with Selective Visual Revisitation** (Arxiv 2025.05) [[paper]]() 训一个linear head，输出input token positions的概率分布。最终输出的logit包含原始词汇空间和图片的position空间。训练数据构建方法：取QvQ的文本推理链，用Gemini提取视觉query，输给Qwen用relative attn机制（ICLR25那篇）获取bounding box
1. **【🔧SFT+🚀RL】Chain-of-Focus: Adaptive Visual Search and Zooming for Multimodal Reasoning via RL** (Arxiv 2025.05) [[paper]](http://arxiv.org/abs/2505.15436) SFT+RL两阶段训练。SFT数据构造过程：让gpt4.1生成问题和回答，回答正确性由qwen-vl-72b校对；让qwen-vl-72b判断问题是否可以回答还是需要更高的分辨率（zoom-in）；gpt4.1作为agent，调用detection、bbox adjusting、mm understanding等工具完成问题（工具其实就是qwen-vl-max），中间依靠ds-v3作为verifier进行反馈。
1. **【🚀RL】GRIT: Teaching MLLMs to Think with Images** (Arxiv 2025.05) [[paper]](http://arxiv.org/abs/2505.15879) 不需要SFT或bbox标注。只需要20个训练数据。reward包括：1）format：包括think、bbox（有bbox就给分）、rethink；2）counting：bbox数量和gt数量一致就给分；3）acc：gpt-4o + BELU-1相似度给分；当输出了bbox，并不需要把crop下来的小图作为新的image输入，而是直接让模型依据bounding box进行推理（后续实验发现输出bbox能提升对image的attention）
1. **【🔧SFT+🚀RL】SRPO: Enhancing Multimodal LLM Reasoning via Reflection-Aware Reinforcement Learning** （没有什么针对多模态的优化）两阶段训练。1）SFT：为了注入新知识，先让模型产生回答，然后让gpt4o-mini照着gt cot，进行简化或者改正；2）RL：GRPO+reflection reward：根据reflection前后的正确性给不同的得分
1. **【🚀RL】DeepEyes: Incentivizing "Thinking with Images" via Reinforcement Learning** (Arxiv 2025.05) [[paper]](http://arxiv.org/abs/2505.14362)  不需要SFT和外部模型蒸馏，只通过outcome reward就能激发出grounding能力。RL reward: acc+format+tool，其中tool reward是回答正确且至少调用一次perception时给分. 在高分辨率、grounding、多模态推理上都有提升，在高分辨率任务上提升尤其显著（V*bench 91.3）.
1. **【🔧SFT+🚀RL, ⭐NEW SOTA】Advancing Multimodal Reasoning: From Optimized Cold Start to Staged Reinforcement Learning** (Arxiv 2025.06) [[paper]](http://arxiv.org/abs/2506.04207) （没有什么针对多模态的优化）实验发现现有的部分mm sft数据长度短、难度低，在其上冷启动效果不如在更难的纯文本cot上训练。提出了三阶段训练：1）纯文本SFT cold-start；2）multimodal RL；3）text RL（文本任务上训，冻结vision tower）。ablation发现先MRL再TRL性能最好，单独用一种或顺序反过来都更差。
1. **【🔧SFT+🚀RL】MINT-CoT: Enabling Interleaved Visual Tokens in Mathematical Chain-of-Thought Reasoning** (Arxiv 2025.06) [[paper]](https://arxiv.org/abs/2506.05331) 亮点：interleaved CoT当中的visual cues是token，而不是bbox，这样crop比较灵活。数据：构建了一个数学数据集（需要借助gpt4o），每一步有token-level的图像区域标注；训练：text-sft，interleaved-sft，interleaved-RL
1. **【🔧SFT+🚀RL】Reinforcing Spatial Reasoning in Vision-Language Models with Interwoven Thinking and Visual Drawing** (Arxiv 2025.05) [[paper]](https://arxiv.org/pdf/2505.23678) Qwen2.5 VL-72B蒸馏SFT+RL
1. **【🔧SFT+🚀RL】Grounded Reinforcement Learning for Visual Reasoning** (Arxiv 2025.05) [[paper]](https://arxiv.org/pdf/2505.23678) 方法：1）构建SFT data：用qwen2.5-VL-72B做MCTS，要求每一步都输出grounding的坐标，选出答案正确的路径和corrected路径用于SFT；2）SFT+RL，RL reward中包含format reward，要求按照think-tool call-observation-answer的顺序输出
1. **【🚀RL】Advancing Multimodal Reasoning Capabilities of Multimodal Large Language Models via Visual Perception Reward** (Arxiv 2025.06) [[paper]](http://arxiv.org/abs/2506.07218) 不需要SFT，只需要从现有的mm cot里用一个LLM提取视觉相关的步骤作为gt，之后在这些问题上GRPO时加入perception reward：让一个LLM判断在RL rollout中是否存在gt中的视觉信息，按照出现的比例给分，出现0个就是0分，出现全部就是1分。只需要1.4K数据就能达到很好的性能。
1. **【❄Training-free】PyVision: Agentic Vision with Dynamic Tooling** (Arxiv 2025.07) [[paper]](http://arxiv.org/abs/2507.07998) prompt engineering，让advanced closed-source MLLM获得“合成新工具”的能力
1. **【🚀RL】Perception-Aware Policy Optimization for Multimodal Reasoning** (Arxiv 2025.07) [[paper]](http://arxiv.org/abs/2507.06448) 实验上发现perception error占了MLLM推理错误的大多数情况。提出PAPO，将corrupted image、question和正常GRPO rollout得到的response一起重新输给模型，得到corrupted response。最大化corrupted response和原始response的KL散度。为了解决最大化KL距离导致的collpase，还引入了一个entropy loss，同时降低原始和corrupted的entropy
1. **【🔧SFT+🚀RL】M2-Reasoning: Empowering MLLMs with Unified General and Spatial Reasoning** (Arxiv 2025.07) [[paper]](http://arxiv.org/abs/2507.08306) **数据：**构建了pure-text cot和RLVR的数据，包含general reasoning和spatial reasoning，用MLLM筛出了不同难度和推理质量较高的数据。**训练：**tricks包括：1）data sampling时每个batch任务一样，每个step从所有任务均匀采（但没有对此的ablation）；2）训练过程中online acc为0.5的会被分配最高的权重（eq 6），权重向acc=0和acc=1递减；3）空间推理问题，因为有些问题需要估计大小和距离，提出了一种连续reward
1. **【🔧SFT+🚀RL】Open Vision Reasoner: Transferring Linguistic Cognitive Behavior for Visual Reasoning** (Arxiv 2025.07) [[paper]](http://arxiv.org/abs/2507.05255) 1）language only SFT；2）language/multimodal PPO，verifiable 0/1 reward
1. **【🔧SFT+🚀RL】OpenThinkIMG: Learning to Think with Images via Visual Tool Reinforcement Learning** (Arxiv 2025.08) [[paper]](http://arxiv.org/abs/2505.08617) 合成了工具调用的CoT。SFT+GRPO。
1. **【🚀RL】Learning Only with Images: Visual Reinforcement Learning with Reasoning, Rendering, and Visual Feedback** [[paper]](http://arxiv.org/abs/2507.20766) 应用场景很局限，解决的是image-to-code任务（从chart或webpage生成图片）。提了一个仅需要图片数据的RL框架：让模型调用工具渲染图片，然后比较渲染出来的图片和原始图片的相似度作为reward。
1. **【🔧SFT+🚀RL】Look Again, Think Slowly: Enhancing Visual Reflection in Vision-Language Models** (EMNLP 2025) [[paper]](https://arxiv.org/pdf/2509.12132) 发现随着生成的进行，对vision token的注意力下降。提出在RL中将对vision token的attn加入reward。
1. **More Thought, Less Accuracy? On the Dual Nature of Reasoning in Vision-Language Models** [[paper]](https://arxiv.org/pdf/2509.25848) 有趣的实验发现：1）perception error为主 2）随着cot变长，立即让其输出答案时acc先上升后下降；3）提前终止回答可减少perception error比例。方法：用GPT5生成一堆针对图片的正误描述，插到RL的推理链中并立即让模型判断对错，作为perception reward，与正常的outcome acc reward一起使用。

### **2024**

1. **Thinking Before Looking: Improving Multimodal LLM Reasoning via Mitigating Visual Hallucination** (Arxiv Nov 2024) [[paper]](http://arxiv.org/abs/2411.12591) 对于VQA任务，提出thinking-before-looking范式，先利用一个LLM根据文本问题生成一堆更细致的问题，然后将这些问题和图片一起输给MLLM让其生成推理步骤。最终将原始问题、图片、推理步骤一起输给MLLM让其生成答案。

2. **Visual Sketchpad: Sketching as a Visual Chain of Thought for Multimodal Language Models** (NeurIPS 2024) [[paper]](http://arxiv.org/abs/2406.09403) 让模型生成代码来调用工具根据现有的视觉输入产生新的视觉图像来作为推理的辅助，可以提升在各种视觉相关任务上的能力。

3. **Task Navigator: Decomposing Complex Tasks for Multimodal Large Language Models** (CVPR 2024) [[paper]](https://openaccess.thecvf.com/content/CVPR2024W/MAR/papers/Ma_Task_Navigator_Decomposing_Complex_Tasks_for_Multimodal_Large_Language_Models_CVPRW_2024_paper.pdf) 工程文章，借助LLM根据历史子问题和模型回答，迭代产生多个子问题，提升MLLM完成复杂视觉理解任务的能力。提出了VersaChallenge benchmark，包括常识推理、物理关系推理、未来预测等。

4. **SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities** (CVPR 2024) [[paper]](https://ieeexplore.ieee.org/document/10658310/) 构建数据集，训了一个spatial-VLM用以解决空间任务

5. **【📊dataset】SpatialRGPT: Grounded Spatial Reasoning in Vision Language Models** (NeurIPS 2024) [[paper]](http://arxiv.org/abs/2406.01584) 构建空间位置关系数据集，添加了一个深度图->语言模块，来增强几何推理

6. **Multimodal Chain-of-Thought Reasoning in Language Models** (TMLR 2024) [[paper]](http://arxiv.org/abs/2302.00923) 两阶段训练，第一阶段接受文本和视觉的融合特征输出一个rationale（推理过程的文本描述），第二阶段将生成的rationale和原始文本结合，再与视觉特征融合重新输入模型产生预测。

7. **Thinking Before Looking: Improving Multimodal LLM Reasoning via Mitigating Visual Hallucination** (Arxiv Nov 2024) [[paper]](http://arxiv.org/abs/2411.12591) 对于VQA任务，提出thinking-before-looking范式，先利用一个LLM根据文本问题生成一堆更细致的问题，然后将这些问题和图片一起输给MLLM让其生成推理步骤。最终将原始问题、图片、推理步骤一起输给MLLM让其生成答案。

8. **Link-Context Learning for Multimodal LLMs** (CVPR 2024) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/html/Tai_Link-Context_Learning_for_Multimodal_LLMs_CVPR_2024_paper.html) 提出一种新的fine-tune MLLM的方法：让context和query具有一定的causal联系，发现能提升模型通过context学习新概念的能力

9. **Lever LM: Configuring In-Context Sequence to Lever Large Vision Language Models** (NeurIPS 2024) [[paper]](http://arxiv.org/abs/2312.10104) 先构建一个优质的ICL数据集，然后将该数据集中的image-text对视作token，用CLIP抽取特征作为token embedding，训练一个很小的Transformer（lever-LM）来在该数据集上进行next-token prediction（序列是从query到context这样倒着来的）。测试时，最后给定测试样本，拿lever-LM从该预先挑选好的数据集中预测后续的example来构成context。

10. **Natural Language Inference Improves Compositionality in Vision-Language Models** (ICLR 2025 Ratings 8866) [[paper]](https://openreview.net/forum?id=G3aXjVAJjU) prompt工程。任务是判断caption和图片相不相符。做法是让LLM生成与原始caption相符、不相符的yes or no问题，然后根据VLM在相符/不相符/原始问题上的logit来做出最终判断。

11. **Interleaved-Modal Chain-of-Thought** (Arxiv 2024.11) [[paper]](https://arxiv.org/pdf/2411.19488) 在每一个reasoning step选出attention最高的visual tokens，保持原图的顺序插入到视觉和文本输入之后、文本rationale开始之前的位置，之后再据此生成rationale。按此方法迭代生成多个reasoning step，然后再在其后生成最终答案。

12. **Progressive Multimodal Reasoning via Active Retrieval** (Arxiv 2024.12) [[paper]](Progressive Multimodal Reasoning via Active Retrieval) 提出了一个从外部知识库中根据当前推理步搜索相关知识，并通过MCTS来构建CoT的框架，并提出了在生成的CoT数据上进行PRM的方法。推理时根据PRM的打分，选取得分topk高的推理路径。

13. **Mulberry: Empowering MLLM with o1-like Reasoning and Reflection via Collective Monte Carlo Tree Search** (Arxiv 2024.12) [[paper]](Mulberry: Empowering MLLM with o1-like Reasoning and Reflection via Collective Monte Carlo Tree Search) [[code]](https://github.com/HJYao00/Mulberry) 用MCTS构建CoT，其中每一步打分利用多个模型；同时构建反思链，做法是构建一个“低得分节点-反思prompt-高得分节点”的思维链。然后用生成的总共260K数据进行fine-tune。

14. **Perception Tokens Enhance Visual Reasoning in Multimodal Language Models** (Arxiv 2024.12) [[paper]](http://arxiv.org/abs/2412.03548) 针对相对深度估计问题或计数问题，将深度图或bounding box转换为MLLM能处理的token来提供更精细的视觉信息，并加入到CoT中，来fine-tune MLLM。

15. **MR-MLLM: Mutual Reinforcement of Multimodal Comprehension and Vision Perception** (Arxiv 2024.06) [[paper]](http://arxiv.org/abs/2406.15768)

16. **【📊dataset】Visual CoT: Advancing Multi-Modal Language Models with a Comprehensive Dataset and Benchmark for Chain-of-Thought Reasoning** (NeurIPS 2024 DB track) [[paper]](https://proceedings.neurips.cc/paper_files/paper/2024/file/0ff38d72a2e0aa6dbe42de83a17b2223-Paper-Datasets_and_Benchmarks_Track.pdf) 造了一个数据集Visual CoT，包含推理关键视觉区域的bounding box的坐标。提出的方法：训练MLLM在推理时输出bounding box。

17. **Cantor: Inspiring Multimodal Chain-of-Thought of MLLM** (MM 2024) [[paper]](http://arxiv.org/abs/2404.16033) 纯prompt engineering文章。为了增强perception，提示MLLM根据问题找出具体该看什么图片细节，然后问一个MLLM让它专门去看，最后再综合它的输出来做最终回答

18. **Self-Correction is More than Refinement: A Learning Framework for Visual and Language Reasoning Tasks** (Arxiv 2024.10) [[paper] ](https://arxiv.org/pdf/2410.04055) 给MLLM提供Self-correction Prompt，然后选出改对的和改错的样本分别作为正负样本进行DPO。

19. **Beyond Embeddings: The Promise of Visual Table in Visual Reasoning** (EMNLP 2024) [[paper] ](http://arxiv.org/abs/2403.18252) 训练一个visual table generator，来产生对图片的详细描述。训练generator的方法：prompt GPT4V来生成visual table。总共从COCO找了61K数据。三阶段训练：1）caption数据上训练connector 2）在GPT生成的instruction tunning数据集上训练connector和LLM 3）在vis table数据上训练LLM。

20. **From the Least to the Most: Building a Plug-and-Play Visual Reasoner via Data Synthesis** (EMNLP 2024) [[paper]](https://arxiv.org/pdf/2406.19934) 先用grounding DINO检测图中物体获得一系列节点（单物体/多物体/整张图），让GPT4根据这些节点反推每一步回答什么样的子问题、怎样调用工具，才能从前一步的图片节点得到下一步的图片节点。最后让GPT4把子图、GPT4生成的子问题和工具调用参数合成一个推理链。让gpt4生成10k这样的数据用来训练llama3-8b做提提问题和合成的任务。之后让这个sft之后的llama3-8b生成50k推理链，用来sft一个llava-1.5-7b作为reasoner，其具备提出子问题和调用工具的能力。

 ### 2023

1. **Multi-modal Latent Space Learning for Chain-of-Thought Reasoning in Language Models** (Arxiv 2023.12) [[paper]](http://arxiv.org/abs/2312.08762) 认为CLIP的视觉特征不利于CoT推理。训练一个diffusion model来获取视觉特征。
2. **DDCoT: Duty-Distinct Chain-of-Thought Prompting for Multimodal Reasoning in Language Models** (NeurIPS 2023) [[paper]](http://arxiv.org/abs/2310.16436) 方法流程：1）让LLM拆解问题并判断哪些子问题不需要视觉信息就能回答；2）对于LLM回答不了的、需要视觉信息的子问题，调用现成的的VQA模型； 3）将子问题和它们的回答（包含视觉信息的文本描述）作为rationale让LLM推理。



## ⭐Think with Images

### Survey/Benchmark/Dataset/Understanding

### 2026

1. **What, Whether and How? Unveiling Process Reward Models for Thinking with Images Reasoning** [[paper]](http://arxiv.org/abs/2602.08346) 首个用于评测VLM在TWI推理任务重的PRM能力的benchmark；将TWI推理过程中的错误类型归为7类
2. **VTC-Bench: Evaluating Agentic Multimodal Models via Compositional Visual Tool Chaining** [[paper]](https://arxiv.org/abs/2603.15030) 合成了long-horizon（其实长程的构造主要也还是人工刻意扰动为主，如加噪、旋转，加一些任务用来测试开源和闭源模型visual tool-use的能力。测了opencv支持的32种工具（相比常见的，多了如颜色变换、二值化、边缘检测、调整亮度、计算连通区域等传统CV操作）。一些比较novel的发现:
   * 相比任务的GT tool-chain（工具调用次数平均3~7次，不过肉眼看case发现有些工具调用比较牵强，并非必需），绝大部分情况下模型会倾向于调用更少次数的tool（大部分是1次或两次）
   * 从7B到gemini，system prompt都是越详细越好；给出GT tool时更好
   * 闭源模型中，gemini3.0（code 51.2/interface 51.0）最强，显著强于gpt5.2（code 44.6/interface 40.7）

#### 2025

1. **【Survey】Thinking with Images for Multimodal Reasoning: Foundations, Methods, and Future Frontiers** (Arxiv 2025.06) [[paper]](http://arxiv.org/abs/2506.23918)
1. **【Dataset】Zebra-CoT: A Dataset for Interleaved Vision Language Reasoning** (Arxiv 2025.07) [[paper]](http://arxiv.org/abs/2507.16746)
1. **【Understanding】Visual Thoughts: A Unified Perspective of Understanding Multimodal Chain-of-Thought** (Arxiv 2025.05) [[paper]](http://arxiv.org/abs/2505.15510) 理解不同类型的visual thought（pure-text、edited-image、generated-image等）的性能、适用场景、内在机制
1. **【Survey】Explain Before You Answer: A Survey on Compositional Visual Reasoning** (Arxiv 2025.08) [[paper]](https://arxiv.org/pdf/2508.17298)
1. **【Benchmark】TIR-Bench: A Comprehensive Benchmark for Agentic Thinking-with-Images Reasoning** [[paper]](http://arxiv.org/abs/2511.01833) 构建了一些强烈依赖于工具调用才能做对的任务。一些takeaway：1）在一些复杂任务（比如给出拼图顺序，fig 5）上，单纯的perception（o3展现出的”understanding the images as a whole”）没用，必须得借助code。2）在rotationOCR任务上，单纯增加text-based COT的数据进行SFT几乎没有提升



### Methods

### 2026

1. **Reliable Thinking with Images** (Arxiv 2026.02) [[paper]](http://arxiv.org/abs/2602.12916) 提出了reliability metric来衡量TWI推理过程的可靠性：计算高熵token的平均熵作为reliability。实验发现reliability与acc负相关，且视觉证据阶段到后续推理阶段的reliability上升越多，则acc越高。据此提出了一种先筛选高reliability traces再以reliability加权做majority voting的方法。

#### 2025

1. **【🔧SFT】CogCoM: A Visual Language Model with Chain-of-Manipulations Reasoning** (ICLR 2025) [[paper]](http://arxiv.org/abs/2402.04236) 让GPT4生成针对多模态问题的工具调用链，然后将其转为多轮的VQA链，每轮包含子图片、子问题和答案，用这些数据对MLLM做SFT
2. **【🔧SFT+🚀RL】Pixel Reasoner: Incentivizing Pixel-Space Reasoning with Curiosity-Driven Reinforcement Learning** (Arxiv 2025.05) [[paper]](http://arxiv.org/abs/2505.15966) 两阶段训练，第一阶段通过SFT让MLLM初步具备输出bounding box的能力（训练数据构建：自带visual cues的数据集，或者是gpt4o生成）；第二阶段curiosity-driven RL，强制模型用bounding box辅助推理的比例不能低于某个阈值
3. **【🔧SFT】Don't Look Only Once: Towards Multimodal Interactive Reasoning with Selective Visual Revisitation** (Arxiv 2025.05) [[paper]]() 训一个linear head，输出input token positions的概率分布。最终输出的logit包含原始词汇空间和图片的position空间。训练数据构建方法：取QvQ的文本推理链，用Gemini提取视觉query，输给Qwen用relative attn机制（ICLR25那篇）获取bounding box
4. **【🔧SFT+🚀RL】Chain-of-Focus: Adaptive Visual Search and Zooming for Multimodal Reasoning via RL** (Arxiv 2025.05) [[paper]](http://arxiv.org/abs/2505.15436) SFT+RL两阶段训练。SFT数据构造过程：让gpt4.1生成问题和回答，回答正确性由qwen-vl-72b校对；让qwen-vl-72b判断问题是否可以回答还是需要更高的分辨率（zoom-in）；gpt4.1作为agent，调用detection、bbox adjusting、mm understanding等工具完成问题（工具其实就是qwen-vl-max），中间依靠ds-v3作为verifier进行反馈。
5. **【🔧SFT】Thinking with Generated Images** (Arxiv 2025.05) [[paper]](http://arxiv.org/abs/2505.22525) 主要目标是更好地生成。构建SFT数据：包含反思和设定中间目标。
6. **【🚀RL】Visual Planning: Let's Think Only with Images** (Arxiv 2025.05) [[paper]](http://arxiv.org/abs/2505.11409) 主要解决grid-based navigation问题。纯视觉CoT.
7. **【🚀RL】GRIT: Teaching MLLMs to Think with Images** (Arxiv 2025.05) [[paper]](http://arxiv.org/abs/2505.15879) 不需要SFT或bbox标注。只需要20个训练数据。reward包括：1）format：包括think、bbox（有bbox就给分）、rethink；2）counting：bbox数量和gt数量一致就给分；3）acc：gpt-4o + BELU-1相似度给分；当输出了bbox，并不需要把crop下来的小图作为新的image输入，而是直接让模型依据bounding box进行推理（后续实验发现输出bbox能提升对image的attention）
8. **【🔧SFT+🚀RL】Grounded Reinforcement Learning for Visual Reasoning** (Arxiv 2025.05) [[paper]](https://arxiv.org/pdf/2505.23678) 方法：1）构建SFT data：用qwen2.5-VL-72B做MCTS，要求每一步都输出grounding的坐标，选出答案正确的路径和corrected路径用于SFT；2）SFT+RL，RL reward中包含format reward，要求按照think-tool call-observation-answer的顺序输出
9. **ReFocus: Visual Editing as a Chain of Thought for Structured Image Understanding** (ICML 2025) [[paper]](https://openreview.net/forum?id=a7qFlPOTix)
10. **【🔧SFT+🚀RL】MINT-CoT: Enabling Interleaved Visual Tokens in Mathematical Chain-of-Thought Reasoning** (Arxiv 2025.06) [[paper]](https://arxiv.org/abs/2506.05331) 亮点：interleaved CoT当中的visual cues是token，而不是bbox，这样crop比较灵活。数据：构建了一个数学数据集（需要借助gpt4o），每一步有token-level的图像区域标注；训练：text-sft，interleaved-sft，interleaved-RL
11. **【❄Training-free】PyVision: Agentic Vision with Dynamic Tooling** (Arxiv 2025.07) [[paper]](http://arxiv.org/abs/2507.07998) prompt engineering，让advanced closed-source MLLM获得“合成新工具”的能力
12. **【🔧SFT+🚀RL】Machine Mental Imagery: Empower Multimodal Reasoning with Latent Visual Tokens** (Arxiv 2025.07) [[paper]](http://arxiv.org/abs/2506.17218) 让模型生成latent token辅助推理。两阶段SFT+RL。SFT阶段一对齐MLLM生成的latent和gt helper image；SFT阶段二将生成的latent作为input，进行SFT。RL为GRPO，loss只加在text上（因为生成的latent
13. **【🔧SFT+🚀RL】OpenThinkIMG: Learning to Think with Images via Visual Tool Reinforcement Learning** (Arxiv 2025.08) [[paper]](http://arxiv.org/abs/2505.08617) 合成了工具调用的CoT。SFT+GRPO。
14. **【🚀RL】Learning Only with Images: Visual Reinforcement Learning with Reasoning, Rendering, and Visual Feedback** (Arxiv 2025.07) [[paper]](http://arxiv.org/abs/2507.20766) 应用场景很局限，解决的是image-to-code任务（从chart或webpage生成图片）。提了一个仅需要图片数据的RL框架：让模型调用工具渲染图片，然后比较渲染出来的图片和原始图片的相似度作为reward。
15. **【🔧SFT+🚀RL】Thyme: Think Beyond Images** (Arxiv 2025.08) [[paper]](https://arxiv.org/pdf/2508.11630) SFT+RL训练模型生成code来操作图片进行推理的能力。构建了SFT和RL数据集。提出了一种dynamic temperature的策略：生成代码时temperature=0，生成文本推理时temperature=1.0
16. **【🔧SFT+🚀RL】Reinforced Visual Perception with Tools ** (Arxiv 2025.09) [[paper]](https://arxiv.org/pdf/2509.01656)
17. **【🔧SFT+🚀RL】Mini-o3: Scaling Up Reasoning Patterns and Interaction Turns for Visual Search** (Arxiv 2025.09) [[paper] ](https://arxiv.org/pdf/2509.07969)构建了一个多轮visual search的SFT数据集。针对RL rollout时回复过长导致超出context从而无法判断对错的问题，提出将这部分回复mask掉，不计算reward。
18. **【🔧SFT+🚀RL】DeepeyesV2: Toward Agentic Multimodal Model** (Arxiv 2025.11) [[paper]](http://arxiv.org/abs/2511.05271) 比较接近真正agent MLLM的形态，能产生code调用工具并联网搜索。
19. **【🔧SFT+🚀RL】V-Thinker: Interactive Thinking with Images** [[paper]](http://arxiv.org/abs/2511.04460) (Arxiv 2025.11) 生成code编辑图片。设计了一种数据生成策略，借助GPT5，从一个知识集和和一个工具集和出发，让GPT5生成问题以及cot的同时不断对它们进行扩充。cot中包含代码以及渲染出的图片（V-Interaction-400K）。perception SFT + cold start SFT + GRPO RL。
20. **【🔧SFT】DeepSketcher: Internalizing Visual Manipulation for Multimodal Reasoning** (Arxiv 2025.09, ICLR26 withdrawn) [[paper]](http://arxiv.org/abs/2511.04460) 给MLLM加了一个image embedding editing模块，输入为原始图片emb和模型自己生成的action embedding，输出为编辑后的图片（但是没给可视化）。监督信号为code渲染出的中间步图片。还构建了一个用code渲染图片的cot数据集。
21. **【🔧SFT】Skywork-R1V4: Toward Agentic Multimodal Intelligence through Interleaved Thinking with Images and DeepResearch** (Arxiv 2025.12) [[paper]](https://arxiv.org/pdf/2512.02395) 能think with images和web search的agent MLLM。数据构建流程是关键。纯SFT训练。
22. **【🔧SFT+🚀RL】Thinking with Programming Vision: Towards a Unified View for Thinking with Images** (Arxiv 2025.12) [[paper]](http://arxiv.org/abs/2512.03746) 在构造数据时，通过对原图做增强扰动来保证工具调用的必要性。RL时候通过给问题预先标注好标准工具的元数据，实现了dense reward：奖励使用预先定义的工具、crop的IoU、以及对使用超出定义的有用工具的奖励。同时还使用了多种惩罚reward以避免reward hacking等行为。
23. **【🚀RL】Thinking with Images via Self-Calling Agent** (Arxiv 2025.12) [[paper]](http://arxiv.org/abs/2512.08511)
24. **【🚀RL】Figure It Out: Improve the Frontier of Reasoning with Active Visual Thinking** [[paper]](https://www.alphaxiv.org/abs/2512.24297?chatId=019b7d9d-3028-76a9-9b9f-a0b17bc39c79) 提出FIGR。RL中用了一个adaptive reward：当问题依赖辅助图片时用了工具做对给1.0，不依赖时用了工具给0.2，否则0。测的是纯文本数学任务（AIME、AMC）等。让qwen3-vl-32b用code渲染图像，能超过qwen3-32b-thinking。
25. **【🔧SFT+🚀RL】SenseNova-MARS: Empowering Multimodal Agentic Reasoning and Search via Reinforcement Learning** (Arxiv 2025.12) [[paper]](https://www.alphaxiv.org/abs/2512.24330?chatId=019b7da4-9112-7df0-beb3-ab57204a2b4c)  
    * 工具：crop +（txt/img）search。
    * 数据合成：先选出qwen2.5-vl-7b 8次回答中答对少于1次的难样本，用gemini2.5-pro-flash合成trajectory，用gpt4o校验格式、逻辑和答案正确性（3000条SFT数据）。
    * RL设计：针对多模态工具调用回复之间长度、reward差异大的问题，提出BN-GSPO，在GSPO的基础上，算出group relative adv之后，再在batch之内将各group的adv进行normalization。



## ⭐Improving Perception/Mitigating Hallucination

### 2026

1. **【🚀RL】Zooming without Zooming: Region-to-Image Distillation for Fine-Grained Multimodal Perception** (Arxiv 2026.02) [[paper]](https://www.alphaxiv.org/abs/2602.11858) 针对细粒度感知任务，提出了无需在测试时调用工具的方法：让教师模型基于原始图片I和问题Q找出关键区域B，然后在关键区域上加bbox，得到I‘，同时对原始问题Q加一个“关注bounding box区域“的prompt。直接在这个数据上直接RL，感知能力提升显著。

### 2025

1. **The Hidden Life of Tokens: Reducing Hallucination of Large Vision-Language Models via Visual Information Steering** (Arxiv 2025.02) [[paper]](http://arxiv.org/abs/2502.03628) 发现随着生成的进行，图片中真实出现的元素的token在logit中的排名会逐渐下降，而幻觉词的排名会逐渐靠前。提出了一种较为启发式的类似task vector的方法来缓解。实验效果上主要是降低幻觉，而不是增强推理。
1. **MLLMs Know Where to Look: Training-free Perception of Small Visual Details with Multimodal LLMs** （ICLR 2025) [[paper]](https://openreview.net/forum?id=DgaY5mDdmT) 发现MLLM在object identification任务中能够关注到正确的视觉区域，即使回答错误。提出了几个自动化的training-free的裁剪出目标区域的方法。将目标区域的visual token连接到原始图片token后面。
1. **See What You Are Told: Visual Attention Sink in Large Multimodal Models **(ICLR 2025) [[paper]](https://openreview.net/forum?id=7uDI7w5RQA) 发现VLM中存在一些image token被分配的attention score总是很高，称为visual sink token。发现：mask它们造成的性能下降远不如mask等量随机token。提出的方法：先找到对于sink token的attention和non-sink token attention之比较高的head（这些head是关注于图像的head），然后将sink token的attention砍掉一定比例，将这部分score按比例分配到其他vis token上。
1. **Stop Looking for Important Tokens in Multimodal Language Models:  Duplication Matters More** 
1. **Towards Self-Improving Systematic Cognition for Next-Generation Foundation MLLMs** (Arxiv 2025.03) [[paper]](http://arxiv.org/abs/2503.12303) 让gpt-4o做chain-of-description，生成高质量perception数据，来做fine-tune
1. **Socratic Questioning: Learn to Self-guide Multimodal Reasoning in the Wild** (Arxiv 2025.01) [[paper]](http://arxiv.org/abs/2501.02964) 让模型自己提出子问题并回答，得到对图片的细致描述，再回答最开始的问题。构造这样的数据集之后用来做Fine-tune
1. **Perception-R1: Pioneering Perception Policy with Reinforcement Learning** (Arxiv 2025.04) [[paper]](http://arxiv.org/abs/2504.07954) 用GRPO训perception任务。一些发现：explicit thinking对于visual grounding、OCR、counting等perception任务不利；RL相比RL+SFT和SFT在复杂感知任务（多物体计数、detection）上提升较大，但在相对不那么复杂的grounding和OCR任务上相比RL+SFT和SFT提升有限。
1. **【❄training-free】Your Large Vision-Language Model Only Needs A Few Attention Heads For Visual Grounding** (Arxiv 2025.04) [[paper]](http://arxiv.org/abs/2503.06287) 发现存在少量的attn head的attention map对物体的标注很准。找这样的head的方法：考虑最后一个input文本token对全部image token的attention，先从所有head中选出对image attention比较大的，然后从中选出10个spatial entropy最低的（计算方法为eq3）。然后统计每个head被选为top-10低 entropy的频率。选出最被频繁选中的head作为grounding head。取它们的attention map作为grounding的依据。
1. **【🔧SFT，hallucination new SOTA】Generate, but Verify: Reducing Visual Hallucination in Vision-Language Models with Retrospective Resampling** [[blog]](https://reverse-vlm.github.io/) 在生成过程中随时监测幻觉的产生并在产生幻觉时启动回溯，重新生成
1. **【📊dataset】Weaving Context Across Images: Improving Vision-Language Models through Focus-Centric Visual Chains** (Arxiv 2025.04) [[paper]](https://www.arxiv.org/pdf/2504.20199) 解决多图片任务，提了一个多图问题数据集，每个样本包含一个推理路径，每一步包含应该看哪一张图片。
1. **【❄training-free】DyFo: A Training-Free Dynamic Focus Visual Search for Enhancing LMMs in Fine-Grained Visual Understanding** (CVPR 2025) [[paper]](https://arxiv.org/pdf/2504.14920) 1）MCTS的reward：每一个节点表示一个子图，该节点的reward为：如果该节点的子图片和该节点的文本一致，则为1乘以子图占全图的面积比（？）2）根据树搜索结果获取最终预测的方法：每个节点对应于一个prediciton，权重为节点的reward。然后进行reweighted majority vote得出最终预测。3）根据文本获取子图的方法：让一个expert（“Lang-Segment-Anything”）来做，expert接受focus文本、action（focus或scatter），crop出一个子图4）提出下一个观测对象的过程：让MLLM基于当前的子图和文本，提出一个新的文本，用以提供给vision expert crop子图。
1. **【⚖DPO】 Unsupervised Visual Chain-of-Thought Reasoning via Preference Optimization** (Arxiv 2025.04) [[paper]](http://arxiv.org/abs/2504.18397)
1. **【🔧SFT 】Analyzing and Mitigating Object Hallucination: A Training Bias Perspectiv (Arxiv 2025.08)** [[paper]](https://www.alphaxiv.org/abs/2508.04567) 构建了一个benchmark发现MLLM更容易在训练见过的图片上出现幻觉，且用一个probe发现lm_head的输出相比其他MLLM模块的输出导致幻觉。提出了只SFT lm head的一种做法。

### 2024

1. **Mitigating Hallucination in Large Vision-Language Models via Modular Attribution and Intervention** (ICLR 2025 8866) [[paper]](https://openreview.net/forum?id=Bjq4W7P2Us) 发现幻觉的产生是由于某些特定的attention head，这些head是源自VLM的LM部分。他们会给文本分配更高的attention。提出了在推理时关闭这些幻觉head和在instruction tunning时专门调这些head两种改进方法。
2. **Reducing Hallucinations in Large Vision-Language Models via Latent Space Steering** (ICLR 2025 886) [[paper]](https://openreview.net/forum?id=LBl7Hez0fF) 动机：发现使用扰动后再平均的vision feature能降低幻觉，认为幻觉来自vision encoder的不够鲁棒。提出使用in-context vector的做法，计算从正常feature到扰动平均后的feature的主成分，加到推理的时候。
3. **Analyzing and Mitigating Object Hallucination in Large Vision-Language Models** (ICLR 2024) [[paper]](http://arxiv.org/abs/2310.00754) 发现了幻觉产生的几个触发因素：1)训练数据中的某两种对象的spurious共现关系 2)decoding过程的不确定性会将幻觉词采样出来（即使幻觉词的生成概率本不应该是最高） 3)幻觉更容易出现在生成文本中靠后的位置
4. **Debiasing Multimodal Large Language Models** (Arxiv Mar 2024) [[paper]](http://arxiv.org/abs/2403.05262) 同样发现了VLM关注text token的问题。提出了两种decoding的策略。其中一种类似Trusting Your Evidence那篇增强对于context的关注的contrastive decoding方法： $y=\text{softmax}((1+\alpha) p_\theta(y|v,x)-\alpha p_\theta(y|v',x))$ ，其中第一项和第二项分别表示正常的图文输入和仅文本输入时的输出。
5. **IBD: Alleviating Hallucinations in Large Vision-Language Models via Image-Biased Decoding** (Arxiv Feb 2024) [[paper]](http://arxiv.org/abs/2402.18476) 也提出了contrastive decoding的方法，用一个更加关注视觉token的模型 $\hat{\theta}$ 的logit减去原始模型 $\theta$ 的logit，该项称为CD score。构建“更加关注视觉token的模型”的方法：增大对视觉token的attention score。同时使用两个自适应权重来调节该contrastive decoding的程度：1) $\hat{\theta}$ 和 $\theta$ 的预测越像，CD score权重越小；2) 由于发现生成content token（有实际意义的）相比function token（无实际意义的连词等）的CD score更大，也就是说更加关注image只对content token的正确生成更有利，所以对content token添加更大的权重，而对function token添加较小的权重。
6. **Paying More Attention to Image: A Training-Free Method for Alleviating Hallucination in LVLMs** (ECCV 2024) [[paper]](https://arxiv.org/pdf/2407.21771) 发现当去掉图像，且让模型在其在有图像的情况下所生成的文本的基础上继续生成，仍然会出现相同的幻觉。这种现象被称为text inertia（文本惯性）幻觉。提出的方法也是contrastive decoding：用正常的prediction减去纯文本的prediction
7. **Mitigating object hallucinations in large vision-language models through visual contrastive decoding** (CVPR 2024) Visual Contrastive Decoding (VCD)
8. **Mitigating hallucinations in large vision-language models with instruction contrastive decoding** (ACL Findings 2024) Instruction Contrastive Decoding (ICD)
9. **OPERA: Alleviating Hallucination in Multi-Modal Large Language Models via Over-Trust Penalty and Retrospection-Allocation** (CVPR 2024) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_OPERA_Alleviating_Hallucination_in_Multi-Modal_Large_Language_Models_via_Over-Trust_CVPR_2024_paper.pdf) 发现生成回答中的summary token（指attn都集中在其上的token，且往往是无意义token，无法蕴含丰富的视觉信息）越多，幻觉越严重。提出了识别生成token中的summary token并据此减轻幻觉的策略
10. **Self-Introspective Decoding: Alleviating Hallucinations for Large Vision-Language Models** (ICLR 2025 Ratings: 8665) [[paper]](http://arxiv.org/abs/2408.02032) 首先指出了过往的contrastive decoding方法的问题：有可能所减去的幻觉输出“不够幻觉”，导致正常输出减去它之后反而不准确了。本文认为低attention score的vision token更容易导致幻觉，因此为了更好地引发幻觉输出再减去它，提出在推理时仅保留低attention score的token。 
11. **Intervening Anchor Token: Decoding Strategy in Alleviating Hallucinations for MLLMs** (ICLR 2025 Ratings: 8866) [[paper]](https://openreview.net/forum?id=zGb4WgCW5i) 先定义了一种分析工具：token propagation probability $\rho$ ，来描述一个token在前传时的贡献。发现幻觉和 $\rho$ 的低熵有关（attention都集中在summary token上了，从而丢失了视觉token的信息）。理论证明了将QK矩阵的二范数控制在一个合理范围内可以增大 $\rho$ 的熵，提了一个启发式策略来实现这一目标。
12. **Visual Description Grounding Reduces Hallucinations and Boosts Reasoning in LVLMs** (ICLR 2025 Ratings: 8666) [[paper]](https://openreview.net/forum?id=3PRvlT8b1R) 现有的解决幻觉的方法难以提升在视觉推理benchmark上的能力。VLM能识别视觉元素，但难以利用它们进行推理。
13. **Look Twice Before You Answer: Memory-Space Visual Retracing for Hallucination Mitigation in Multimodal Large Language Models** (ICLR 2025 rejected) [[openreview]](https://openreview.net/forum?id=tkg9XMFo0H) 找output prediction entropy最大的层，然后将visual token作为额外信息，加入到FFN之后
14. **Self-Correcting Decoding with Generative Feedback for Mitigating Hallucinations in Large Vision-Language Models** (ICLR 2025) [[openreview]](https://openreview.net/forum?id=tTBXePRKSx) idea：生成模型引导VLM以减少幻觉。用LVLMs产生的初始响应生成图像，该图像充当辅助视觉参考，并提供自我反馈。
15. **Dense Connector for MLLMs** [[paper]](https://arxiv.org/abs/2405.13800) (NeurIPS 2024)



## Video models

### 2025

**Thinking with Video: Video Generation as a Promising Multimodal Reasoning Paradigm** (Arxiv 2025.11) [[paper]](https://arxiv.org/pdf/2511.04570) 发现在视觉中心任务上，视频生成模型（sora2）性能逼近顶尖闭源vlm（gpt5、gemini2.5pro等）。但在文本中心任务上性能差距较大。可以通过Self-consistency和ICL来提升sora做推理任务的能力。



## Think with Videos

### 2026

1. **Video-Thinker: Sparking "Thinking with Videos" via Reinforcement Learning** (Arxiv 2025.10) [[paper]](https://www.arxiv.org/abs/2510.23473) SFT+GRPO教会模型先输出<time>定位关键片段，再输出<caption>来描述，最后<think>的结构化思维方式。数据合成策略：
   * 针对 “有描述无推理” 的数据：这类数据具备精确的时间段标注和详尽的动作描述，但缺乏深度的逻辑问答。利用 DeepSeek-R1 强大的逻辑推理能力，以原有的细粒度片段描述为上下文，合成出需要跨越多个时间片段进行综合分析的复杂多跳问题，将感知任务升级为逻辑推理任务。
   * 针对 “有问答无细节” 的数据（如 STAR、ScaleLong、LVBench）：这类数据虽然包含极具挑战性的推理问答，却往往缺失了支撑答案的具体视觉描述。团队借助 Gemini-2.5-Flash-Lite 的长窗口视觉理解能力，以标准答案为锚点进行反向推导，为关键时间窗口生成了与答案强相关的精细化视觉描述（Answer-Conditioned Captions），填补了推理过程中视觉证据的空白。





## Vision-language Alignment

### 2025

1. **Visual Representation Alignment for Multimodal Large Language Models** [[paper]](https://arxiv.org/pdf/2509.07979) 发现MLLM随着层数加深，视觉表示离CLIP encoder的输出越来越远。提出对齐模型中间某一层表示和visual encoder的输出。（发现32层中，第16层效果最好）。



## Interpretability and Understanding

### 2025

1. **Towards Understanding How Knowledge Evolves in Large Vision-Language Models** (CVPR 2025) [[paper]](https://arxiv.org/pdf/2504.02862) 
1. **Rethinking Visual Layer Selection in Multimodal LLMs** (Arxiv 2025.04) [[paper]]()
1. **SFT or RL? An Early Investigation into Training R1-Like Reasoning Large Vision-Language Models** (Arxiv 2025.05) [[paepr]](http://arxiv.org/abs/2504.11468) （还没细看）主要结论：先SFT会影响后续RL的性能；提了一个适用于多模态的GRPO：包括math输出准确性、bounding box的IoU等、开放式问题上的来自LLM as reward model的打分的多种奖励信号。
1. **More Thinking, Less Seeing? Assessing Amplified Hallucination in Multimodal Reasoning Models** (Arxiv 2025.06) [[paper]](http://arxiv.org/abs/2505.21523) 主要结论：1）base、RL、 SFT+RL的perception越来越差。2）reasoning会导致perception变差的原因包括对visual tk的attn降低；3）SFT+RL相比纯RL，RH-AUC更低，即perception和reasoning无法同时更好。
1. **Hidden in plain sight: VLMs overlook their visual representations** (Arxiv 2025.06) [[paper]](http://arxiv.org/abs/2506.08008) 对于视觉中心任务，标准的视觉评估策略（只采用视觉特征）的效果往往远比转向VLM评估策略后效果好；视觉信息在逐层中并没有发生明显的衰减现象，但是在最后一层中会倾向于发生性能的大幅度下降；对比微调视觉编码器和微调视觉连接器，微调底座LLM的提升最为明显，但仍然对比视觉本身存在一定差距；LLM微调显著提升了模型在关键区域定位并利用视觉表征的能力。
1. **Pixels, Patterns, but No Poetry: To See The World like Humans** (Arxiv 2025.07) [[paper] ](https://www.alphaxiv.org/abs/2507.16863) 提了一个benchmark（TET），包含一些像识别验证码之类的perception任务。对于这些任务，SFT vision encoder是关键，只训LLM几乎没用。
1. **SEEING BUT NOT BELIEVING: PROBING THE DISCONNECT BETWEEN VISUAL ATTENTION AND ANSWER CORRECTNESS IN VLMS** (Arxiv 2025.10) [[paper]](https://www.arxiv.org/pdf/2510.17771) 在qwen、llava、gemma上都发现了：浅层attn关注文本，深层attn关注局部视觉区域；发现了seeing but not believing现象，提出了一个training free的方法让模型关注深层区域：在大约100个样本上找出定位能力最强的top 10%的层，然后用这些层的attn来强调关键的视觉区域。

### 2024 

1. **Towards Interpreting Visual Information Processing in Vision-language Models** (ICLR 2025 Ratings: 8866) 发现object token（图像中对应于物体的token）去掉之后模型掉点最严重。且发现阻塞object token到last token的attention之后掉点最严重。说明在识别物体时，信息直接从object token传递到last token。
1. **Explainable and Interpretable Multimodal Large Language Models: A Comprehensive Survey** (Arxiv Dec 2024) [[paper]](http://arxiv.org/abs/2412.02104) Survey



## Unifying Understanding and Generation

### 2025

1. **OneCAT: Decoder-Only Auto-Regressive Model for Unified Understanding and Generation** (Arxiv 2025.09) [[paper]](http://arxiv.org/abs/2509.03498) 将图像理解、生成、编辑用一个统一的transformer实现。每个transformer block中的FFN有三个，分别处理image、text和discrete visual token。
1. **FutureSightDrive: Thinking Visually with Spatio-Temporal CoT for Autonomous Driving** (NeurIPS 2025) [[paper]](http://arxiv.org/abs/2505.17685) 将VQGAN的词汇表和原本的文本词汇表拼到一起，Qwen-VL学会生成图片，来实现自动驾驶规划。
1. **MathCanvas: Intrinsic Visual Chain-of-Thought for Multimodal Mathematical Reasoning** (Arxiv 2025.10) [[paper]](http://arxiv.org/abs/2510.14958) 训BAGEL去学会如何在几何题上做辅助线。两阶段训练，第一阶段用5.2M数据训BAGEL的generation expert怎么根据instruction生成编辑后的图像，第二阶段用219K数据做SFT，让模型学会

### 2024

1. **Emu3: Next-Token Prediction is All You Need** (Arxiv September 2024) [[paper]](http://arxiv.org/abs/2409.18869) 将文本、图片、视频都转化为token，进行next-token prediction的预训练。能同时做图片视频的生成、视觉-语言理解。训练模型：包含文本encoder（T5）、视觉encoder（ViT-large）和文本decoder（T5，输入为视觉-文本融合特征，输出为文本）。训练资源：8*32G V100。
2. **Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation** (Arxiv Oct 2024) [[paper]](http://arxiv.org/abs/2410.13848) 用一个自回归transformer统一实现多模态的理解和生成任务



## Multimodal ICL

### 2024

1. **Link-Context Learning for Multimodal LLMs** (CVPR 2024) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/html/Tai_Link-Context_Learning_for_Multimodal_LLMs_CVPR_2024_paper.html) 提出一种新的fine-tune MLLM的方法：让context和query具有一定的causal联系，发现能提升模型通过context学习新概念的能力
2. **Can Vision Language Models Learn from Visual Demonstrations of Ambiguous Spatial Reasoning?** (Arxiv Sep 2024) 
3. **Finding Visual Task Vectors** (ECCV 2024) [[paper]](https://arxiv.org/pdf/2404.05729) 
4. **Lever LM: Configuring In-Context Sequence to Lever Large Vision Language Models** (NeurIPS 2024) [[paper]](http://arxiv.org/abs/2312.10104) 先构建一个优质的ICL数据集，然后将该数据集中的image-text对视作token，用CLIP抽取特征作为token embedding，训练一个很小的Transformer（lever-LM）来在该数据集上进行next-token prediction（序列是从query到context这样倒着来的）。测试时，最后给定测试样本，拿lever-LM从该预先挑选好的数据集中预测后续的example来构成context。
5. **Towards Global Optimal Visual In-Context Learning Prompt Selection** (NeurIPS 2024) [[paper]](http://arxiv.org/abs/2405.15279) 没细看，也是做ICL example排序的。base idea都是与测试样本越相似的example效果越好。训练一个用于排序的transformer进行局部排序，再根据局部排序训练一个全局排序信息的向量。
6. **What Factors Affect Multi-Modal In-Context Learning? An In-Depth Exploration** (NeurIPS 2024) [[paper]](http://arxiv.org/abs/2410.20482) 从demo选择、demo顺序和context的构建三个角度探究了影响多模态ICL的因素
7. **What Makes Multimodal In-Context Learning Work?** (CVPR 2024 Workshop on Prompting in Vision) [[paper]](https://arxiv.org/abs/2404.15736) 对Multimodal ICL的实验性分析，主要发现：文本和图像同时输入时，MLLM更依赖文本；目前的MICL基本上是在做从context copy
8. **Task vectors are cross-modal** (ICLR 2025 submission) 

### 2023

1. **What Makes Good Examples for Visual In-Context Learning?** [[paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/398ae57ed4fda79d0781c65c926d667b-Paper-Conference.pdf) 纯vision ICL。找和query最相近的样本来做ICL，类似Link-context learning。



## Reward Model

### 2025

1. **MM-RLHF: The Next Step Forward in Multimodal LLM Alignment** (Arxiv 2025.02) [[paper]](http://arxiv.org/abs/2502.10391) 提出Critique-Based Reward Model, 以及一整套从收集数据到laligenmt的pipeline。





# LLM

## ⭐In-Context Learning

### 2024

1. **Explore Spurious Correlations at the Concept Level in Language Models for Text Classification** (Arxiv Jan 2024) [[paper]](http://arxiv.org/abs/2311.08648) 发现了LLM在文本分类中会依赖的concept-label spurious correlation，提出使用ChatGPT来扩充数据来消除虚假关联。

2. **Positional Information Matters for Invariant In-Context Learning: A Case Study of Simple Function Classes** (ongoing work) [[paper]](Positional Information Matters for Invariant In-Context Learning: A Case Study of Simple Function Classes) 发现模型对于demonstration的permutation invariance或许是ICL OOD的关键。提出使用相同的positional encoding来提升ICL OOD性能。

3. **Simple synthetic data reduces sycophancy in large language models** (Arxiv Feb 2024) [[paper]](http://arxiv.org/abs/2308.03958) LLMs会迎合提问者的观点而罔顾事实。提出合成一些用户的观点和正确性无关的新prompt，然后在这些数据上fine-tune来解决sycophancy问题。

4. **Understanding In-Context Learning in Transformers and LLMs by Learning to Learn Discrete Functions** (ICLR 2024 Oral) [[paper]](https://arxiv.org/abs/2310.03016) 探究transformer在一系列离散任务上的能力。特别地，发现经过预训练的模型相比随机初始化的模型获得了更强的最近邻、disjunction和conjunction的能力。

5. **Batch-ICL: Effective, Efficient, and Order-Agnostic In-Context Learning**  (Arxiv Jan 2024) 发现使用batch ICL，将N个example设置为N个one-shot inference，再把每个inference得到的token做平均，替换到query sample做aggregation最终再预测能带来提升。一个奇特的发现是做aggregation时从某一层往后做性能会突增，在那之前性能接近零。对此解释是transformer的低层是在学语义信息。

6. **RefuteBench: Evaluating Refuting Instruction-Following for Large Language Models** (Arxiv Feb 2024) [[paper]](http://arxiv.org/abs/2402.13463) 评估模型的改变它们的原始输出并遵循和一开始相违背的指令的能力。主要观察：1)大部分模型都会倾向于遵守它们的预训练知识 2)模型很难根据人类后续的反馈泛化到新的问题 3)所有模型都会逐步忘记人类反馈并落回到它们的内部知识里 4)模型是不是第一时间遵守了人类的反馈，对于后续的行为起到关键作用

7. **Function Vectors in Large Language Models** (ICLR 2024) [[paper]](http://arxiv.org/abs/2310.15213) 发现context prompt的最后一个token的隐层表示encode了这个任务的信息，称为function vector（FV）。将其加到zero-shot的prompt上，发现有显著提升。5

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

41. **Learning to grok: Emergence of in-context learning and skill composition in modular arithmetic tasks** (NeurIPS 2024) [[paper]](https://openreview.net/pdf/5737b58d308dafc16130635934df4276a7a574aa.pdf) 探究在modular加法问题上的ICL的OOD能力，并解释了模型组件是如何实现OOD的能力的

42. **Improving In-Context Learning with Small Language Model Ensembles** (NeurIPS 2024 Workshop on Adaptive Foundation Models) [[paper]](http://arxiv.org/abs/2410.21868) 将在下游任务上fine-tune的多个小模型预测的label和confidence与原始label组合到一起，再输给大模型来做ICL，发现可以提升性能

43. **Algorithmic Phases of In-context Learning** (ICLR 2025 Ratings 10 8 6 6) [[paper]](https://openreview.net/pdf?id=XgH1wfHSX8) 在一个马尔可夫链上，识别了ICL的四种推理模式：unigram/bigram-inference/retrieval，这几种模式之间的切换可以解释目前的一系列ICL现象，如task diversity threshold, transient nature, task retreival/task learning, early ascent等。

44. **Can In-context Learning Really Generalize to Out-of-distribution Tasks?** (ICLR 2025) [[paper]](https://arxiv.org/abs/2410.09695) 通过一系列实验分析发现了ICL在OOD任务上只能实现从预训练任务中寻找一个最优任务来拟合下游任务。并从理论上论证了ICL的算法选择机制的存在。

    



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
5. **What and How Does In-Context Learning Learn? Bayesian Model Averaging, Parameterization, and Generalization** (Arxiv Oct 2023) [[paper]](arXiv:2305.19420) 数据生成模型是隐马尔可夫模型（和An Explanation of In-context Learning as Implicit Bayesian Inference这篇如出一辙），理论证明了ICL能先根据context推断一个“任务概念” $\theta$，然后根据 $\theta$ ，query和context来推断y。
6. **Transformers as Statisticians: Provable In-Context Learning with In-Context Algorithm Selection** (NeurIPS 2023) [[paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/b2e63e36c57e153b9015fece2352a9f9-Paper-Conference.pdf) 证明了存在一个L-层线性transformer在线性回归、lasso、ridge问题上error有上界。同时在理论和实验上发现了会自动选择最优预训练知识的现象。
7. **The Learnability of In-Context Learning** (NeurIPS 2023) [[paper]](https://openreview.net/forum?id=f3JNQd7CHM) 证明了当预训练分布包含下游任务的分布的mixuture，ICL能逼近下游任务上的贝叶斯最优分类器。

### 2022

1. **An Explanation of In-context Learning as Implicit Bayesian Inference** (Arxiv 2022) [[paper]](https://arxiv.org/pdf/2111.02080) 早期经典之作，隐马尔可夫模型，证明ICL能实现bayesian-optimal prediction。







## ⭐Reasoning and Test-time compute

### 2025

1. **Benchmarking and Understanding Compositional Relational Reasoning of LLMs** (AAAI 2025) [[paper]](http://arxiv.org/abs/2412.12841) 提出了GAR benchmark来测试模型的Compositional Relational Reasoning能力。发现compositional gap随着模型增大而增大。同时发现了Vicunna-33b存在一些共享的circuit能在不同任务中都发挥作用。

1. **Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach** (Arxiv 2025.02) [[paper]](http://arxiv.org/abs/2502.05171) 提出一种循环结构来提升reasoning能力：类似RNN，循环结构的每一个循环块都接受原始prompt和上一个状态作为输入；循环越多性能越好。

1. **SoftCoT: Soft Chain-of-Thought for Efficient Reasoning with LLMs** (Arxiv 2025.02) [[paper]](http://arxiv.org/abs/2502.12134) 用一个小网络最后一层的隐层表示接上一个projector得到所谓的soft thoughts，将之与问题文本一同输入，后续让做文本CoT。不用像COCONUT那样fine-tune整个LLM，避免了灾难性遗忘导致的掉点。但是提升也比较有限，有点像一个简单的prompt tuning + CoT。

1. **Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solvers** (ICLR 2025) [[paper]](https://openreview.net/forum?id=6aHUmotXaw) 提出了rStar，training-free MCTS，人工定义action  space，reward是self-consistency：找另一个SLM，如果它和policy SLM的某一推理步的输出一致，那么就认为这是一个好的step（被喷可能存在consistent but wrong的情况）。性能提升巨大。

1. **rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking** (Arxiv 2025.01) [[paper]](http://arxiv.org/abs/2501.04519) self-evolution训练：每一轮让policy mode和一个本文提出的process preference model（PPM）做MCTS产生高质量推理路径，然后再用它们来训练policy model和PPM。PPM的提出是由于：很难给一个step打一个衡量好坏的分数，由此训练的PRM可能会不准。因此，提出优化正负样本偏好的方法来训练PPM。正负样本选择方法：每一步选出得分最高的action和最低的action，并强制要求它们分别导向正确和错误的答案，来作为正负样本。

1. **【综述】Test-time Computing: from System-1 Thinking to System-2 Thinking** [[paper]](https://arxiv.org/pdf/2501.02497) test-time reasoning 综述

1. **ReasonFlux: Hierarchical LLM Reasoning via Scaling Thought Templates** [[paper]](https://arxiv.org/pdf/2410.02884?) 

1. **DOTS: Learning to Reason Dynamically in LLMs via Optimal Reasoning Trajectories Search** (ICLR 2025) [[paper]](http://arxiv.org/abs/2410.03864) **核心点：**训练模型自动选择最优的推理方案。与rstar有些类似，都是将任务先从更高层次的动作空间进行规划。**方法：**将解决问题的过程分成analysis、solution、verification三个阶段，每个阶段有不同的选择，也可以选择什么都不做。给定问题-答案对，为每个问题按照success rate搜索出最优的推理方案（algo1）。选出最优方案后用gpt4o结合问题给一个对这个推理方案的解释，然后进行SFT，训练LLM预测推理方案、解释和最终答案。

1. **Don’t Get Lost in the Trees: Streamlining LLM Reasoning by Overcoming Tree Search Exploration Pitfalls** (Arxiv 2025.03) [[paper]](https://arxiv.org/pdf/2502.11183) 发现tree search中会存在大量语义相近的节点

1. **Better Process Supervision with Bi-directional Rewarding Signals** (Arxiv 2025.03) [[paper]](http://arxiv.org/abs/2503.04618) 发现PRM在靠后的step上不准，基于terminal的MC估计在靠前的step上不准。因此设计了一个双头PRM：一个头的监督信号为从开始到第t步的推理正确与否（通过一个大模型标注得到）；另一个头的监督信号是MC估计得到的。两个头分别在这两个目标上和LLM backbone一起训。

1. **Entropy-based Exploration Conduction for Multi-step Reasoning** (Arxiv 2025.03) [[paper]](http://arxiv.org/abs/2503.15848) 某一步的不确定性大，则代表问题有更多可能的解，值得进一步探索。反之则说明探索路径应该更确定。方法：计算每个推理步（一个句子）的沿着所有token的熵，以及每个token沿着词汇表的熵在整个句子的方差，根据这两个指标来决定对于某一推理步，接下来是deepen、expand还是stop。

1. **DAPO: An Open-Source LLM Reinforcement Learning System at Scale** (Arxiv 2025.03) [[paper]](http://arxiv.org/abs/2503.14476) 对GRPO的改进

1. **From Chaos to Order: The Atomic Reasoner Framework for Fine-grained Reasoning in Large Language Models** (Arxiv 2025.03) [[paper]](http://arxiv.org/abs/2503.15944) 参考o1的推理特征，定义macro-action：分析前提条件和问题/进行推理（假设生成和验证）/终止，让模型自己选这些macro-action。同时设计了一个让一个check对多种细粒度的错误类型进行分别检测。    

1. **【benchmark】Prmbench: A fine-grained and challenging benchmark for process-level reward models** [[paper]](https://www.google.com/search?q=Prmbench%3A+A+fine-grained+and+challenging+benchmark+for+process-level+reward+models&oq=Prmbench%3A+A+fine-grained+and+challenging+benchmark+for+process-level+reward+models&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIGCAEQRRg60gEHNDAyajBqN6gCCLACAfEFC2miFWHXxNE&sourceid=chrome&ie=UTF-8) 将PRM对于reasoning step的评价能力划分为：评价推理过程是否冗余、推理过程是否错误、鲁棒性（是否能察觉到关键前提的丢失、陈述中的陷阱、对于多个正确的解答能否保持评价一致）。

1. **Inference-Time Scaling for Generalist Reward Modeling** [[paper]](http://arxiv.org/abs/2504.02495) 针对所有领域而不是单一领域训练scalable的reward model。方法为GRM （Generate Reward Modeling）通过大量采样critique并以此生成reward score，来实现reward model的test-time scaling。

1. **Heimdall: test-time scaling on the generative verification** (Arxiv 2025.04) [[paper]](http://arxiv.org/abs/2504.10337) 生成式的RM，用PPO训练。

1. **Genius: A Generalizable and Purely Unsupervised Self-Training Framework For Advanced Reasoning** (Arxiv 2025.04) [[paper]](http://arxiv.org/abs/2504.08672) 完全不依赖任何RM和监督信号，只靠问题进行自监督训练。某一步的奖励信号为从该步开始的剩余步的mean log prob。

1. **Step-by-Step Reasoning for Math Problems via Twisted Sequential Monte Carlo** (ICLR 2025) [[paper]](http://arxiv.org/abs/2410.01920) 方法：如何推理：在每个推理步t，让policy model产生N个下一步。利用训练好的value function给N个步打分，然后根据打分重新sample该步（line 18），之后到t+1，再让policy model在经过resample的第t步的基础上再生成下一步；如何训练value function（一个network）：loss function的优化目标为减小value function估计的分布和ground-truth分布之间的KL散度，其实让value function对于不同solution的某一步的打分接近outcome reward（每一步的监督信号相同，都是拟合outcome reward）

1. **Understanding R1-Zero-Like Training: A Critical Perspective** (Arxiv 2025.03) [[paper]](https://www.alphaxiv.org/abs/2503.20783) base model已经有aha moment。由于normalization，GRPO训练会倾向于输出更短的正确回答和更长的错误回答。

1. **【🚀RL】Group-in-Group Policy Optimization for LLM Agent Training** (Arxiv 2025.05) [[paper]](http://arxiv.org/abs/2505.10978) agent领域的文章。setting是每一步和环境交互之后都能立即得到环境给该step的score反馈。方法：在不额外增加GRPO rollout的情况下，合并相同的状态（对于agent领域，状态可能指所位于的网页页面，因此可以通过hash直接很快地合并），并把相同状态的下一步组成一个group进行GRPO训练。group内每个下一步的reward就是它们各自后续的的step-wise环境reward的累加。

1. **【🚀RL】S-GRPO: Early Exit via Reinforcement Learning in Reasoning Models** (Arxiv 2025.05) [[paper] ](https://arxiv.org/pdf/2505.07686) 主要解决GRPO导致大量无用思考的问题。RL 时每次只生成一条链，然后随机从中间步开始，停止思考，直接给出答案。对于正确的response，退出思考的位置越晚，reward越低，从而鼓励简洁的思考。

1. **【🚀RL】Spurious Rewards: Rethinking Training Signals in RLVR** (Arxiv 2025.05) [[paper]](Spurious Rewards: Rethinking Training Signals in RLVR) 核心发现：对于qwen系列模型，使用随机/错误的reward进行RLVR也能带来显著提升；对于其他模型基本不行；原因分析（fig6、7）：对于code本身很强的模型如qwen2.5-math，虚假reward能带来推理模式的转变：anguage->code，从而导致性能提升）；对于code不行的如qwen2.5，wrong reward会导致language->code，从而带来提升。即，虚假reward能鼓励模型用自己擅长的方式推理从而获得提升。

1. **【🚀RL】Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning** (Arxiv 2025.06) [[paper]](http://arxiv.org/abs/2506.01939) 少量的high-entropy token上训练是获得多样的推理路径的关键，且有不错的scalability。还发现在其余大量的low-entropy token上训会导致性能下降。

1. **【🚀RL】The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning** (Arxiv 2025.06) [[paper]](https://www.alphaxiv.org/abs/2506.01347) 发现在RL中，单独抑制错误回复能在pass@k up to 256都超过base，达到或赶超GRPO；而只强化正确回复能提升pass@1，但是pass@k会降低。

1. **【🚀RL】The Hallucination Dilemma: Factuality-Aware Reinforcement Learning for Large Reasoning Models** (Arxiv 2025.05) [[paper]](http://arxiv.org/abs/2505.24630)

1. **【Latent CoT】CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation** (Arxiv 2025.05) [[paper]](http://arxiv.org/abs/2502.21074) 性能堪比正常cot的latent cot，做法是对齐teacher model（正常cot）的"The answer is:"的":"与student（latent）cot的":"的hidden states，而不对latent cot做额外的限制。

1. **【Latent CoT】Think Silently, Think Fast: Dynamic Latent Compression of LLM Reasoning Chains** (Arxiv 2025.06) [[paper]](https://www.alphaxiv.org/abs/2505.16552v4)

1. **【Understanding】Part I: Tricks or Traps? A Deep Dive into RL for LLM Reasoning** (Arxiv 2025.08) [[paper]](http://arxiv.org/abs/2508.08221) 在Qwen3 4B/8B、base/aligned上验证了batch/group normalization、sequence/token-level loss aggregation、clip-higher等因素在不同组合下对RL训练的dynamic和performance的影响

1. **【Latent CoT】Soft Thinking: Unlocking the Reasoning Potential of LLMs in Continuous Concept Space** (Arxiv 2025.05) [[paper]](http://arxiv.org/abs/2505.15778) 提出了一种training-free的soft thinking：用模型预测的next-token概率分布去加权input embedding，作为下一位置的输入（这是针对7B以上模型input embedding layer 和 lm_head的不share weight的问题：说明input embedding和last hidden state不在同一空间，像COCONUT那样直接输入回去会导致输入OOD）。性能可以超过token CoT.

1. **【Latent CoT】LLMs are Single-threaded Reasoners: Demystifying the Working Mechanism of Soft Thinking** (Arxiv 2025.08) [[paper]](http://arxiv.org/abs/2505.16552) 实验上发现soft thinking的性能、模型输出概率分布、logit lens的解码词汇都很像greedy。提出采用Gumbel-softmax，将模型原先的输出概率进行扰动，然后再soft thinking，性能就能超过vanilla cot。

1. **【🔧SFT+🚀RL】On-Policy RL Meets Off-Policy Experts: Harmonizing Supervised Fine-Tuning and Reinforcement Learning via Dynamic Weighting** 发现SFT时模型的性能变化趋势：性能下降-性能恢复-过拟合。提出了RL和SFT同时进行的策略：1）通过一个总的、慢慢decay的weight从SFT逐步过渡到RL；2）对SFT的loss进行token-wise reweighting：模型预测概率过高和过低的都会降低weight（概率过低的会导致policy shift太严重；概率过高的会限制RL探索）

1. **【Latent CoT】Soft Tokens, Hard Truths** (Arxiv 2025.09) 用RL来训Latent thinking，不需要discrete cot监督

1. **【Latent CoT】SIM-CoT: Supervised Implicit Chain-of-Thought** (Arxiv 2025.09) 对每个latent token直接加监督：单独将第k个latent作为prefix输入进一个独立的支路（仍然是LLM作为backbone）去预测第k步的CoT文本。

1. **【🔧SFT，实验效果显著】On the Generalization of SFT: a Reinforcement Learning Perspective with Reward Rectification** [[paper]](https://www.alphaxiv.org/abs/2508.05629) 将SFT的loss写成RL的形式后，SFT可以视作：当模型输出严格=专家序列时reward才为1（奖励稀疏）、且乘以了 $\frac{1}{\pi_\theta(y^*|x)}$ 因子（会导致policy当对专家action给出低概率时，policy grad被放大，作者认为这会导致过拟合）。方法：对每个token的loss乘以 $\pi_\theta(y^*_t|y^*_{t-1},x)$

1. **【🚀RL】Group Sequence Policy Optimization** (Arxiv 2025.11) [[paper]](http://arxiv.org/abs/2511.20347) 提出GSPO：将重要性采样ratio从token-wise计算改为整个sequence的log sum exp，同一序列内所有token使用相同的权重，避免了token-wise的ratio的高方差。

1. **【🚀RL】Soft Adaptive Policy Optimization** (Arxiv 2025.11) [[paper]](http://arxiv.org/abs/2511.20347) 提出SRPO：

1. **【Latent CoT】Seek in the Dark: Reasoning via Test-Time Instance-Level Policy Gradient in Latent Space** [[paper]](http://arxiv.org/abs/2505.13308) 用self-reward作为奖励信号，在测试时通过REINFORCE算法迭代优化生成的latent，取得了相比discrete CoT的显著提升。

1. **【🚀RL, step-wise reward】Segment Policy Optimization: Effective Segment-Level Credit Assignment in RL for Large Language Models** (NeurIPS 2025)[[paper]](http://arxiv.org/abs/2505.23564) 

1. **【Analysis】On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models** (Arxiv 2025.12) [[paper]](https://arxiv.org/abs/2512.07783v1) 构建合成任务训练集，探究了不同难度的数据上进行RL的影响。发现：对于OOD任务，仅有ID边缘（能答对部分）进行RL才能获得提升；当基模型没有OOD能力时，RL没用，但混入至少1%的数据时，RL就能提升OOD了；引入过程奖励能提升OOD能力。

1. **RLAR: ** (Arxiv 2025.12) [[paper]]()

1. **【🚀RL, step-wise reward】Supervised Reinforcement Learning: From Expert Trajectories to Step-wise Reasoning** (Arxiv 2025.10) [[paper]](http://arxiv.org/abs/2510.25992) 提出SRL，RL rollout时让policy基于专家序列的前k-1步开始，生成下一步k，计算policy生成的第k步与专家第k步的相似度作为reward。

### 2024 

1. **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models** (Arxiv April 2024) 提出GRPO （Group Relative Policy Optimization）

2. **Scaling LLM Test-time Compute Optimally can be More Effective than Scaling Model Parameters**  [[paper]](https://arxiv.org/pdf/2408.03314) 研究了两种scaling test-time compute的策略：1）基于verifier（process reward model）的；2）基于模型的self-revision的。发现了根据具体任务（不同难度）来选择最优scaling策略能在达到相同性能时相比best-of-N降低四倍计算量

3. **Training Large Language Model to Reason in a Continuous Latent Space** (COCONUT Arxiv Dec 2024, ICLR 2025 被拒，主要是因为相比于普通CoT会在GSM8K上掉点) [[paper]](https://openreview.net/forum?id=tG4SgayTtk) 将reasoning step的某些中间步从word embedding 替换为该token的last hidden state。 

4. **Beyond Examples: High-level Automated Reasoning Paradigm in In-Context Learning via MCTS** (Arxiv 2024.11) [[paper]](http://arxiv.org/abs/2411.18478) 用了rStar的self-consistent reward和人工定义的action space，但是加入了thought card的技术。性能和rstar差不多，但是计算代价小了很多，因为测试时不用MCTS了，只需要从seed dataset中找出card即可。

5. **ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search** (NeurIPS 2024) [[paper]](https://arxiv.org/pdf/2406.03816) 同时训练policy model和一个process reward model（一个LLM-based打分模型）。第k个推理步的process reward $v_k$的监督信号为：1）如果该步距离最终答案越近，$v_k$越大；2）如果最终答案是错的，$v_k$​为0. 在用MCTS生成推理路径的过程中，也使用value model的打分指导生成，每次只探索得分最高的路径。也就是说，MCTS路径生成和模型训练是交替迭代进行的。

   MCTS的过程为（原文algo2），以下过程重复T次：

   1. 根据UCB选一个节点C_select
   2. 将C_select用policy model展开成b个子节点（b个推理branch），用value model选出得分最高的C子节点C’
   3. 从C’开始再推理m步，记录下最高得分并更新V_C’的得分
   4. 更新从根节点到所选的起始节点C_select这条路上的所有结点的访问次数和得分，每个节点得分的更新方法：eq36，用孩子更新parent


6. **Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning** (Arxiv 2024.05) [[paper]](http://arxiv.org/abs/2402.05808) 思想：让模型基于已有的推理链的中间步进行后续推理，降低搜索到正确答案的难度。做法：从T-1开始选择起始步进行policy gradien的计算，逐渐将起始步往前推，慢慢增大学习难度。
6. **Calibrating Reasoning in Language Models with Internal Consistency** (NeurIPS 2024) [[paper]](https://arxiv.org/pdf/2405.18711) 发现模型在给出错误回答时中间各层的预测一致性较低
6. **V-STaR: Training Verifiers for Self-Taught Reasoners** (COLM 2024) [[paper]](https://openreview.net/pdf?id=stmqBSW2dV) 用模型生成的正确和错误回答通过DPO训练一个verifier，测试时用这个verifier来给不同回答打分
6. **Mindstar: Enhancing math reasoning in pre-trained llms at inference time** (Arxiv 2024.05) [[paper]](https://arxiv.org/abs/2405.16265) PRM+tree search。LLM as PRM, PRM的输入为当前所有推理步和下一推理步。
6. **LLaMA-Berry: Pairwise Optimization for O1-like Olympiad-Level Mathematical Reasoning** (Arxiv 2024.11) [[paper]](https://arxiv.org/pdf/2410.02884) MCTS+pair-wise preference reward model (PPRM)。一个节点是一个完整的解决方案（而不是一个推理步）。先利用现有的preference数据集（PRM800K等）训练一个PPRM（一个2B LLM），能够对两个solution输出偏好。每个节点的打分方式：局部得分（反映某节点与孩子节点的win rate）和全局得分（反映某节点在所有node里的排名）的加权平均。
6. **Stepwise Self-Consistent Mathematical Reasoning with Large Language Models** (Arxiv2024.02) [[paper]](http://arxiv.org/abs/2402.17786) consistency的计算方法是TF (Term Frequency) - IDF (Inverse Document Frequency) vector，一种基于词频统计的文档相似度计算方法（只能反映词频上的相似度，反应不了语义相似度）
6. **Universal Self-Consistency for Large Language Models** (ICML 2024 ICL workshop) [[paper]](https://openreview.net/pdf?id=LjsjHF7nAN) 针对self-consistency难以提取答案的问题，prompt一个gpt-3.5来从一系列回答中选取最consistent的那一个。

### 2023

1. **Self-Consistency Improves Chain of Thought Reasoning in Language Models ** (ICLR 2023) [[paper]](https://arxiv.org/pdf/2203.11171) Self-consistency	
2. **Self-Refine: Iterative Refinement with Self-Feedback** (Arxiv 2023.05) [[paper]](https://arxiv.org/pdf/2303.17651) Self-refine
3. **Large Language Models Cannot Self-Correct Reasoning Yet** (ICLR 2024) [[paper]](https://arxiv.org/abs/2310.01798) Self-correct 有时会失败



## Distillation

### 2024

1. **On-Policy Distillation of Language Models** (ICLR 2024) [[paper]](https://proceedings.iclr.cc/paper_files/paper/2024/file/5be69a584901a26c521c2b51e40a4c20-Paper-Conference.pdf?utm_source=chatgpt.com) 提出了on-policy distillation



## Alignment

### 2024

1. **LET’S VERIFY STEP BY STEP** (ICLR 2024) 发现PRM比ORM好
1. **The Unlocking Spell on Base LLMs: Rethinking Alignment via In-Context Learning** (ICLR 2024) [[paper]](https://openreview.net/forum?id=wxJ0eXwwda) 通过ICL，添加system prompt和风格化的输出，实现只用很少的样本（3个）来提升LLM alignment。
1. **The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions** (Arxiv April 2024) [[paper]](http://arxiv.org/abs/2404.13208) 构造训练数据来教模型学习不同指令的优先级来防御有害指令。具体方法为，对于不同的任务，分别构造与最高指令aligned/misaligned的指令，然后训练模型输出期望的回答。

### 2023

1. **(DPO) Direct Preference Optimization: Your Language Model is Secretly a Reward Model** (NeurIPS 2023) [[paper]](http://arxiv.org/abs/2305.18290)

### 2017

1. **(RLHF) Deep reinforcement learning from human preferences** (NeurIPS 2017) [[paper]](http://arxiv.org/abs/1706.03741)
2. **(PPO) Proximal Policy Optimization Algorithms** (Arxiv 2017) [[paper]](http://arxiv.org/abs/1707.06347)



## Interpretability

### 2025

1. **Latent Space Chain-of-Embedding Enables Output-free LLM Self-Evaluation** (ICLR 2025) [[paper]](http://arxiv.org/abs/2410.13640) 定义LLM的从第一层到最后一层的各层的表示为CoE，发现回答正确时CoE相邻状态的magnitude差距较大，角度差距较小；而回答错误时正好相反。由此提出了一个指标用于在无label情况下判断模型输出的对错。

### 2024

1. **LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations** (ICLR 2025 Ratings:8666) [[paper]](https://openreview.net/forum?id=KRnsX5Em3W) 用一个线性probe来根据模型中间层表示判断模型输出的正确与否。然后让LLM对同一个问题生成多个答案，并用该分类器筛选出正确概率最高的答案，发现能相比原本的答案正确率更高。

1. **Insights into LLM Long-Context Failures: When Transformers Know but Don't Tell** (EMNLP 2024 Findings) [[paper] ](http://arxiv.org/abs/2406.14673)用一个线性probe来根据模型中间层表示来直接预测问题的答案。发现probe acc比直接生成的acc好。

1. **Does Representation Matter? Exploring Intermediate Layers in Large Language Models** (NeurIPS 2024 workshop) [[paper]](http://arxiv.org/abs/2412.09563) LLM的中间层下游性能比最后一层好。探究了Prompt Entropy、Curvature等representation quality的指标和下游acc的关系。

   

## Other

### 2024

1. **Model Editing with Canonical Examples** [[paper]](http://arxiv.org/abs/2402.06155) 提出了一个新任务：让模型学习几个特定的文本例子，以实现某些纠正，同时还不能让模型改变很多。

1. **Evaluating Large Language Models at Evaluating Instruction Following** [[paper]](https://openreview.net/forum?id=tr0KidwPLc) (ICLR 2024) 

1. **Not all Layers of LLMs are Necessary during Inference** (Arxiv April 2024) 训练一个对LLM中间层feature的分类器判断是否应该早停来获取早停层数，来加速LLM推理。还发现中间层预测的top prob和top prob-second top prob在各个任务上都呈现出随着层数加深而增加并逐渐稳定的趋势（但在不同任务上层数不一样）。[[paper]](http://arxiv.org/abs/2403.02181)

1. **Demonstrating Mutual Reinforcement Effect through Information Flow** (Arxiv March 2024) [[paper]](https://arxiv.org/pdf/2403.02902) 研究了同时进行word分类和text分类的MRE（Mutual Reinforcement Effect）任务，也观察到了anchor那篇中的三种attention activation随layer的分布趋势。

1. **A Theoretical Understanding of Self-Correction through In-context Alignment** (Arxiv May 2024) [[paper]](http://arxiv.org/abs/2405.18634) 理论分析transformer中的各个模块在self-correction中发挥的作用

1. **Mechanics of Next Token Prediction with Self-Attention** (AISTATS 2024) [[paper]](https://proceedings.mlr.press/v238/li24f.html) 构造了一个graph来描述next token prediction任务，在简化setting下理论分析出last token更倾向于给更经常作为label的token分配更高的attention。

1. **The pitfalls of next-token prediction** (Arxiv April 2024) [[paper]](http://arxiv.org/abs/2403.06963) 指出了自回归模型的缺陷：错误滚雪球效应和在一个单一token路径上只能学出一个类似induction head的shortcut模型

1. **A Law of Next-Token Prediction in Large Language Models** (Arxiv Aug 2024) [[paper]](https://arxiv.org/pdf/2408.13442v1)

1. **SEMIEVOL: Semi-supervised Fine-tuning for LLM Adaptation** (Arxiv Oct 2024) [[paper]](https://arxiv.org/pdf/2410.14745) 提出了半监督fine-tuning框架SEMIEVOL。

   

### 2023

1. **Instruction-following Evaluation through Verbalizer Manipulation** (Arxiv July 2023) [[paper]](http://arxiv.org/abs/2307.10558) 发现LLM遵循flipped-label instructions的能力很差，说明ICL可能只是直接利用了预训练语料的知识，而不是学习了context。即使是强如GPT-4的模型也不能很好地遵循flipped-label instructions。
2. **Reasoning or Reciting? Exploring the Capabilities and Limitations of Language Models Through Counterfactual Tasks** (Arxiv Aug 2023) [[paper]](http://arxiv.org/abs/2307.02477) 一些主要发现：①模型在counterfactual的setting中性能会变差，且setting和常见的、符合事实的setting相差越远，性能越差，说明了模型可能的记忆现象。②在算术任务上，ICL能提升counterfactual（不同进制的计算）性能，但和default setting的差距难以抹平。
3. **Can the Inference Logic of Large Language Models be Disentangled into Symbolic Concepts?** (Arxiv Apr 2023) [[paper]](https://arxiv.org/abs/2304.01083) 提出了一种empirical的指标来衡量输入句子里的某些词和词组对某一特定输出的决定程度。
4. **Contrastive Chain-of-Thought Prompting** (Arxiv Nov 2023) [[paper]](http://arxiv.org/abs/2311.09277) 使用对比CoT，即一个正确CoT搭配一个错误CoT能相比常规的CoT带来提升.

### 2022

1. **Same Pre-training Loss, Better Downstream: Implicit Bias Matters for Language Models** [[paper]](https://arxiv.org/pdf/2210.14199.pdf)

### 2021

**LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS** 将对模型权重矩阵的更新限制为低秩矩阵乘积$BA$的形式，极大减少了pre-trained model迁移到新任务的代价（不用fine-tune所有参数） [[paper]](https://arxiv.org/abs/2106.09685)

### 2019

1. **Are Sixteen Heads Really Better than One?** (NeurIPS 2019) [[paper]](https://proceedings.neurips.cc/paper_files/paper/2019/hash/2c601ad9d2ff9bc8b282670cdd54f69f-Abstract.html) 在某些层上，只用一个head性能也能保持不变。同时提出了使用attention梯度来衡量head的重要性，提出了剪枝策略。







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





## Other

### 2024

1. **VisionLLaMA: A Unified LLaMA Interface for Vision Tasks** (Arxiv Mar 2024) [[paper]](https://arxiv.org/pdf/2403.00522) Vision LLaMa
1. **Are We on the Right Way for Evaluating Large Vision-Language Models?** (Arxiv April 2024) [[paper]](http://arxiv.org/abs/2403.20330) 现有的vision-language数据集质量不够好，很多问题都是只看语言部分就能解决，或者问题在类似的训练语料中见过，根本不需要图片；构建了一个高质量的vision-language数据集。
   1. **Visual Instruction Tuning** (NeurIPS 2023) [[paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/6dcf277ea32ce3288914faf369fe6de0-Paper-Conference.pdf) LLaVA



# Agents

### 2026

1. **AI Agent Systems: Architectures, Applications, and Evaluation** [[paper]](http://arxiv.org/abs/2601.01743) 综述
2. **Unlocking Implicit Experience: Synthesizing Tool-Use Trajectories from Text** [[paper]](http://arxiv.org/abs/2601.10355) 美团提出了一套从互联网原始文本合成多轮工具调用序列并定义工具的框架：
   1. **粗筛：**从原始文本筛选出带有多步操作的；
   2. **提取：**模型从中提取工作流和工具定义；
   3. **序列合成：**用一个strong teacher（GLM4.6）基于工作流和工具来合成序列，每条序列为 $[s, (u_t,a_t,o_t)]$ ，$s$ 为sys prompt、 $u_t$  为user query、 $a_t$ 为模型action、 $o_t$ 为observation 
   4. **提高序列复杂度（见A.4）：** 通过让teacher做refinement实现。增加sys prompt中的限制条件、提高用户要求的模糊度和复杂性、提高assistant回复质量、提高环境复杂度等 。**ablation显示这部分提升显著**




### 2025

1. **Multi-modal Agent Tuning: Building a VLM-Driven Agent for Efficient Tool Usage** (ICLR 2025 Spotlight) T3-Agent。提了一套数据合成策略：先让gpt4o-mini合成文本问题（没有file），然后让其根据这个问题去找files（图片等），然后用gpt4o-mini作为agent合成SFT数据来fine-tune Qwen2-VL-7B
2. **Step-DeepResearch Technical Report** [[paper]](http://arxiv.org/abs/2512.20491) (Arxiv 2025.12) Search Agent
