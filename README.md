# My LLM
All about large language models

## My Practice
- [my-alpaca](https://github.com/l294265421/my-alpaca) reproduce alpaca
- [multi-turn-alpaca](https://github.com/l294265421/multi-turn-alpaca) train alpaca with multi-turn dialogue datasets
- [alpaca-rlhf](https://github.com/l294265421/alpaca-rlhf) train multi-turn alpaca with RLHF (Reinforcement Learning with Human Feedback) based on DeepSpeed Chat
- [my-autocrit](https://github.com/l294265421/my-autocrit) experiments using autocrit
- [try-large-models](https://github.com/l294265421/try-large-models) try large models
- [my-rl](https://github.com/l294265421/my-rl) learn reinforcement learning using tianshou

## My Articles
- [ChatGPT-Techniques-Introduction-for-Everyone](https://github.com/l294265421/ChatGPT-Techniques-Introduction-for-Everyone)

## Pre-train
### Models
- T5
  - [Paper](./papers/pre-train/models/2020-JMLR-Exploring%20the%20Limits%20of%20Transfer%20Learning%20with%20a%20Unified%20Text-to-Text%20Transformer.pdf)
  - Architecture
    - Encoder-Decoder
- GPT
  - Paper
    - [GPT](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
    - [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
    - [GPT-3](https://arxiv.org/pdf/2005.14165.pdf)
- GPT-Neo
- GPT-J-6B
- Megatron-11B
- Pangu-a-13B
- FairSeq
- GLaM
  - [Paper](./papers/pre-train/models/2022-ICML-GLaM-%20Efficient%20Scaling%20of%20Language%20Models%20with%20Mixture-of-Experts.pdf)
- LaMDA
  - [Paper](./papers/pre-train/models/2022-LaMDA-%20Language%20Models%20for%20Dialog%20Applications.pdf)
- JURASSIC-1
  - [Paper](https://uploads-ssl.webflow.com/60fd4503684b466578c0d307/61138924626a6981ee09caf6_jurassic_tech_paper.pdf)
- MT-NLG
  - [Paper](https://arxiv.org/pdf/2201.11990.pdf)
- ERNIE
  - Paper
    - [ERNIE](https://arxiv.org/pdf/1904.09223.pdf)
    - [ERNIE 2.0](https://arxiv.org/pdf/1907.12412.pdf)
    - [ERNIE 3.0](./papers/pre-train/models/2021-ERNIE%203.0-%20LARGE-SCALE%20KNOWLEDGE%20ENHANCED%20PRE-TRAINING%20FOR%20LANGUAGE%20UNDERSTANDING%20AND%20GENERATION.pdf)
- Gopher
  - [Paper](./papers/pre-train/models/2021-Scaling%20Language%20Models-%20Methods,%20Analysis%20&%20Insights%20from%20Training%20Gopher.pdf)
  - Conclusion
    - Gains from scale are largest in areas such as reading comprehension, fact-checking, and the identification of toxic language, but logical and mathematical reasoning see less benefit.
- Chinchilla
  - [Paper](./papers/pre-train/models/2022-Training%20Compute-Optimal%20Large%20Language%20Models.pdf)
  - Conclusion
    - We find that current large language models are significantly under trained, a consequence of the recent focus on scaling language models whilst keeping the amount of training data constant.
    - we find that for compute-optimal training, the model size and the number of training tokens should be scaled equally: for every doubling of model size the number of training tokens should also be doubled.
- PaLM
  - [Paper](./papers/pre-train/models/2022-PaLM-%20Scaling%20Language%20Modeling%20with%20Pathways.pdf)
  - Architecture
    - Decoder
- PaLM 2
  - [Blog](https://ai.google/discover/palm2)
  - [PaLM 2 Technical Report](https://ai.google/static/documents/palm2techreport.pdf)
- OPT
  - [Paper](./papers/pre-train/models/2022-OPT-%20Open%20Pre-trained%20Transformer%20Language%20Models.pdf)
  - Architecture
    - Decoder
- Gpt-neox
  - [Paper](./papers/pre-train/models/2022-Gpt-neox-20b-%20An%20open-source%20autoregressive%20language%20model.pdf)
  - [GitHub](https://github.com/EleutherAI/gpt-neox)
  - Architecture
    - Decoder
- BLOOM
  - [Paper](./papers/pre-train/models/2023-BLOOM-%20A%20176B-Parameter%20Open-Access%20Multilingual%20Language%20Model.pdf)
  - Architecture
    - Decoder
- LLaMA
  - [Paper](./papers/pre-train/models/2023-LLaMA-%20Open%20and%20Efficient%20Foundation%20Language%20Models.pdf)
  - [Model](https://huggingface.co/decapoda-research)
  - Architecture
    - Decoder
- GLM
  - Paper
    - 2022-ACL-GLM- General Language Model Pretraining with Autoregressive Blank Infilling [paper](./papers/pre-train/models/2022-ACL-GLM-%20General%20Language%20Model%20Pretraining%20with%20Autoregressive%20Blank%20Infilling.pdf)
      - [GitHub](https://github.com/THUDM/GLM)
    - 2023-ICLR-GLM-130B- An Open Bilingual Pre-trained Model [paper](./papers/pre-train/models/2023-ICLR-GLM-130B-%20An%20Open%20Bilingual%20Pre-trained%20Model.pdf)
      - [GitHub](https://github.com/THUDM/GLM-130B)
      - Architecture
        - Autoregressive Blank Infilling
- BloombergGPT
  - [Paper](./papers/pre-train/models/2023-BloombergGPT-%20A%20Large%20Language%20Model%20for%20Finance.pdf)
- MOSS
  - [GitHub](https://github.com/OpenLMLab/MOSS)
- OpenLLaMA: An Open Reproduction of LLaMA
  - [GitHub](https://github.com/openlm-research/open_llama)
- dolly
  - [GitHub](https://github.com/databrickslabs/dolly)
- panda
  - [GitHub](https://github.com/dandelionsllm/pandallm)
  - [Paper](./papers/pre-train/models/2023-Panda%20LLM-%20Training%20Data%20and%20Evaluation%20for%20Open-Sourced%20Chinese%20Instruction-Following%20Large%20Language%20Models.pdf)
- WeLM
  - [Paper](./papers/pre-train/models/2022-WeLM-%20A%20Well-Read%20Pre-trained%20Language%20Model%20for%20Chinese.pdf)
- Baichuan
  - [Baichuan-7B](https://github.com/baichuan-inc/Baichuan-7B)
  - [Baichuan-13B](https://github.com/baichuan-inc/Baichuan-13B)


### Survey
- 2023-A Survey of Large Language Models [[paper](https://arxiv.org/abs/2303.18223)]

### Methods
#### Max Sequence Length
- Blog
  - [Transformer升级之路：7、长度外推性与局部注意力](https://spaces.ac.cn/archives/9431)
- Paper
  - 2023-Scaling Transformer to 1M tokens and beyond with RMT [[paper](./papers/pre-train/methods/max_sequence_length/2023-Scaling%20Transformer%20to%201M%20tokens%20and%20beyond%20with%20RMT.pdf)]
    - 2022-NIPS-Recurrent Memory Transformer [[paper](./papers/pre-train/methods/max_sequence_length/2022-NIPS-Recurrent%20Memory%20Transformer.pdf)]
  - 2022-Parallel Context Windows Improve In-Context Learning of Large Language Models [[paper](./papers/pre-train/methods/max_sequence_length/2022-Parallel%20Context%20Windows%20Improve%20In-Context%20Learning%20of%20Large%20Language%20Models.pdf)]

#### Position
- Rotary
- ALiBi [[paper](./papers/pre-train/methods/position/2022-ICLR-Train%20short,%20test%20long-%20Attention%20with%20linear%20biases%20enables%20input%20length%20extrapolation.pdf)]
- Survey
  - [让研究人员绞尽脑汁的Transformer位置编码](https://zhuanlan.zhihu.com/p/352898810)

#### Normalization
- RMSNorm
- Layer Normalization
  - Pre-LN
  - Post-LN
  - Sandwich-LN
  - DeepNorm

#### Activation Function
- SwiGLU
- GeLUs
- Swish

#### Tokenizer
- BPE [paper](./papers/pre-train/methods/tokenizer/2016-ACL-Neural%20Machine%20Translation%20of%20Rare%20Words%20with%20Subword%20Units.pdf)

#### Interpretability
- [Transformer Circuits Thread](https://transformer-circuits.pub/)

#### LR Scheduler

#### 
- 2020-Scaling Laws for Neural Language Models [[paper](https://arxiv.org/pdf/2001.08361v1.pdf)]

## Fine-tune
### Models
#### General
- T0
  - [Paper](https://arxiv.org/pdf/2110.08207.pdf)
- FLAN
  - [Paper](./papers/fine-tune/models/2022-iclr-FINETUNED%20LANGUAGE%20MODELS%20ARE%20ZERO-SHOT%20LEARNERS.pdf)
  - [GitHub](https://github.com/google-research/flan)
- Flan-LM
  - [Paper](https://arxiv.org/pdf/2210.11416.pdf)
- BLOOMZ & mT0
  - [Paper](https://arxiv.org/pdf/2211.01786.pdf)
- ChatGPT
  - [Blog](https://openai.com/blog/chatgpt)
- Alpaca: A Strong, Replicable Instruction-Following Model
  - [Site](https://crfm.stanford.edu/2023/03/13/alpaca.html)
  - [GitHub](https://github.com/tatsu-lab/stanford_alpaca#fine-tuning)
- Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90%* ChatGPT Quality
  - [GitHub](https://github.com/lm-sys/FastChat)
  - [Site](https://vicuna.lmsys.org/)
  - [Online Demo](https://chat.lmsys.org/)
- Koala: A Dialogue Model for Academic Research
  - [Blog](https://bair.berkeley.edu/blog/2023/04/03/koala/)
  - GitHub
      - [Koala_data_pipeline](https://github.com/young-geng/koala_data_pipeline)
      - [Koala Evaluation Set](https://github.com/arnav-gudibande/koala-test-set)
- alpaca-lora
  - [GitHub](https://github.com/tloen/alpaca-lora)
- ChatGLM-6B
  - [GitHub](https://github.com/THUDM/ChatGLM-6B)
  - [Blog](https://chatglm.cn/blog)
- Firefly
  - [GitHub](https://github.com/yangjianxin1/Firefly)
- thai-buffala-lora-7b-v0-1
  - [Model](https://huggingface.co/Thaweewat/thai-buffala-lora-7b-v0-1)
- multi-turn-alpaca
  - [GitHub](https://github.com/l294265421/multi-turn-alpaca)
- Open-Assistant
  - [Site](https://open-assistant.io/zh)
  - [GitHub](https://github.com/LAION-AI/Open-Assistant)
  - [Paper](./papers/2023-OpenAssistant%20Conversations%20-%20Democratizing%20Large%20Language%20Model%20Alignment.pdf)

#### Chinese
- Chinese-ChatLLaMA
  - [GitHub](https://github.com/ydli-ai/Chinese-ChatLLaMA)
  - Blog
    - [训练中文LLaMA大规模语言模型](https://zhuanlan.zhihu.com/p/612752963)
    - [ChatLLaMA：用指令微调训练中文对话大模型](https://zhuanlan.zhihu.com/p/616748134)
- BELLE
  - [GitHub](https://github.com/LianjiaTech/BELLE)
- Chinese-LLaMA-Alpaca
  - [GitHub](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
- Luotuo-Chinese-LLM
  - [GitHub](https://github.com/LC1332/Luotuo-Chinese-LLM)
- Chinese-Vicuna
  - [GitHub](https://github.com/Facico/Chinese-Vicuna)
- Chinese-alpaca-lora
  - [GitHub](https://github.com/LC1332/Chinese-alpaca-lora)

#### Japanese
- Japanese-Alpaca-LoRA
  - [GitHub](https://github.com/kunishou/Japanese-Alpaca-LoRA)

#### Medical
- 2023-ChatDoctor: A medical chat model fine-tuned on llama model using medical domain knowledge
  - [Paper](./papers/2023-ChatDoctor-%20A%20Medical%20Chat%20Model%20Fine-tuned%20on%20LLaMA%20Model%20using%20Medical%20Domain%20Knowledge.pdf)
- 华驼(HuaTuo): 基于中文医学知识的LLaMA微调模型
  - [GitHub](https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese)

#### Law
- LawGPT_zh：中文法律大模型（獬豸）
  - [GitHub](https://github.com/LiuHC0428/LAW-GPT)

#### Recommendation
- 2023-Recalpaca: Low-rank llama instruct-tuning for recommendation

#### Other
- 2023-A Survey of Domain Specialization for Large Language Models [[paper](https://arxiv.org/pdf/2305.18703.pdf)]

### Methods

#### RL
- 2017-Proximal Policy Optimization Algorithms [[paper](./papers/fine-tune/methods/rl/2017-Proximal%20Policy%20Optimization%20Algorithms.pdf)]
  - [Why is the log probability replaced with the importance sampling in the loss function?](https://ai.stackexchange.com/questions/7685/why-is-the-log-probability-replaced-with-the-importance-sampling-in-the-loss-fun)
- 2016-Asynchronous methods for deep reinforcement learning [[paper](./papers/fine-tune/methods/rl/2016-Asynchronous%20methods%20for%20deep%20reinforcement%20learning.pdf)]
- 2015-High-dimensional continuous control using generalized advantage estimation [[paper](./papers/fine-tune/methods/rl/2015-High-dimensional%20continuous%20control%20using%20generalized%20advantage%20estimation.pdf)]
- 2015-mlr-Trust Region Policy Optimization [[paper](./papers/fine-tune/methods/rl/2015-mlr-Trust%20Region%20Policy%20Optimization.pdf)]

##### Reward Modeling
- 2023-REWARD DESIGN WITH LANGUAGE MODELS [[paper](./papers/fine-tune/methods/rl/reward_modeling/2023-REWARD%20DESIGN%20WITH%20LANGUAGE%20MODELS.pdf)]
- 2022-Scaling Laws for Reward Model Overoptimization [[paper](./papers/fine-tune/methods/rl/reward_modeling/2022-Scaling%20Laws%20for%20Reward%20Model%20Overoptimization.pdf)]
- autocrit
  - [GitHub](https://github.com/CarperAI/autocrit/tree/contrastive-scalar-rm)
  - reward-modeling [GitHub](https://github.com/Dahoas/reward-modeling)
- 2023-On The Fragility of Learned Reward Functions [[paper](./papers/fine-tune/methods/rl/reward_modeling/2023-On%20The%20Fragility%20of%20Learned%20Reward%20Functions.pdf)]

#### peft 
- 2021-LoRA- Low-Rank Adaptation of Large Language Models [[paper](./papers/fine-tune/methods/peft/2021-LoRA-%20Low-Rank%20Adaptation%20of%20Large%20Language%20Models.pdf)]

#### align
- 2023-Preference Ranking Optimization for Human Alignment [[paper](https://arxiv.org/pdf/2306.17492.pdf)]
- 2023-Is Reinforcement Learning (Not) for Natural Language Processing: Benchmarks, Baselines, and Building Blocks for Natural Language Policy Optimization [[paper](./papers/fine-tune/methods/align/2023-ICLR-Is%20Reinforcement%20Learning%20(Not)%20for%20Natural%20Language%20Processing-%20Benchmarks,%20Baselines,%20and%20Building%20Blocks%20for%20Natural%20Language%20Policy%20Optimization.pdf)]
- 2023-Fine-Grained Human Feedback Gives Better Rewards for Language Model Training [paper](./papers/fine-tune/methods/align/2023-Fine-Grained%20Human%20Feedback%20Gives%20Better%20Rewards%20for%20Language%20Model%20Training.pdf)]
- 2023-Chain of Hindsight Aligns Language Models with Feedback [[paper](./papers/fine-tune/methods/align/2023-Chain%20of%20Hindsight%20Aligns%20Language%20Models%20with%20Feedback.pdf)]
- 2023-Training Socially Aligned Language Models in Simulated Human Society [[paper]](./papers/fine-tune/methods/align/2023-Training%20Socially%20Aligned%20Language%20Models%20in%20Simulated%20Human%20Society.pdf)
- 2023-Let’s Verify Step by Step [[paper](./papers/fine-tune/methods/align/2023-Let’s%20Verify%20Step%20by%20Step.pdf)]
- 2023-The False Promise of Imitating Proprietary LLMs [[paper](./papers/fine-tune/methods/align/2023-The%20False%20Promise%20of%20Imitating%20Proprietary%20LLMs.pdf)]
- 2023-AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback [[paper](./papers/fine-tune/methods/align/2023-AlpacaFarm-%20A%20Simulation%20Framework%20for%20Methods%20that%20Learn%20from%20Human%20Feedback.pdf)]
- 2023-LIMA- Less Is More for Alignment [[paper](./papers/fine-tune/methods/align/2023-LIMA-%20Less%20Is%20More%20for%20Alignment.pdf)]
- 2023-RRHF: Rank Responses to Align Language Models with Human Feedback without tears [[paper](./papers/fine-tune/methods/align/2023-RRHF-%20Rank%20Responses%20to%20Align%20Language%20Models%20with%20Human%20Feedback%20without%20tears.pdf)] [[code](https://github.com/GanjinZero/RRHF)]
- 2022-Solving math word problems with process-and outcome-based feedback [[paper](https://arxiv.org/pdf/2211.14275.pdf)]
- 2022-Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback [[paper](./papers/fine-tune/methods/align/2022-Training%20a%20Helpful%20and%20Harmless%20Assistant%20with%20Reinforcement%20Learning%20from%20Human%20Feedback.pdf)]
- 2022-Training language models to follow instructions with human feedback [[paper](./papers/fine-tune/methods/align/2022-Training%20language%20models%20to%20follow%20instructions%20with%20human%20feedback.pdf)]
  - [GitHub](https://github.com/anthropics/hh-rlhf)
- 2022-Red teaming language models to reduce harms: Methods, scaling behaviors, and lessons learned [[paper](https://arxiv.org/abs/2209.07858)]
- 2022-LaMDA- Language Models for Dialog Applications [[Paper](./papers/pre-train/models/2022-LaMDA-%20Language%20Models%20for%20Dialog%20Applications.pdf)]
- 2022-Constitutional ai- Harmlessness from ai feedback [[paper](./papers/fine-tune/methods/align/2022-Constitutional%20ai-%20Harmlessness%20from%20ai%20feedback.pdf)]
- 2021-A general language assistant as a laboratory for alignment [[paper](./papers/fine-tune/methods/align/2021-A%20general%20language%20assistant%20as%20a%20laboratory%20for%20alignment.pdf)]
- 2021-Ethical and social risks of harm from language models [[paper](./papers/fine-tune/methods/align/2021-Ethical%20and%20social%20risks%20of%20harm%20from%20language%20models.pdf)]
- 2020-nips-Learning to summarize from human feedback [[paper](./papers/fine-tune/methods/align/2020-nips-Learning%20to%20summarize%20from%20human%20feedback.pdf)]
- 2019-Fine-Tuning Language Models from Human Preferences [[paper](./papers/fine-tune/methods/align/2019-Fine-Tuning%20Language%20Models%20from%20Human%20Preferences.pdf)]
- 2018-Scalable agent alignment via reward modeling: a research direction [[paper](./papers/fine-tune/methods/align/2018-Scalable%20agent%20alignment%20via%20reward%20modeling-%20a%20research%20direction.pdf)]
- Reinforcement Learning for Language Models [Blog](https://gist.github.com/yoavg/6bff0fecd65950898eba1bb321cfbd81)
- 2017-nips-Deep reinforcement learning from human preferences [[paper](./papers/fine-tune/methods/align/2017-nips-Deep%20reinforcement%20learning%20from%20human%20preferences.pdf)]
- 2016-Concrete problems in ai safety [[paper](./papers/fine-tune/methods/align/2016-Concrete%20problems%20in%20ai%20safety.pdf)]

### Other
- 2022-naacl-MetaICL- Learning to Learn In Context [[paper](./papers/fine-tune/methods/other/2022-naacl-MetaICL-%20Learning%20to%20Learn%20In%20Context.pdf)]
- 2022-iclr-Multitask Prompted Training Enables Zero-Shot Task Generalization [[paper](./papers/fine-tune/methods/other/2022-iclr-Multitask%20Prompted%20Training%20Enables%20Zero-Shot%20Task%20Generalization.pdf)]

## Prompt Learning
- 2023-Tree of Thoughts: Deliberate Problem Solving with Large Language Models [[paper](./papers/prompt-learning/2023-Tree%20of%20Thoughts-%20Deliberate%20Problem%20Solving%20with%20Large%20Language%20Models.pdf)]
- 2023-Guiding Large Language Models via Directional Stimulus Prompting [[paper](./papers/prompt-learning/2023-Guiding%20Large%20Language%20Models%20via%20Directional%20Stimulus%20Prompting.pdf)]
- 2023-ICLR-Self-Consistency Improves Chain of Thought Reasoning in Language Models [[paper](./papers/prompt-learning/2023-ICLR-Self-Consistency%20Improves%20Chain%20of%20Thought%20Reasoning%20in%20Language%20Models.pdf)]
- 2023-Is Prompt All You Need No. A Comprehensive and Broader View of Instruction Learning [[paper](./papers/prompt-learning/2023-Is%20Prompt%20All%20You%20Need%20No.%20A%20Comprehensive%20and%20Broader%20View%20of%20Instruction%20Learning.pdf)]

### Survey
- 2021-Pre-train, Prompt, and Predict- A Systematic Survey of Prompting Methods in Natural Language Processing [[paper](./papers/prompt-learning/2021-Pre-train,%20Prompt,%20and%20Predict-%20A%20Systematic%20Survey%20of%20Prompting%20Methods%20in%20Natural%20Language%20Processing.pdf)]

### Prompt Tuning
- 2023-Prompt Pre-Training with Twenty-Thousand Classes for Open-Vocabulary Visual Recognition [[paper](./papers/prompt-learning/2023-Prompt%20Pre-Training%20with%20Twenty-Thousand%20Classes%20for%20Open-Vocabulary%20Visual%20Recognition.pdf)]
- 2022-AC-PPT- Pre-trained Prompt Tuning for Few-shot Learning [[paper](./papers/prompt-learning/2022-AC-PPT-%20Pre-trained%20Prompt%20Tuning%20for%20Few-shot%20Learning.pdf)]
- 2022-ACL-P-Tuning- Prompt Tuning Can Be Comparable to Fine-tuning Across Scales and Tasks [[paper](./papers/prompt-learning/2022-ACL-P-Tuning-%20Prompt%20Tuning%20Can%20Be%20Comparable%20to%20Fine-tuning%20Across%20Scales%20and%20Tasks.pdf)]
- 2021-EMNLP-The Power of Scale for Parameter-Efficient Prompt Tuning [[paper](./papers/prompt-learning/2021-EMNLP-The%20Power%20of%20Scale%20for%20Parameter-Efficient%20Prompt%20Tuning.pdf)]
- 2021-acl-Prefix-Tuning- Optimizing Continuous Prompts for Generation [[paper](./papers/prompt-learning/2021-acl-Prefix-Tuning-%20Optimizing%20Continuous%20Prompts%20for%20Generation.pdf)]
- 2021-GPT Understands, Too [[paper](./papers/prompt-learning/2021-GPT%20Understands,%20Too.pdf)]

## Integrating External Data

### Tool Learning
- [ToolLearningPapers](https://github.com/thunlp/ToolLearningPapers)

### Methods
- 2023-OpenAGI: When LLM Meets Domain Experts [[paper](https://arxiv.org/abs/2304.04370)]
- 2023-WebCPM: Interactive Web Search for Chinese Long-form Question Answering [[paper](https://arxiv.org/abs/2305.06849)]
- 2023-Evaluating Verifiability in Generative Search Engines [[paper](./papers/integrating_external_data/2023-Evaluating%20Verifiability%20in%20Generative%20Search%20Engines.pdf)]
- 2023-Enabling Large Language Models to Generate Text with Citations [[paper](./papers/integrating_external_data/2023-Enabling%20Large%20Language%20Models%20to%20Generate%20Text%20with%20Citations.pdf)]
- 2022-ACL-Lifelong Pretraining: Continually Adapting Language Models to Emerging Corpora [[paper](https://aclanthology.org/2022.bigscience-1.1.pdf)]
- 2022-findings-acl-ELLE: Efficient Lifelong Pre-training for Emerging Data  [[paper](https://aclanthology.org/2022.findings-acl.220.pdf)]
- langchain
  - GitHub
    - [langchain](https://github.com/hwchase17/langchain)
    - [Chinese-LangChain](https://github.com/yanqiangmiffy/Chinese-LangChain)
- 2023-Check Your Facts and Try Again- Improving Large Language Models with External Knowledge and Automated Feedback [[paper](./papers/integrating_external_data/2023-Check%20Your%20Facts%20and%20Try%20Again-%20Improving%20Large%20Language%20Models%20with%20External%20Knowledge%20and%20Automated%20Feedback.pdf)]
- 2022-Teaching language models to support answers with verified quotes
- 2021-Webgpt: Browser-assisted question-answering with human feedback [[paper](./papers/integrating_external_data/2021-Webgpt-%20Browser-assisted%20question-answering%20with%20human%20feedback.pdf)]
- 2021-Improving language models by retrieving from trillions of tokens
- 2020-REALM: retrieval-augmented language model pre-training
- 2020-Retrieval-augmented generation for knowledge-intensive NLP tasks

### Other
- [如何为GPT/LLM模型添加额外知识？](https://www.zhihu.com/question/591935281)

## Dataset
### For Pre-training
- RedPajama-Data
  - [GitHub](https://github.com/togethercomputer/RedPajama-Data)
  - [RedPajama-Data-1T-HuggingFace](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T)
  - [Blog](https://www.together.xyz/blog/redpajama)
- C4
- Pile
- ROOTS
- Wudao Corpora
- 大规模中文自然语言处理语料 Large Scale Chinese Corpus for NLP
  - [GitHub](https://github.com/brightmart/nlp_chinese_corpus)
- CSL: A Large-scale Chinese Scientific Literature Dataset 中文科学文献数据集
  - [GitHub](https://github.com/ydli-ai/CSL)
- 中文图书语料集合
  - [GitHub](https://github.com/FudanNLPLAB/CBook-150K)
- Chinese Open Instruction Generalist (COIG)
  - [Paper](https://arxiv.org/pdf/2304.07987v1.pdf)
- 医疗数据集
  - [GitHub1](https://github.com/NLPxiaoxu/LLM-FineTune)
- 金融数据
  - [FinNLP-GitHub](https://github.com/AI4Finance-Foundation/FinNLP)
  - [SmoothNLP 金融文本数据集(公开) | Public Financial Datasets for NLP Researches](https://github.com/smoothnlp/FinancialDatasets)
  
### For SFT
- ChatAlpaca
  - [GitHub](https://github.com/cascip/ChatAlpaca)
- InstructionZoo
  - [GitHub](https://github.com/FreedomIntelligence/InstructionZoo)
- [FlagInstruct](https://github.com/FlagOpen/FlagInstruct)
- fnlp/moss-002-sft-data
  - [Hugging Face Datasets](https://huggingface.co/datasets/fnlp/moss-002-sft-data)

### For Reward Model

### For Evaluation
- [SuperCLUE：中文通用大模型综合性测评基准](https://www.cluebenchmarks.com/superclue.html)
- [Open LLMs benchmark大模型能力评测标准计划](https://mp.weixin.qq.com/s/oGo9GJUeUn09mOpUutToJw)
- 中文医疗大模型评测基准-PromptCBLUE
- GLUE、SuperGLUE、SQuAD、CoQA、WMT、LAMBADA、ROUGE、智源指数CUGE、MMLU、Hellaswag、OpenBookQA、ARC、TriviaQA、TruthfulQA

### Methods
- 2023-A Pretrainer’s Guide to Training Data: Measuring the Effects of Data Age, Domain Coverage, Quality, & Toxicity [[paper](./papers/dataset/methods/2023-A%20Pretrainer’s%20Guide%20to%20Training%20Data-%20Measuring%20the%20Effects%20of%20Data%20Age,%20Domain%20Coverage,%20Quality,%20&%20Toxicity.pdf)]
- 2023-DoReMi: Optimizing data mixtures speeds up language model pretraining
- 2023-Data selection for language models via importance resampling
- 2022-SELF-INSTRUCT- Aligning Language Model with Self Generated Instructions [[paper](./papers/dataset/methods/2022-SELF-INSTRUCT-%20Aligning%20Language%20Model%20with%20Self%20Generated%20Instructions.pdf)]
- 2022-acl-Deduplicating training data makes language models better [[paper](./papers/dataset/methods/2022-acl-Deduplicating%20training%20data%20makes%20language%20models%20better.pdf)]

## Evaluation
- 2023-Harnessing the Power of LLMs in Practice- A Survey on ChatGPT and Beyond [[paper](./papers/evaluation/2023-Harnessing%20the%20Power%20of%20LLMs%20in%20Practice-%20A%20Survey%20on%20ChatGPT%20and%20Beyond.pdf)]
- 2023-INSTRUCTEVAL: Towards Holistic Evaluation of Instruction-Tuned Large Language Models [[paper](./papers/evaluation/2023-INSTRUCTEVAL-%20Towards%20Holistic%20Evaluation%20of%20Instruction-Tuned%20Large%20Language%20Models.pdf)]
- LLMZoo: a project that provides data, models, and evaluation benchmark for large language models.
  - [GitHub](https://github.com/FreedomIntelligence/LLMZoo)
- 2023-Evaluating ChatGPT's Information Extraction Capabilities- An Assessment of Performance, Explainability, Calibration, and Faithfulness [paper](./papers/evaluation/2023-Evaluating%20ChatGPT's%20Information%20Extraction%20Capabilities-%20An%20Assessment%20of%20Performance,%20Explainability,%20Calibration,%20and%20Faithfulness.pdf)
- 2023-Towards Better Instruction Following Language Models for Chinese- Investigating the Impact of Training Data and Evaluation [paper](./papers/evaluation/2023-Towards%20Better%20Instruction%20Following%20Language%20Models%20for%20Chinese-%20Investigating%20the%20Impact%20of%20Training%20Data%20and%20Evaluation.pdf)
- [PandaLM](https://github.com/WeOpenML/PandaLM)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [BIG-bench](https://github.com/google/BIG-bench)
- 2023-HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models [[paper](https://arxiv.org/pdf/2305.11747v2.pdf)]
- 2023-C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models [[paper](https://arxiv.org/abs/2305.08322)]
- 2023-Safety Assessment of Chinese Large Language Models [[paper](https://arxiv.org/pdf/2304.10436.pdf)]
- 2022-Holistic Evaluation of Language Models [[paper](https://arxiv.org/pdf/2211.09110.pdf)]

### Aspects
- helpfulness
- honesty
- harmlessness
- truthfulness
- robustness
- Bias, Toxicity and Misinformation

### 评估挑战
- 已有的评估通常只用已有的常见NLP任务，海量的其它任务并没有评估，比如写邮件

## Inference

## Analysis
- Pythia: Interpreting Autoregressive Transformers Across Time and Scale 
  - [GitHub](https://github.com/EleutherAI/pythia)
- 2023-Inspecting and Editing Knowledge Representations in Language Models [[paper](https://arxiv.org/abs/2304.00740)]

## Products
- [ChatGPT](https://chat.openai.com/)
- [文心一言](https://yiyan.baidu.com/)
- [通义千问](https://tongyi.aliyun.com/)
- AgentGPT
  - [GitHub](https://github.com/reworkd/AgentGPT)
- HuggingGPT
  - [GitHub](https://github.com/microsoft/JARVIS)
  - [Paper](https://arxiv.org/abs/2303.17580)
- AutoGPT
  - [GitHub](https://github.com/Significant-Gravitas/Auto-GPT)
- MiniGPT-4
  - [GitHub](https://github.com/Vision-CAIR/MiniGPT-4)
  - [Paper](./papers/2023-MiniGPT_4.pdf)
- ShareGPT
  - [GitHub](https://github.com/domeccleston/sharegpt)
- character ai
  - [Site](https://beta.character.ai/)
- LLaVA
  - [Paper](./papers/2023-Visual%20Instruction%20Tuning.pdf)
  - [Site](https://llava-vl.github.io/)
- Video-LLaMA
  - [Paper](./papers/2023-Video-LLaMA-%20An%20Instruction-tuned%20Audio-Visual%20Language%20Model%20for%20Video%20Understanding.pdf)
- ChatPaper
  - [GitHub](https://github.com/kaixindelele/ChatPaper)

## Tools
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)
  - [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)
- [ColossalAI](https://github.com/hpcaitech/ColossalAI)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [trlx](https://github.com/CarperAI/trlx)
- [trl](https://github.com/lvwerra/)
- [MOSS-RLHF](https://github.com/OpenLMLab/MOSS-RLHF)

## Traditional Nlp Tasks
- 2023-AnnoLLM- Making Large Language Models to Be Better Crowdsourced Annotators [[paper](./papers/traditional_nlp_tasks/2023-AnnoLLM-%20Making%20Large%20Language%20Models%20to%20Be%20Better%20Crowdsourced%20Annotators.pdf)]
- 2022-Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks [[paper](https://arxiv.org/abs/2204.07705)]

### Sentiment Analysis
- 2023-Sentiment Analysis in the Era of Large Language Models- A Reality Check [[Paper](./papers/traditional_nlp_tasks/sentiment_analysis/2023-Sentiment%20Analysis%20in%20the%20Era%20of%20Large%20Language%20Models-%20A%20Reality%20Check.pdf)] [[GitHub](https://github.com/DAMO-NLP-SG/LLM-Sentiment)]
- 2023-Can chatgpt understand too? A comparative study on chatgpt and fine-tuned BERT
- 2023-Is chatgpt a good sentiment analyzer? A preliminary study
- 2023-Llms to the moon? reddit market sentiment analysis with large language models

## Related Topics
### Neural Text Generation
- 2020-ICLR-Neural text generation with unlikelihood training [[paper](./papers/related-topics/neural_text_generation/2020-ICLR-Neural%20text%20generation%20with%20unlikelihood%20training.pdf)]
- 2021-findings-emnlp-GeDi- Generative Discriminator Guided Sequence Generation [[paper](./papers/related-topics/neural_text_generation/2021-findings-emnlp-GeDi-%20Generative%20Discriminator%20Guided%20Sequence%20Generation.pdf)]
- 2021-ACL-DExperts- Decoding-Time Controlled Text Generation with Experts and Anti-Experts [[paper](./papers/related-topics/neural_text_generation/2021-ACL-DExperts-%20Decoding-Time%20Controlled%20Text%20Generation%20with%20Experts%20and%20Anti-Experts.pdf)]
- 2021-ICLR-Mirostat- a neural text decoding algorithm that directly controls perplexity [[paper](./papers/related-topics/neural_text_generation/2021-ICLR-Mirostat-%20a%20neural%20text%20decoding%20algorithm%20that%20directly%20controls%20perplexity.pdf)]
- 2022-NIPS-A Contrastive Framework for Neural Text Generation [[paper](./papers/related-topics/neural_text_generation/2022-NIPS-A%20Contrastive%20Framework%20for%20Neural%20Text%20Generation.pdf)]

#### Controllable Generation
- 2022-ACL-Length Control in Abstractive Summarization by Pretraining Information Selection [[paper](./papers/related-topics/neural_text_generation/controllable_generation/2022-ACL-Length%20Control%20in%20Abstractive%20Summarization%20by%20Pretraining%20Information%20Selection.pdf)]

### Distributed Training
- [Pytorch 分布式训练](https://zhuanlan.zhihu.com/p/76638962)
  - [浅谈Tensorflow分布式架构：ring all-reduce算法](https://zhuanlan.zhihu.com/p/69797852)
  - [Optimizer state sharding (ZeRO)](https://zhuanlan.zhihu.com/p/394064174)
    - [ZeRO-Offload](https://www.deepspeed.ai/tutorials/zero-offload/)
  - 图解大模型训练
    - [图解大模型训练之：流水线并行（Pipeline Parallelism），以Gpipe为例](https://zhuanlan.zhihu.com/p/613196255)
    - [图解大模型训练之：数据并行上篇(DP, DDP与ZeRO)](https://zhuanlan.zhihu.com/p/617133971)
    - [图解大模型训练之：数据并行下篇( DeepSpeed ZeRO，零冗余优化)](https://zhuanlan.zhihu.com/p/618865052)

### Quantization
- 2020-Integer Quantization for Deep Learning Inference Principles and Empirical Evaluation [[paper](./papers/related-topics/quantization/2020-Integer%20Quantization%20for%20Deep%20Learning%20Inference%20Principles%20and%20Empirical%20Evaluation.pdf)]
- 2023-ICLR-GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers [[paer](./papers/related-topics/quantization/2023-ICLR-GPTQ-%20Accurate%20Post-Training%20Quantization%20for%20Generative%20Pre-trained%20Transformers.pdf)]
- 2023-QLORA: Efficient Finetuning of Quantized LLMs [[paper](./papers/related-topics/quantization/2023-QLORA-%20Efficient%20Finetuning%20of%20Quantized%20LLMs.pdf)]

## Other
- [如何为GPT/LLM模型添加额外知识？](https://www.zhihu.com/question/591935281/answer/2979220793)
- [如何正确复现 Instruct GPT / RLHF?](https://zhuanlan.zhihu.com/p/622134699)
- [ChatGPT在单个NLP数据集任务上比SOTA有多大提升？](https://www.zhihu.com/question/595938881)
- [影响PPO算法性能的10个关键技巧（附PPO算法简洁Pytorch实现）](https://zhuanlan.zhihu.com/p/512327050)
- [灌水新方向 偏好强化学习概述](https://zhuanlan.zhihu.com/p/622056740?utm_source=wechat_session&utm_medium=social&utm_oi=556103293550534656)
- [为什么说大模型训练很难？](https://www.zhihu.com/question/498271491)
- [如何基于深度学习大模型开展小模型的研发，如何把大模型和小模型相结合？](https://www.zhihu.com/question/594938636/answer/3081636786)

## Related Project
- [open-llms](https://github.com/eugeneyan/open-llms) A list of open LLMs available for commercial use.
- [safe-rlhf](https://github.com/PKU-Alignment/safe-rlhf)
- [Awesome-Multimodal-Large-Language-Models](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models)
- [Awesome-Chinese-LLM](https://github.com/HqWu-HITCS/Awesome-Chinese-LLM)