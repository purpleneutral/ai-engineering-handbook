# A Brief History of AI and LLMs

Last reviewed: 2026-02-10

[Contents](README.md) | [Prev](10-reading-list.md) | [Next](README.md)

## Summary

The field of artificial intelligence is over 70 years old, and most of the ideas behind today's LLMs have roots that stretch back decades. Understanding this history helps practitioners see current tools not as magic but as the latest iteration in a long sequence of engineering trade-offs. This chapter walks through the major eras, from the first mathematical model of a neuron in 1943 to today's reasoning models and agent frameworks, with an emphasis on *why* each development mattered and what practical consequences followed.

## See Also
- [LLM Fundamentals](01-llm-fundamentals.md) — How transformers work under the hood, and the training pipeline that produces modern chat models.
- [Agents](04-agents.md) — The current state of the art in agentic systems, whose history runs through expert systems, tool use, and the ReAct pattern.
- [Staying Current (Without Chasing Hype)](13-staying-current.md) — How to follow new developments without losing perspective.
- [Reading List (Curated)](10-reading-list.md) — Primary sources and foundational papers, many of which appear in this chapter's references.

---

## The Foundations (1943--1956)

Three developments set the stage for everything that followed.

In 1943, neurophysiologist Warren McCulloch and logician Walter Pitts published ["A Logical Calculus of the Ideas Immanent in Nervous Activity."](https://doi.org/10.1007/BF02478259) They proposed a mathematical model of an artificial neuron: a simple binary unit that fires (outputs 1) or stays silent (outputs 0) based on whether the weighted sum of its inputs exceeds a threshold. The paper demonstrated that networks of these units could, in principle, compute any logical function. It was the first formal argument that brains could be modeled as computational systems, and it planted the seed for every neural network that would follow.

In 1950, Alan Turing published ["Computing Machinery and Intelligence"](https://doi.org/10.1093/mind/LIX.236.433) in the journal *Mind*. Rather than trying to define intelligence philosophically, Turing proposed a practical test: if an interrogator, communicating by text, cannot reliably distinguish a machine from a human, the machine can be said to exhibit intelligent behavior. The "imitation game" (now called the Turing Test) was not meant as a final definition of intelligence. It was a pragmatic reframing that let researchers stop arguing about consciousness and start building systems. That instinct --- measure behavior, not intentions --- remains central to how we evaluate LLMs today through benchmarks and evals.

In the summer of 1956, John McCarthy, Marvin Minsky, Claude Shannon, and Nathaniel Rochester organized a workshop at [Dartmouth College](https://home.dartmouth.edu/about/artificial-intelligence-ai-coined-dartmouth). The proposal stated its goal plainly: to study "the conjecture that every aspect of learning or any other feature of intelligence can in principle be so precisely described that a machine can be made to simulate it." McCarthy coined the term "artificial intelligence" for the workshop, and it stuck. Attendees included Allen Newell and Herbert Simon, who presented the Logic Theorist --- a program that could prove mathematical theorems. The Dartmouth workshop gave the field its name, its mission, its first working system, and many of its founding researchers.

## Early Enthusiasm and the First AI Winter (1956--1980s)

The decade after Dartmouth produced a burst of optimism. Researchers at MIT, Carnegie Mellon, and Stanford built programs that could solve algebra problems, prove geometry theorems, and play checkers. In 1958, Frank Rosenblatt at Cornell built the [Mark I Perceptron](https://www.ling.upenn.edu/courses/cogs501/Rosenblatt1958.pdf), a hardware device that could learn to classify simple visual patterns by adjusting connection weights automatically. The Navy, which funded the project, generated breathless press coverage. The New York Times reported it as "the embryo of an electronic computer that [the Navy] expects will be able to walk, talk, see, write, reproduce itself and be conscious of its existence."

In 1966, Joseph Weizenbaum at MIT created [ELIZA](https://www.csee.umbc.edu/courses/331/papers/eliza.html), a program that simulated a Rogerian psychotherapist by pattern-matching user input and reflecting it back as questions. ELIZA had no understanding whatsoever --- it operated on about 200 lines of pattern-matching rules. Yet users formed emotional attachments to it. Weizenbaum's secretary reportedly asked him to leave the room so she could talk to the program privately. This tendency to attribute understanding to systems that merely produce plausible text was later named the "ELIZA effect." It is directly relevant to working with LLMs today: users (and sometimes developers) consistently overestimate what a fluent response implies about the system's actual understanding.

The optimism broke on two fronts. First, in 1969, Marvin Minsky and Seymour Papert published [*Perceptrons*](https://en.wikipedia.org/wiki/Perceptrons_%28book%29), a rigorous mathematical analysis showing that single-layer perceptrons could not learn certain functions --- most famously, XOR. The book's critique applied to single-layer networks specifically, and Minsky and Papert acknowledged that multi-layer networks might overcome the limitation. But nobody knew how to train multi-layer networks at the time, and the book's impact was devastating. Funding agencies, particularly DARPA, took it as evidence that neural network research was a dead end.

Second, by the early 1970s, the grand promises of early AI researchers remained unfulfilled. In 1973, the UK Parliament commissioned the [Lighthill Report](https://en.wikipedia.org/wiki/Lighthill_report), in which mathematician Sir James Lighthill criticized AI research for its "utter failure" to achieve its ambitious objectives. He identified "combinatorial explosion" as a fundamental barrier: algorithms that worked on toy problems collapsed on real-world ones. The report led the UK Science Research Council to withdraw virtually all funding for AI research. In the US, the 1969 Mansfield Amendment required DARPA to fund only mission-oriented research, and the Lighthill Report reinforced skepticism about pure AI research. By 1974, AI funding was difficult to find on both sides of the Atlantic. This period, roughly 1974--1980, became known as the first AI winter.

During the late 1970s and 1980s, a different approach gained traction: expert systems. These were rule-based programs that encoded human domain expertise as if-then rules and could make decisions within narrow domains. Systems like MYCIN (medical diagnosis) and R1/XCON (computer configuration at DEC) demonstrated real commercial value. The Japanese Fifth Generation project (1982) and a wave of corporate investment drove a billion-dollar industry built on specialized LISP hardware and large knowledge bases. But expert systems were brittle --- they could not handle situations outside their coded rules, and maintaining the rule bases was expensive. In 1984, Roger Schank and Minsky warned at the AAAI conference that hype was outpacing reality. Three years later, the market for specialized AI hardware collapsed as general-purpose workstations from Sun Microsystems became cheaper and more capable. The result was a second AI winter that lasted from roughly 1987 to 1993.

## Neural Network Revival and the Quiet Years (1980s--2000s)

The revival of neural networks began before the second AI winter set in. In 1986, David Rumelhart, Geoffrey Hinton, and Ronald Williams published ["Learning representations by back-propagating errors"](https://www.nature.com/articles/323533a0) in *Nature*. Backpropagation was not entirely new --- the mathematical foundations had been developed by Seppo Linnainmaa in 1970 and Paul Werbos in the 1970s --- but the 1986 paper demonstrated its practical effectiveness for training multi-layer networks. For the first time, researchers had a general-purpose method for training networks with hidden layers, which directly addressed the limitation Minsky and Papert had identified.

The key insight was this: by computing how much each weight in the network contributed to the overall error and adjusting all weights simultaneously using gradient descent, a multi-layer network could learn internal representations that were never explicitly programmed. This is the same basic algorithm that trains every neural network today, including every LLM.

Despite this breakthrough, neural networks did not dominate the 1990s. The computers of the era lacked the power to train large networks on realistic datasets, so neural network approaches worked in theory but struggled in practice. Meanwhile, other machine learning methods thrived. In 1995, Corinna Cortes and Vladimir Vapnik published the modern formulation of [Support Vector Machines (SVMs)](https://doi.org/10.1007/BF00994018), which used the "kernel trick" to classify non-linearly separable data with strong theoretical guarantees. SVMs, along with methods like random forests and boosted trees, became the standard toolkit for machine learning through the 2000s. They required less compute than neural networks, worked well on smaller datasets, and had well-understood theoretical properties.

One neural network innovation from this period deserves special mention: [Long Short-Term Memory (LSTM)](https://doi.org/10.1162/neco.1997.9.8.1735), introduced by Sepp Hochreiter and Jurgen Schmidhuber in 1997. LSTMs solved the "vanishing gradient" problem that made it nearly impossible for standard recurrent neural networks to learn long-range dependencies in sequential data. By introducing gated memory cells that could selectively store, update, and forget information, LSTMs became the dominant architecture for sequence modeling --- speech recognition, machine translation, time-series prediction --- and remained so until transformers replaced them two decades later.

## Deep Learning Breaks Through (2006--2017)

The deep learning era began quietly. In 2006, Geoffrey Hinton and colleagues published ["A Fast Learning Algorithm for Deep Belief Nets,"](https://doi.org/10.1162/neco.2006.18.7.1527) introducing a method for training networks with many layers by pre-training them one layer at a time using unsupervised learning. The paper mattered less for the specific technique (deep belief nets are rarely used today) than for the message it sent: deep networks with many layers were tractable, and depth was valuable. It renewed interest in neural network research at a time when most of the field had moved on.

Two developments outside of AI made the deep learning revolution possible. First, the internet produced vast amounts of data for training. Second, GPUs, originally designed for video games, turned out to be excellent hardware for the parallel matrix operations that neural network training requires. Both were necessary conditions for what happened next.

In 2012, Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton entered the [ImageNet](https://www.image-net.org/) Large Scale Visual Recognition Challenge with [AlexNet](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html), a deep convolutional neural network trained on GPUs. Previous winners had achieved top-5 error rates around 25--26% using carefully hand-engineered features fed into traditional classifiers like SVMs. AlexNet achieved 15.3% --- a 10.8 percentage-point improvement that stunned the computer vision community. Yann LeCun called it "an unequivocal turning point in the history of computer vision." The result demonstrated something that the field had debated for years: features learned automatically by deep networks from data could dramatically outperform features designed by human experts.

AlexNet triggered a paradigm shift. Within two years, virtually every competitive entry in ImageNet used deep neural networks. Google acquired the startup that Hinton, Krizhevsky, and Sutskever formed. Industry investment in deep learning accelerated rapidly.

The years 2013--2016 produced a series of advances that built the foundation for modern LLMs:

- **[Word2Vec](https://arxiv.org/abs/1301.3781) (2013).** Tomas Mikolov and colleagues at Google showed that simple neural networks trained to predict neighboring words in text produce vector representations (embeddings) that encode semantic relationships. The famous result: the vector for "king" minus "man" plus "woman" produces a vector close to "queen." Word2Vec demonstrated that neural networks could discover linguistic structure from raw text without explicit supervision --- a precursor to the pre-training paradigm that defines modern LLMs. See [Embeddings and Vector Search](12-embeddings-and-vector-search.md) for how embeddings are used in practice today.

- **Sequence-to-Sequence Models (2014).** Ilya Sutskever, Oriol Vinyals, and Quoc Le published ["Sequence to Sequence Learning with Neural Networks,"](https://arxiv.org/abs/1409.3215) showing that an LSTM encoder-decoder architecture could map variable-length input sequences to variable-length output sequences. This became the standard architecture for machine translation and established the encoder-decoder pattern that the transformer would later refine.

- **The Attention Mechanism (2014).** Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio published ["Neural Machine Translation by Jointly Learning to Align and Translate."](https://arxiv.org/abs/1409.0473) The key insight: instead of compressing the entire input into a single fixed-length vector (which degraded performance on long sequences), let the decoder selectively attend to different parts of the input at each step. This "attention" mechanism was the direct ancestor of the self-attention that powers transformers.

- **[Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) (2014).** Ian Goodfellow and colleagues introduced GANs --- two neural networks (a generator and a discriminator) trained against each other in an adversarial game. GANs became the leading technique for generating realistic images and demonstrated that neural networks could create, not just classify. While GANs themselves have been largely superseded by diffusion models for image generation, the adversarial training concept influenced the broader field.

- **[ResNet](https://arxiv.org/abs/1512.03385) (2015).** Kaiming He and colleagues introduced residual connections (skip connections), which allowed training of networks with over 100 layers by enabling gradient flow through shortcut paths. ResNets won ImageNet 2015 and demonstrated that depth itself was a powerful lever, provided the training dynamics could be managed.

## The Transformer Revolution (2017--2020)

In June 2017, Ashish Vaswani and seven co-authors at Google published ["Attention Is All You Need."](https://arxiv.org/abs/1706.03762) The paper proposed the transformer architecture, which dispensed entirely with recurrence and convolutions, relying solely on self-attention mechanisms. The core ideas --- multi-head self-attention, positional encoding, and the encoder-decoder structure --- are covered in detail in [LLM Fundamentals](01-llm-fundamentals.md).

What made the transformer consequential was not any single technique but the combination of properties it offered. Unlike RNNs, transformers process all tokens in parallel during training, which makes them dramatically faster to train on modern hardware. Unlike CNNs, they can model arbitrary-length dependencies without stacking many layers. The result was a model that was simultaneously more powerful and more parallelizable than anything before it. The paper has been cited over 173,000 times and every major language model since 2018 is built on the transformer architecture.

The transformer spawned two major lineages, each emphasizing a different half of the original encoder-decoder architecture:

**[BERT](https://arxiv.org/abs/1810.04805) (2018).** Jacob Devlin and colleagues at Google published "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." BERT used only the encoder half of the transformer, pre-trained with a masked language model objective (predict randomly masked tokens using both left and right context). BERT set new state-of-the-art results on 11 NLP tasks and proved that pre-training on large unlabeled corpora followed by task-specific fine-tuning was a devastatingly effective strategy. It became the backbone of Google Search and launched an era of "BERTology" --- dozens of BERT variants optimized for specific tasks and domains.

**[GPT-1](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (2018).** Alec Radford and colleagues at OpenAI published "Improving Language Understanding by Generative Pre-Training," which used only the decoder half of the transformer, pre-trained to predict the next token left-to-right. Where BERT was bidirectional and excelled at understanding tasks (classification, extraction), GPT was autoregressive and excelled at generation. GPT-1 had 117 million parameters and demonstrated strong transfer learning: pre-train once, fine-tune for many tasks.

**[GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (2019).** OpenAI scaled GPT to 1.5 billion parameters and trained it on a larger dataset. GPT-2 could generate remarkably coherent multi-paragraph text and performed reasonably well on various tasks without any fine-tuning at all --- just by being prompted with the right text. OpenAI initially withheld the full model, citing concerns about potential misuse for generating disinformation. The decision was controversial (some called it a publicity stunt), but it foreshadowed the real safety debates that would intensify with later models.

**[GPT-3](https://arxiv.org/abs/2005.14165) (2020).** OpenAI scaled to 175 billion parameters. GPT-3 demonstrated that scale alone could produce qualitatively different capabilities. With appropriate prompting, it could perform translation, arithmetic, code generation, and creative writing without any task-specific training --- a capability called "few-shot learning" because you could guide it by providing a few examples in the prompt. GPT-3 was released only as an API, establishing the "model-as-a-service" business model that dominates today. The GPT-3 paper, ["Language Models are Few-Shot Learners,"](https://arxiv.org/abs/2005.14165) is required reading for understanding why prompting works the way it does.

## The Instruction-Following Era (2020--2023)

GPT-3 was impressive but impractical for most applications. It would happily generate toxic text, follow instructions embedded in user data (prompt injection avant la lettre), and produce confident nonsense. It behaved like what it was: a very good autocomplete engine that had absorbed the internet.

The breakthrough that made LLMs usable as assistants came from changing how they were trained after pre-training. In March 2022, OpenAI published the [InstructGPT](https://arxiv.org/abs/2203.02155) paper, "Training language models to follow instructions with human feedback." The approach had three steps: (1) fine-tune GPT-3 on human-written demonstrations of instruction following; (2) collect human rankings of model outputs; (3) use those rankings to train a reward model and optimize the language model against it using reinforcement learning --- a technique called RLHF (Reinforcement Learning from Human Feedback). InstructGPT was smaller than GPT-3 but was strongly preferred by human raters because it actually did what you asked.

This training pipeline --- pre-training, supervised fine-tuning (SFT), and RLHF/alignment --- is now the standard recipe for producing a useful chat model. It is covered in detail in [LLM Fundamentals](01-llm-fundamentals.md).

On November 30, 2022, OpenAI released [ChatGPT](https://openai.com/blog/chatgpt) --- a fine-tuned version of GPT-3.5 (itself an iteration on GPT-3 with InstructGPT techniques applied) --- as a free research preview. It reached one million users in five days and 100 million monthly active users within two months, making it the fastest-growing consumer application in history. ChatGPT did not introduce new technical capabilities. What it introduced was accessibility: a simple chat interface that let anyone interact with an instruction-tuned LLM. The result was an explosion of public awareness, corporate investment, and competitive pressure that reshaped the entire technology industry.

In March 2023, OpenAI released [GPT-4](https://arxiv.org/abs/2303.08774), which brought multimodal capabilities (image understanding), substantially improved reasoning, and better instruction following. Exact architectural details remain undisclosed.

Anthropic, founded in 2021 by former OpenAI executives Dario and Daniela Amodei, took a different approach to alignment with [Constitutional AI (CAI)](https://arxiv.org/abs/2212.08073). Instead of relying solely on human raters to judge outputs, CAI trains models to critique and revise their own responses against a written set of principles (a "constitution"), then uses AI-generated preferences to train a reward model. Anthropic released Claude 1 in March 2023, Claude 2 (with a 100K-token context window) in July 2023, and the Claude 3 family (Haiku, Sonnet, Opus) with multimodal input in March 2024.

The open-weight movement began in earnest in February 2023, when Meta released [LLaMA](https://arxiv.org/abs/2302.13971) --- a collection of foundation models from 7B to 65B parameters that, remarkably, could match or exceed GPT-3's performance at a fraction of the parameter count. LLaMA's weights were leaked almost immediately and spread through online communities, democratizing access to capable base models. Meta embraced this trajectory with Llama 2 (July 2023), released under a permissive commercial license. The availability of open-weight models created an entire ecosystem of fine-tuned variants, quantized models for consumer hardware, and community-driven tooling that would have been impossible with API-only access.

## The Current Landscape (2023--Present)

As of early 2026, several trends are reshaping the field:

**Reasoning models.** In September 2024, OpenAI released [o1](https://openai.com/index/learning-to-reason-with-llms/), a model trained with reinforcement learning to "think before it answers" by generating an internal chain of thought. The key insight is test-time compute scaling: giving the model more time to reason (at inference) improves accuracy on hard problems, adding a new dimension to performance alongside model size and training data. In January 2025, DeepSeek released [R1](https://arxiv.org/abs/2501.12948), which demonstrated that reasoning capabilities can emerge from pure RL without expensive supervised fine-tuning data, and open-sourced the model and distilled variants. The reasoning model paradigm trades latency and cost for accuracy --- a trade-off that makes sense for some applications (complex coding, math, analysis) but not others (chat, simple extraction).

**Multimodal models as the default.** Processing images, audio, and video alongside text has moved from an experimental feature to a baseline capability. GPT-4o (May 2024), the Claude 3 family (March 2024), and Llama 4 (2025, natively multimodal with a mixture-of-experts architecture) all treat multiple modalities as first-class inputs. For practitioners, this means the same model that processes text can now process screenshots, diagrams, and audio --- which changes the design space for applications like document processing and accessibility.

**Agents and tool use.** LLMs that plan, call tools, observe results, and iterate have moved from research prototypes toward production use. In November 2024, Anthropic introduced the [Model Context Protocol (MCP)](https://www.anthropic.com/news/model-context-protocol), an open standard for connecting AI assistants to external tools and data sources. MCP addresses the "N x M integration problem" --- without a standard protocol, every combination of AI application and external tool requires a custom integration. MCP is now hosted by the Linux Foundation and adopted by multiple vendors. See [Agents](04-agents.md) for architectural patterns and [Structured Outputs and Tool Calling](11-structured-outputs-and-tool-calling.md) for the mechanics.

**The scaling debate.** For several years, the dominant hypothesis was that making models bigger and training them on more data would continue to produce new capabilities (the "scaling laws" framework). As of 2025, diminishing returns on pure scale are driving research toward other levers: better data curation, test-time compute (reasoning models), mixture-of-experts architectures (which activate only a fraction of parameters per forward pass), and more efficient training methods. The practical implication for engineers is that model capability per dollar continues to improve, but the improvements increasingly come from architectural and training innovations rather than brute-force scaling.

**Open-weight competition.** Meta's Llama series, Mistral's models, and DeepSeek's releases have closed much of the capability gap with proprietary models. Llama 3.1 405B (July 2024) achieved rough parity with GPT-4 on many benchmarks. As of February 2026, Llama 4 Maverick (17B active parameters, 128 experts) beats GPT-4o on several multimodal benchmarks at less than half the active parameter count. For teams that need data sovereignty, cost control, or customization, open-weight models are now a credible option rather than a compromise. See [Stacks and Difficulty](16-stacks-and-difficulty.md) for guidance on when to use which.

## Timeline of Key Milestones

| Year | Milestone | Why It Mattered |
|------|-----------|-----------------|
| 1943 | [McCulloch & Pitts](https://doi.org/10.1007/BF02478259): mathematical model of artificial neuron | Established that neural computation can be modeled formally |
| 1950 | Turing: ["Computing Machinery and Intelligence"](https://doi.org/10.1093/mind/LIX.236.433) | Proposed measuring machine intelligence by behavior, not essence |
| 1956 | [Dartmouth Conference](https://home.dartmouth.edu/about/artificial-intelligence-ai-coined-dartmouth) | Named the field; launched AI as an academic discipline |
| 1958 | Rosenblatt: [Mark I Perceptron](https://www.ling.upenn.edu/courses/cogs501/Rosenblatt1958.pdf) | First machine that could learn from data |
| 1966 | Weizenbaum: [ELIZA](https://www.csee.umbc.edu/courses/331/papers/eliza.html) | Demonstrated (and warned about) the tendency to over-attribute understanding |
| 1969 | Minsky & Papert: [*Perceptrons*](https://en.wikipedia.org/wiki/Perceptrons_%28book%29) | Exposed limitations of single-layer networks; dampened neural network funding |
| 1973 | [Lighthill Report](https://en.wikipedia.org/wiki/Lighthill_report) | Triggered UK funding cuts; reinforced first AI winter |
| 1986 | [Rumelhart, Hinton & Williams](https://www.nature.com/articles/323533a0): backpropagation paper | Made multi-layer neural network training practical |
| 1995 | [Cortes & Vapnik](https://doi.org/10.1007/BF00994018): modern SVMs | Dominated ML for a decade; strong theoretical foundations |
| 1997 | [Hochreiter & Schmidhuber](https://doi.org/10.1162/neco.1997.9.8.1735): LSTM | Solved vanishing gradients; became standard for sequence modeling |
| 2006 | [Hinton](https://doi.org/10.1162/neco.2006.18.7.1527): deep belief networks | Demonstrated viability of deep networks; reignited interest in depth |
| 2012 | [Krizhevsky, Sutskever & Hinton](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html): AlexNet | Proved deep learning beats hand-engineered features; GPU training |
| 2013 | [Mikolov et al.](https://arxiv.org/abs/1301.3781): Word2Vec | Showed neural networks can discover semantic structure from raw text |
| 2014 | [Bahdanau et al.](https://arxiv.org/abs/1409.0473): attention mechanism | Enabled models to selectively focus on relevant input; precursor to transformers |
| 2014 | [Sutskever et al.](https://arxiv.org/abs/1409.3215): sequence-to-sequence | Established encoder-decoder paradigm for variable-length sequences |
| 2014 | [Goodfellow et al.](https://arxiv.org/abs/1406.2661): GANs | Introduced adversarial training for generative models |
| 2015 | [He et al.](https://arxiv.org/abs/1512.03385): ResNet | Residual connections enabled training of 100+ layer networks |
| 2017 | [Vaswani et al.](https://arxiv.org/abs/1706.03762): "Attention Is All You Need" | Introduced the transformer; foundation of all modern LLMs |
| 2018 | [Devlin et al.](https://arxiv.org/abs/1810.04805): BERT | Pre-train then fine-tune paradigm; dominated NLU benchmarks |
| 2018 | [Radford et al.](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf): GPT-1 | Decoder-only transformer; generative pre-training for NLP |
| 2019 | OpenAI: [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (1.5B params) | Demonstrated coherent long-form text generation; raised safety questions |
| 2020 | OpenAI: [GPT-3](https://arxiv.org/abs/2005.14165) (175B params) | Few-shot learning from prompts; model-as-a-service business model |
| 2022 | OpenAI: [InstructGPT](https://arxiv.org/abs/2203.02155) / RLHF | Made LLMs follow instructions; the recipe behind every modern chat model |
| 2022 | OpenAI: [ChatGPT](https://openai.com/blog/chatgpt) (Nov 30) | Brought LLMs to the mainstream; 100M users in two months |
| 2023 | Meta: [LLaMA](https://arxiv.org/abs/2302.13971) / Llama 2 | Open-weight models matching proprietary performance; democratized access |
| 2023 | OpenAI: [GPT-4](https://arxiv.org/abs/2303.08774) | Multimodal input; substantially improved reasoning |
| 2023 | Anthropic: Claude 1 and 2 | [Constitutional AI](https://arxiv.org/abs/2212.08073) alignment approach; 100K-token context |
| 2024 | Anthropic: Claude 3 family | Multimodal; tiered model lineup (Haiku/Sonnet/Opus) |
| 2024 | OpenAI: [o1](https://openai.com/index/learning-to-reason-with-llms/) reasoning model | Test-time compute scaling; chain-of-thought as a training objective |
| 2024 | Anthropic: [Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) | Open standard for tool integration; addressing the N x M problem |
| 2025 | DeepSeek: [R1](https://arxiv.org/abs/2501.12948) | Open-source reasoning model; RL-only training without SFT |
| 2025 | Meta: Llama 4 | Natively multimodal; mixture-of-experts; 10M-token context |

## What the History Tells Us

Several patterns recur throughout AI's history, and they are worth keeping in mind when building systems today:

**Hype cycles are structural, not accidental.** Every major AI advance has been followed by inflated expectations, then disappointment, then a quieter period of genuine progress. The current wave is not immune. The practical response is not cynicism but calibration: evaluate capabilities empirically (see [Evals and Testing](05-evals.md)) rather than relying on benchmark claims or demo videos.

**The winning approaches tend to be simpler and more scalable.** McCulloch-Pitts neurons beat symbolic logic for learning. Backpropagation beat hand-designed feature engineering. Transformers beat LSTMs by being more parallelizable. RLHF beat manual rule-writing for alignment. At each step, the approach that scaled better with data and compute won, even if it was less elegant or theoretically principled.

**Infrastructure determines what is possible.** AlexNet happened because GPUs existed and ImageNet existed. GPT-3 happened because cloud-scale compute and web-scale data existed. The transformer succeeded partly because its architecture maps well onto GPU hardware. When evaluating new approaches, look at what they require in terms of data, compute, and tooling --- not just what they achieve on benchmarks.

**The ELIZA effect never went away.** Users attributed understanding to a 200-line pattern-matcher in 1966. They attribute understanding to LLMs today. The systems are incomparably more capable now, but the fundamental dynamic is the same: fluent output does not imply genuine understanding. Building reliable systems means designing for the failures that follow from this mismatch, not pretending it does not exist. See [Safety, Privacy, and Security](06-safety-privacy-security.md) for the practical implications.

## References

- McCulloch & Pitts, "A Logical Calculus of the Ideas Immanent in Nervous Activity" (1943). https://doi.org/10.1007/BF02478259
- Turing, "Computing Machinery and Intelligence" (1950). https://doi.org/10.1093/mind/LIX.236.433
- Dartmouth AI Conference proposal (1955). https://home.dartmouth.edu/about/artificial-intelligence-ai-coined-dartmouth
- Rosenblatt, "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain" (1958). https://www.ling.upenn.edu/courses/cogs501/Rosenblatt1958.pdf
- Weizenbaum, "ELIZA — A Computer Program for the Study of Natural Language Communication Between Man and Machine" (1966). https://www.csee.umbc.edu/courses/331/papers/eliza.html
- Minsky & Papert, *Perceptrons* (1969). MIT Press.
- Lighthill, "Artificial Intelligence: A General Survey" (1973). https://en.wikipedia.org/wiki/Lighthill_report
- Rumelhart, Hinton & Williams, "Learning representations by back-propagating errors" (1986). https://www.nature.com/articles/323533a0
- Hochreiter & Schmidhuber, "Long Short-Term Memory" (1997). https://doi.org/10.1162/neco.1997.9.8.1735
- Cortes & Vapnik, "Support-Vector Networks" (1995). https://doi.org/10.1007/BF00994018
- Hinton, Osindero & Teh, "A Fast Learning Algorithm for Deep Belief Nets" (2006). https://doi.org/10.1162/neco.2006.18.7.1527
- Krizhevsky, Sutskever & Hinton, "ImageNet Classification with Deep Convolutional Neural Networks" (2012). https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html
- Mikolov et al., "Efficient Estimation of Word Representations in Vector Space" (2013). https://arxiv.org/abs/1301.3781
- Sutskever, Vinyals & Le, "Sequence to Sequence Learning with Neural Networks" (2014). https://arxiv.org/abs/1409.3215
- Bahdanau, Cho & Bengio, "Neural Machine Translation by Jointly Learning to Align and Translate" (2014). https://arxiv.org/abs/1409.0473
- Goodfellow et al., "Generative Adversarial Nets" (2014). https://arxiv.org/abs/1406.2661
- He et al., "Deep Residual Learning for Image Recognition" (2015). https://arxiv.org/abs/1512.03385
- Vaswani et al., "Attention Is All You Need" (2017). https://arxiv.org/abs/1706.03762
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (2018). https://arxiv.org/abs/1810.04805
- Radford et al., "Improving Language Understanding by Generative Pre-Training" (GPT-1, 2018). https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
- Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019). https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- Brown et al., "Language Models are Few-Shot Learners" (GPT-3, 2020). https://arxiv.org/abs/2005.14165
- Ouyang et al., "Training language models to follow instructions with human feedback" (InstructGPT, 2022). https://arxiv.org/abs/2203.02155
- Bai et al., "Constitutional AI: Harmlessness from AI Feedback" (2022). https://arxiv.org/abs/2212.08073
- Touvron et al., "LLaMA: Open and Efficient Foundation Language Models" (2023). https://arxiv.org/abs/2302.13971
- OpenAI, "GPT-4 Technical Report" (2023). https://arxiv.org/abs/2303.08774
- OpenAI, "Learning to Reason with LLMs" (o1, 2024). https://openai.com/index/learning-to-reason-with-llms/
- Anthropic, "Introducing the Model Context Protocol" (2024). https://www.anthropic.com/news/model-context-protocol
- DeepSeek, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (2025). https://arxiv.org/abs/2501.12948
- History of artificial intelligence (Wikipedia, comprehensive timeline). https://en.wikipedia.org/wiki/History_of_artificial_intelligence
- AI winter (Wikipedia, funding cycles). https://en.wikipedia.org/wiki/AI_winter

---
[Contents](README.md) | [Prev](10-reading-list.md) | [Next](README.md)
