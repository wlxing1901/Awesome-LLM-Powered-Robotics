# Awesome LLM-Powered Robotics: Bridging Foundation Models and Physical Embodiment [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

<img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
<img src="https://img.shields.io/badge/PRs-Welcome-brightgreen" alt="PRs Welcome">

A curated list of awesome resources at the intersection of large language models (LLMs), robotics, and embodied AI, inspired by [awesome-computer-vision](https://github.com/jbhuang0604/awesome-computer-vision).

This repository focuses on bridging the gap between foundation models like LLMs and physical embodiment in robotics. It includes research papers, projects, and resources that explore how language models can enhance robotic capabilities, enable more natural human-robot interaction, and advance embodied AI.

If you find this repository useful, please consider giving it a star ⭐️ to show your support!

Contributions are welcome! Please feel free to submit a pull request to add more resources.

## Contents
- [Papers](#papers)
  - [Vision-Language-Action Models](#vision-language-action-models)
  - [3D Information Integration](#3d-information-integration)
  - [Embodied Chain-of-Thought, In-context Learning and Reasoning](#embodied-chain-of-thought-in-context-learning-and-reasoning)
  - [Planning and Task Planning](#planning-and-task-planning)
  - [Diffusion Models and Diffusion Policies](#diffusion-models-and-diffusion-policies)
  - [Learning from Video](#learning-from-video)
  - [Large Model Safety, Attacks, and Robustness in Embodied AI](#large-model-safety-attacks-and-robustness-in-embodied-ai)
  - [LLMs with RL or World Model](#llms-with-rl-or-world-model)
  - [Robotics Agent Applications](#robotics-agent-applications)
- [Simulators, Datasets and Benchmarks](#simulators-datasets-and-benchmarks)
- [Tutorials, Talks and Workshops](#tutorials-talks-and-workshops)

## Papers

### Vision-Language-Action Models

* **RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control** [arXiv 2023]
  [[Paper](https://robotics-transformer2.github.io/assets/rt2.pdf)]
  [[Project Page](https://robotics-transformer2.github.io/)]

* **PaLM-E: An Embodied Multimodal Language Model** [ICML 2023]
  [[Paper](https://arxiv.org/abs/2303.03378)]
  [[Project Page](https://palm-e.github.io)]

* **VIMA: General Robot Manipulation with Multimodal Prompts** [ICML 2023]
  [[Paper](https://arxiv.org/abs/2210.03094)]
  [[Project Page](https://vimalabs.github.io/)]
  [[Code](https://github.com/vimalabs/VIMA)]

* **CLIPort: What and Where Pathways for Robotic Manipulation** [CoRL 2021]
  [[Paper](https://arxiv.org/pdf/2109.12098.pdf)]
  [[Project Page](https://cliport.github.io/)]
  [[Code](https://github.com/cliport/cliport)]

* **OpenVLA: An Open-Source Vision-Language-Action Model** [arXiv 2024]
  [[Paper](https://arxiv.org/abs/2406.09246)]
  [[Project Page](https://openvla.github.io/)]
  [[Code](https://github.com/UMass-Foundation-Model/OpenVLA)]

* **3D-VLA: A 3D Vision-Language-Action Generative World Model** [arXiv 2024]
  [[Paper](https://arxiv.org/abs/2403.09631)]
  [[Project Page](https://vis-www.cs.umass.edu/3dvla/)]

* **Octo: An Open-Source Generalist Robot Policy** [arXiv 2024]
  [[Paper](https://arxiv.org/abs/2405.12213)]
  [[Project Page](https://octo-models.github.io/)]

* **LLARVA: Vision-Action Instruction Tuning Enhances Robot Learning** [arXiv 2024]
  [[Paper](https://arxiv.org/abs/2406.11815)]

* **LLaRA: Supercharging Robot Learning Data for Vision-Language Policy** [arXiv 2024]
  [[Paper](https://arxiv.org/abs/2406.20095)]

* **Open X-Embodiment: Robotic Learning Datasets and RT-X Models** [arXiv 2024]
  [[Paper](https://arxiv.org/abs/2310.08864)]
  [[Project Page](https://robotics-transformer-x.github.io/)]

* **QUAR-VLA: Vision-Language-Action Model for Quadruped Robots** [arXiv 2024]
  [[Paper](https://arxiv.org/abs/2312.14457)]

* **Bi-VLA: Vision-Language-Action Model-Based System for Bimanual Robotic Dexterous Manipulations** [arXiv 2024]
  [[Paper](https://arxiv.org/abs/2405.06039)]

* **TinyVLA: Towards Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation** [arXiv 2024]
  [[Paper](https://arxiv.org/abs/2409.12514)]
  [[Project Page](https://tiny-vla.github.io/)]

### 3D Information Integration

* **3D-LLM: Injecting the 3D World into Large Language Models** [NeurIPS 2023]
  [[Paper](https://arxiv.org/abs/2307.12981)]
  [[Project Page](https://vis-www.cs.umass.edu/3dllm/)]
  [[Code](https://github.com/UMass-Foundation-Model/3D-LLM)]

* **VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models** [arXiv 2023]
  [[Paper](https://arxiv.org/abs/2307.05973)]
  [[Project Page](https://voxposer.github.io/)]

* **CLIP-Fields: Weakly Supervised Semantic Fields for Robotic Memory** [RSS 2023]
  [[Paper](https://arxiv.org/abs/2210.05663)]
  [[Project Page](https://mahis.life/clip-fields)]
  [[Code](https://github.com/notmahi/clip-fields)]

* **LLM-Grounder: Open-Vocabulary 3D Visual Grounding with Large Language Model as an Agent** [arXiv 2023]
  [[Paper](https://arxiv.org/pdf/2309.12311.pdf)]

* **3D-VisTA: Pre-trained Transformer for 3D Vision and Text Alignment** [ICCV 2023]
  [[Paper](https://arxiv.org/abs/2308.04352)]
  [[Project Page](https://3d-vista.github.io/)]
  [[Code](https://github.com/3d-vista/3D-VisTA)]

* **ConceptFusion: Open-set Multimodal 3D Mapping** [arXiv 2023]
  [[Paper](https://arxiv.org/pdf/2302.07241.pdf)]
  [[Project Page](https://concept-fusion.github.io/)]

* **D3Fields: Dynamic 3D Descriptor Fields for Zero-Shot Generalizable Robotic Manipulation** [arXiv 2023]
  [[Paper](https://arxiv.org/pdf/2309.16118.pdf)]
  [[Project Page](https://robopil.github.io/d3fields/)]

* **ReKep: Spatio-Temporal Reasoning of Relational Keypoint Constraints for Robotic Manipulation** [arXiv 2023]
  [[Paper](https://arxiv.org/pdf/2409.01652.pdf)]
  [[Project Page](https://rekep-robot.github.io)]
  [[Code](https://github.com/huangwl18/ReKep)]


### Embodied Chain-of-Thought, In-context Learning and Reasoning

* **Reasoning with Language Model is Planning with World Model** [arXiv 2023]
  [[Paper](https://arxiv.org/pdf/2305.14992.pdf)]

* **Do Embodied Agents Dream of Pixelated Sheep?: Embodied Decision Making using Language Guided World Modelling** [ICML 2023]
  [[Paper](https://openreview.net/attachment?id=Rm5Qi57C5I&name=pdf)]

* **Language Models Meet World Models: Embodied Experiences Enhance Language Models** [arXiv 2023]
  [[Paper](https://arxiv.org/pdf/2305.10626.pdf)]
  [[Code](https://github.com/szxiangjn/world-model-for-language-model)]

* **Inner Monologue: Embodied Reasoning through Planning with Language Models** [CoRL 2022]
  [[Paper](https://openreview.net/pdf?id=3R3Pz5i0tye)]
  [[Project Page](https://innermonologue.github.io/)]

* **Grounding Language Models to Images for Multimodal Generation** [ICLR 2023]
  [[Paper](https://arxiv.org/abs/2301.13823)]
  [[Project Page](https://grounded-language-models.github.io/)]

* **Socratic Models: Composing Zero-Shot Multimodal Reasoning with Language** [ICLR 2023]
  [[Paper](https://arxiv.org/abs/2204.00598)]
  [[Project Page](https://socraticmodels.github.io/)]

### Planning and Task Planning

* **Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents** [ICML 2022]
  [[Paper](https://arxiv.org/pdf/2201.07207.pdf)]
  [[Project Page](https://wenlong.page/language-planner/)]
  [[Code](https://github.com/huangwl18/language-planner)]

* **LLM-Planner: Few-Shot Grounded Planning for Embodied Agents with Large Language Models** [ICCV 2023]
  [[Paper](https://arxiv.org/pdf/2212.04088.pdf)]
  [[Project Page](https://dki-lab.github.io/LLM-Planner/)]
  [[Code](https://github.com/OSU-NLP-Group/LLM-Planner)]

* **Embodied Task Planning with Large Language Models** [arXiv 2023]
  [[Paper](https://arxiv.org/pdf/2307.01848.pdf)]
  [[Project Page](https://gary3410.github.io/TaPA/)]
  [[Code](https://github.com/Gary3410/TaPA)]

* **Code as Policies: Language Model Programs for Embodied Control** [arXiv 2022]
  [[Paper](https://arxiv.org/pdf/2209.07753)]
  [[Project Page](https://code-as-policies.github.io/)]

* **SayCan: Do As I Can, Not As I Say: Grounding Language in Robotic Affordances** [arXiv 2022]
  [[Paper](https://arxiv.org/pdf/2204.01691.pdf)]
  [[Project Page](https://say-can.github.io/)]

* **Leveraging Pre-trained Large Language Models to Construct and Utilize World Models for Model-based Task Planning** [NeurIPS 2023]
  [[Paper](https://openreview.net/forum?id=zDbsSscmuj)]
  [[Project Page](https://guansuns.github.io/pages/llm-dm/)]
  [[Code](https://github.com/GuanSuns/LLMs-World-Models-for-Planning)]

### Diffusion Models and Diffusion Policies

* **Eureka: Human-Level Reward Design via Coding Large Language Models** [NeurIPS 2023 Workshop ALOE Spotlight]
  [[Paper](https://eureka-research.github.io/assets/eureka_paper.pdf)]
  [[Project Page](https://eureka-research.github.io/)]
  [[Code](https://github.com/eureka-research/Eureka)]

* **Text2Reward: Dense Reward Generation with Language Models for Reinforcement Learning** [ICLR 2024 Spotlight]
  [[Paper](https://openreview.net/pdf?id=tUM39YTRxH)]

* **Diffusion Policy: Visuomotor Policy Learning via Action Diffusion** [NeurIPS 2023]
  [[Paper](https://arxiv.org/abs/2303.04137)]
  [[Project Page](https://diffusion-policy.cs.columbia.edu/)]
  [[Code](https://github.com/real-stanford/diffusion_policy)]

* **Conditional Behavior Transformers: Conditional Generation of Robot Trajectories from Transformer Models** [ICRA 2023]
  [[Paper](https://arxiv.org/abs/2302.11855)]
  [[Project Page](https://conditional-behavior-transformer.github.io/)]

### Learning from Video

* **Learning Interactive Real-World Simulators** [ICLR 2024 Outstanding Paper]
  [[Paper](https://openreview.net/attachment?id=sFyTZEqmUY&name=pdf)]
  [[Project Page](https://universal-simulator.github.io/unisim/)]

* **Voyager: An Open-Ended Embodied Agent with Large Language Models** [NeurIPS 2023 Workshop ALOE Spotlight]
  [[Paper](https://arxiv.org/abs/2305.16291)]
  [[Project Page](https://voyager.minedojo.org/)]
  [[Code](https://github.com/MineDojo/Voyager)]

* **RT-Trajectory: Robotic Task Generalization via Hindsight Trajectory Sketches** [arXiv 2023]
  [[Paper](https://arxiv.org/pdf/2311.01977.pdf)]

* **Vid2Robot: End-to-End Video-to-Robot Task Transfer** [arXiv 2023]
  [[Paper](https://vid2robot.github.io/vid2robot.pdf)]
  [[Project Page](https://vid2robot.github.io/)]

* **Learning from Demonstration in the Wild** [ICRA 2018]
  [[Paper](https://arxiv.org/abs/1704.03732)]

### Large Model Safety, Attacks, and Robustness in Embodied AI

* **Robots Enact Malignant Stereotypes** [FAccT 2022]
  [[Paper](https://arxiv.org/abs/2207.11569)]
  [[Project Page](https://sites.google.com/view/robots-enact-stereotypes/home)]

* **Highlighting the Safety Concerns of Deploying LLMs/VLMs in Robotics** [arXiv 2024]
  [[Paper](https://arxiv.org/abs/2402.10340)]

* **LLM-Driven Robots Risk Enacting Discrimination, Violence, and Unlawful Actions** [arXiv 2024]
  [[Paper](https://arxiv.org/abs/2406.08824)]

### LLMs with RL or World Model

* **Guiding Pretraining in Reinforcement Learning with Large Language Models** [ICML 2023]
  [[Paper](https://openreview.net/attachment?id=63704LH4v5&name=pdf)]

* **Language Reward Modulation for Pretraining Reinforcement Learning** [arXiv 2023]
  [[Paper](https://arxiv.org/pdf/2308.12270.pdf)]
  [[Code](https://github.com/ademiadeniji/lamp)]

* **STARLING: Self-supervised Training of Text-based Reinforcement Learning Agent with Large Language Models** [arXiv 2023]
  [[Paper](https://openreview.net/pdf?id=LXiG2WqKXR)]

* **Informing Reinforcement Learning Agents by Grounding Natural Language to Markov Decision Processes** [arXiv 2023]
  [[Paper](https://openreview.net/pdf?id=P4op21eju0)]

* **Learning to Model the World with Language** [arXiv 2023]
  [[Paper](https://openreview.net/pdf?id=eWLOoaShEH)]

### Robotics Agent Applications

* **Mobile-Agent: Autonomous Multi-Modal Mobile Device Agent with Visual Perception** [arXiv 2024]
  [[Paper](https://arxiv.org/abs/2401.16158)]
  [[Code](https://github.com/X-PLUG/MobileAgent)]

* **ChatGPT for Robotics: Design Principles and Model Abilities** [arXiv 2023]
  [[Paper](https://www.microsoft.com/en-us/research/uploads/prod/2023/02/ChatGPT___Robotics.pdf)]
  [[Project Page](https://www.microsoft.com/en-us/research/group/autonomous-systems-group-robotics/articles/chatgpt-for-robotics/)]

* **Do As I Can, Not As I Say: Grounding Language in Robotic Affordances** [arXiv 2022]
  [[Paper](https://arxiv.org/pdf/2204.01691.pdf)]
  [[Project Page](https://say-can.github.io/)]

* **TidyBot: Personalized Robot Assistance with Large Language Models** [arXiv 2023]
  [[Paper](https://arxiv.org/abs/2305.05658)]
  [[Project Page](https://tidybot.cs.princeton.edu/)]
  [[Code](https://github.com/jimmyyhwu/tidybot)]

* **Instruct2Act: Mapping Multi-modality Instructions to Robotic Actions with Large Language Model** [arXiv 2023]
  [[Paper](https://arxiv.org/pdf/2305.11176.pdf)]
  [[Code](https://github.com/OpenGVLab/Instruct2Act)]

* **RoboCat: A self-improving robotic agent** [arXiv 2023]
  [[Paper](https://arxiv.org/pdf/2306.11706.pdf)]
  [[Project Page](https://www.deepmind.com/blog/robocat-a-self-improving-robotic-agent)]

## Simulators, Datasets and Benchmarks

### Simulators
- **AI2-THOR**: ["AI2-THOR: An Interactive 3D Environment for Visual AI"](https://arxiv.org/abs/1712.05474) - Interactive 3D environment for AI agents [[Project Page](https://ai2thor.allenai.org/)]
- **iGibson**: ["iGibson 1.0: A Simulation Environment for Interactive Tasks in Large Realistic Scenes"](https://arxiv.org/abs/2012.02924) - Large scale interactive simulation environment [[Project Page](https://svl.stanford.edu/igibson/)]
- **Habitat**: ["Habitat: A Platform for Embodied AI Research"](https://arxiv.org/abs/1904.01201) - Simulation platform for embodied AI research [[Project Page](https://aihabitat.org/)]
- **BEHAVIOR**: ["BEHAVIOR: Benchmark for Everyday Household Activities in Virtual, Interactive, and Ecological Environments"](https://arxiv.org/abs/2108.03332) - Benchmark for everyday household activities [[Project Page](https://behavior.stanford.edu/)]
- **MineDojo**: ["MINEDOJO: Building Open-Ended Embodied Agents with Internet-Scale Knowledge"](https://arxiv.org/abs/2206.08853) - Open-ended embodied agent benchmark in Minecraft [[Project Page](https://minedojo.org/)]
- **RoboTHOR**: ["RoboTHOR: An Open Simulation-to-Real Embodied AI Platform"](https://arxiv.org/abs/2004.06799) - Robot simulation environment built on AI2-THOR [[Project Page](https://ai2thor.allenai.org/robothor/)]
- **ManipulaTHOR**: ["ManipulaTHOR: A Framework for Visual Object Manipulation"](https://arxiv.org/abs/2104.11213) - Physics-enabled manipulation environment [[Project Page](https://ai2thor.allenai.org/manipulathor/)]
- **BEHAVIOR**: ["BEHAVIOR Vision Suite: Customizable Dataset Generation via Simulation"](https://openaccess.thecvf.com/content/CVPR2024/papers/Ge_BEHAVIOR_Vision_Suite_Customizable_Dataset_Generation_via_Simulation_CVPR_2024_paper.pdf) - Customizable generation[[Project Page](https://behavior-visionsuite.github.io/)]

### Datasets
- **VIMA**: ["VIMA: General Robot Manipulation with Multimodal Prompts"](https://arxiv.org/abs/2210.03094) - Large-scale visual manipulation dataset [[Project Page](https://vimalabs.github.io/)]
- **SQA3D**: ["SQA3D: Situated Question Answering in 3D Scenes"](https://arxiv.org/abs/2210.07474) - 3D visual question answering dataset [[Project Page](https://sqa3d.github.io/)]
- **BEHAVIOR-1K**: ["BEHAVIOR-1K: A Benchmark for Embodied AI with 1,000 Everyday Activities and Realistic Simulation"](https://arxiv.org/abs/2204.10380) - Large-scale dataset of 3D human activities [[Project Page](https://behavior.stanford.edu/behavior-1k/)]
- **RT-1 Dataset**: ["RT-1: Robotics Transformer for Real-World Control at Scale"](https://arxiv.org/abs/2212.06817) - Large-scale robot manipulation dataset [[Project Page](https://robotics-transformer1.github.io/)]
- **Open X-Embodiment**: ["Open X-Embodiment: Robotic Learning Datasets and RT-X Models"](https://arxiv.org/abs/2310.08864) - Large multi-embodiment robotics dataset [[Project Page](https://robotics-transformer-x.github.io/)]

### Benchmarks  
- **ALFRED**: ["ALFRED: A Benchmark for Interpreting Grounded Instructions for Everyday Tasks"](https://arxiv.org/abs/1912.01734) - Benchmark for grounded language understanding in 3D environments [[Project Page](https://askforalfred.com/)]
- **ALFWorld**: ["ALFWorld: Aligning Text and Embodied Environments for Interactive Learning"](https://arxiv.org/abs/2010.03768) - Benchmark for grounding language in interactive environments [[Project Page](https://alfworld.github.io/)]
- **RoboTHOR**: ["RoboTHOR: An Open Simulation-to-Real Embodied AI Platform"](https://arxiv.org/abs/2004.06799) - Benchmark for visual navigation [[Challenge Page](https://ai2thor.allenai.org/robothor/challenge/)]
- **BEHAVIOR**: ["BEHAVIOR: Benchmark for Everyday Household Activities in Virtual, Interactive, and Ecological Environments"](https://arxiv.org/abs/2108.03332) - Benchmark for everyday household activities [[Project Page](https://behavior.stanford.edu/)]
- **Habitat Challenge**: ["Habitat 2.0: Training Home Assistants to Rearrange their Habitat"](https://arxiv.org/abs/2106.14405) - Benchmarks for embodied AI [[Challenge Page](https://aihabitat.org/challenge/)]
- **VIMA-Bench**: ["VIMA: General Robot Manipulation with Multimodal Prompts"](https://arxiv.org/abs/2210.03094) - Benchmark for vision-language manipulation [[GitHub](https://github.com/vimalabs/VimaBench)]

## Tutorials, Talks and Workshops

- **Embodied AI Workshop Series**
  - [Embodied AI Workshop @ CVPR 2024](https://embodied-ai.org/)
  - [Embodied AI Workshop @ CVPR 2023](https://embodied-ai.org/cvpr2023/)
  - [Embodied AI Workshop @ CVPR 2022](https://embodied-ai.org/cvpr2022/)
  - [Embodied AI Workshop @ CVPR 2021](https://embodied-ai.org/cvpr2021/)
  - [Embodied AI Workshop @ CVPR 2020](https://embodied-ai.org/cvpr2020/)

- **ICML Workshops**
  - [Multi-modal Foundation Model meets Embodied AI @ ICML 2024](https://icml-mfm-eai.github.io/)

- **CoRL Workshops**
  - [LangRob: Workshop on Language and Robot Learning @ CoRL 2023](https://sites.google.com/view/langrob-corl23/)

- [MIT CSAIL Embodied Intelligence Seminar Series](https://ei.csail.mit.edu/seminars.html)
