# Awesome Multi-Agent Autonomous Driving 🚗 🚙 🚓 🚕 🏎️

![MAAD](./src/maad.png)
<!-- <div align="center">
  <a href="./src/wechat.jpg">
    <img src="https://img.shields.io/badge/WeChat-MAADResearch-brightgreen?logo=wechat&logoColor=white" alt="WeChat">
  </a>
</div> -->

This is a repository for collecting resources about **Multi-Agent Autonomous Driving (MAAD)**. Different from single-agent autonomous driving which mainly focus on enhancing the driving capabilities of a single vehicle, MAAD focuses on the collaboration and interaction between multiple agents including vehicles and infrastructure.

If you want to understand the **FULL-STACK** technology of **MULTI-AGENT AUTONOMOUS DRIVING**, then this repo is definitely for you!

## Come and Join Us! 👊🇨🇳🔥

### Contribution
*Feel free to pull requests or contact us if you find any related papers that are not included here.*

The process to submit a pull request is as follows:
1. Fork the project into your own repository.
2. Add the Title, Paper link, Conference, Project/Code link in `papers.md` using the following format:
```markdown
  `[Journal/Conference]` Paper Title [Code/Project](Code/Project link)
```
3. Submit the pull request to this branch.


<!-- ### Join Our Community

In addition, if you want to **join our community** for discussion, sharing, connections, and potential collaborations, please scan the [WeChat QR code](./src/wechat.jpg). [<img src="https://img.shields.io/badge/WeChat-MAADResearch-brightgreen?logo=wechat&logoColor=white" alt="WeChat">](./src/wechat.jpg) -->

## Table of Contents


- [Awesome Multi-Agent Autonomous Driving 🚗 🚙 🚓 🚕 🏎️](#awesome-multi-agent-autonomous-driving-----️)
  - [Come and Join Us! 👊🇨🇳🔥](#come-and-join-us-)
    - [Contribution](#contribution)
  - [Table of Contents](#table-of-contents)
  - [Related Materials](#related-materials)
    - [Surveys](#surveys)
    - [Github Repos](#github-repos)
  - [Paper Collection](#paper-collection)
    - [Perception](#perception)
    - [Decision-Making](#decision-making)
    - [Planning](#planning)
    - [Communication](#communication)
    - [End-to-End](#end-to-end)
    - [Dataset and Simulator](#dataset-and-simulator)
      - [Dataset](#dataset)
      - [Simulator](#simulator)
    - [Security and Robustness](#security-and-robustness)
  - [Star History](#star-history)


## Related Materials

###  Surveys
1. `[TPAMI'24]` 3D Object Detection From Images for Autonomous Driving: A Survey [[PDF](https://ieeexplore.ieee.org/document/10373157/?arnumber=10373157)]
2. `[TITS'24]` A Survey on Recent Advancements in Autonomous Driving Using Deep Reinforcement Learning: Applications, Challenges, and Solutions [[PDF](https://ieeexplore.ieee.org/document/10682977/?arnumber=10682977)]
3. `[ESWA]` Autonomous driving system: A comprehensive survey [[PDF](https://linkinghub.elsevier.com/retrieve/pii/S0957417423033389)]
4. `[TPAMI'24]` Delving Into the Devils of Bird's-Eye-View Perception: A Review, Evaluation and Recipe [[PDF](https://ieeexplore.ieee.org/document/10321736/?arnumber=10321736)]
5. `[TPAMI]` End-to-End Autonomous Driving: Challenges and Frontiers [[PDF](https://ieeexplore.ieee.org/document/10614862/?arnumber=10614862), [Code](https://github.com/OpenDriveLab/End-to-end-Autonomous-Driving)] ![](https://img.shields.io/github/stars/OpenDriveLab/End-to-end-Autonomous-Driving.svg?style=social&label=Star&maxAge=2592000)
6. `[arXiv]` LLM4Drive: A Survey of Large Language Models for Autonomous Driving [[PDF](http://arxiv.org/abs/2311.01043), [Code](https://github.com/Thinklab-SJTU/Awesome-LLM4AD)] ![](https://img.shields.io/github/stars/Thinklab-SJTU/Awesome-LLM4AD.svg?style=social&label=Star&maxAge=2592000)
7. `[arXiv]` Multi-Agent Autonomous Driving Systems with Large Language Models: A Survey of Recent Advances [[PDF](http://arxiv.org/abs/2502.16804), [Code](https://anonymous.4open.science/r/LLM-based_Multi-agent_ADS-3A5C/README.md)]
8. `[WACV Workshop]` A Survey on Multimodal Large Language Models for Autonomous Driving [[PDF](https://openaccess.thecvf.com/content/WACV2024W/LLVM-AD/papers/Cui_A_Survey_on_Multimodal_Large_Language_Models_for_Autonomous_Driving_WACVW_2024_paper.pdf)]
9. `[arXiv]` A Survey of Reasoning with Foundation Models [[PDF](https://arxiv.org/abs/2312.11562), [Code](https://github.com/reasoning-survey/Awesome-Reasoning-Foundation-Models)] ![](https://img.shields.io/github/stars/reasoning-survey/Awesome-Reasoning-Foundation-Models.svg?style=social&label=Star&maxAge=2592000)
10. `[arXiv]` Collaborative Perception for Connected and Autonomous Driving: Challenges, Possible Solutions and Opportunities [[PDF](https://arxiv.org/abs/2401.01544)]
11. `[Annual Review of Control, Robotics, and Autonomous Systems]` Planning and decision-making for autonomous vehicles [[PDF](https://www.annualreviews.org/content/journals/10.1146/annurev-control-060117-105157)]
12. `[Chinese Journal of Mechanical Engineering]` Planning and Decision-making for Connected Autonomous Vehicles at Road Intersections: A Review [[PDF](https://cjme.springeropen.com/articles/10.1186/s10033-021-00639-3)]
13. `[COMST'22]` A Survey of Collaborative Machine Learning Using 5G Vehicular Communications [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9706268)]
14. `[arXiv]` Collaborative Perception for Connected and Autonomous Driving: Challenges, Possible Solutions and Opportunities [[PDF](https://arxiv.org/abs/2401.01544)]
15. `[Proceedings of the IEEE]` 6G for Vehicle-to-Everything (V2X) Communications: Enabling Technologies, Challenges, and Opportunities [[PDF](https://ieeexplore.ieee.org/document/9779322)]
16. `[arXiv'25]` Vision-Language-Action Models for Autonomous Driving: Past, Present, and Future [[PDF](https://arxiv.org/abs/2512.16760)]
17. `[ICCV'25 Workshop]` A Survey on Vision-Language-Action Models for Autonomous Driving [[PDF](https://openaccess.thecvf.com/content/ICCV2025W/WDFM-AD/papers/Jiang_A_Survey_on_Vision-Language-Action_Models_for_Autonomous_Driving_ICCVW_2025_paper.pdf)]
18. `[IEEE Trans'25]` Large (Vision) Language Models for Autonomous Vehicles: Current Trends and Future Directions [[PDF](https://ieeexplore.ieee.org/abstract/document/11264491/)]
19. `[IEEE Trans'25]` Multi-agent Reinforcement Learning for Connected and Automated Vehicles Control: Recent Advancements and Future Prospects [[PDF](https://ieeexplore.ieee.org/abstract/document/11016811/)]
20. `[IEEE'25]` Cooperative Perception for Automated Driving: A Survey of Algorithms, Applications, and Future Directions [[PDF](https://ieeexplore.ieee.org/abstract/document/11186152/)]
21. `[arXiv'25]` Collaborative Perception Datasets for Autonomous Driving: A Review [[PDF](https://arxiv.org/abs/2504.12696)]
22. `[arXiv'25]` Cooperative Safety Intelligence in V2X-Enabled Transportation: A Survey [[PDF](https://arxiv.org/abs/2512.00490)]
23. `[arXiv'25]` Recent Advances in Multi-Agent Human Trajectory Prediction [[PDF](https://arxiv.org/abs/2506.14831)]




###  Github Repos
1. [Awesome Autonomous Driving](https://github.com/PeterJaq/Awesome-Autonomous-Driving) ![](https://img.shields.io/github/stars/PeterJaq/Awesome-Autonomous-Driving.svg?style=social&label=Star&maxAge=2592000)
2. [Autonomous Driving Datasets](https://github.com/MingyuLiu1/autonomous_driving_datasets) ![](https://img.shields.io/github/stars/MingyuLiu1/autonomous_driving_datasets.svg?style=social&label=Star&maxAge=2592000)
3. [Awesome 3D Object Detection for Autonomous Driving](https://github.com/PointsCoder/Awesome-3D-Object-Detection-for-Autonomous-Driving) ![](https://img.shields.io/github/stars/PointsCoder/Awesome-3D-Object-Detection-for-Autonomous-Driving.svg?style=social&label=Star&maxAge=2592000)
4. [CVPR 2024 Papers on Autonomous Driving](https://github.com/autodriving-heart/CVPR-2024-Papers-Autonomous-Driving) ![](https://img.shields.io/github/stars/autodriving-heart/CVPR-2024-Papers-Autonomous-Driving.svg?style=social&label=Star&maxAge=2592000)
5. [End-to-End Autonomous Driving](https://github.com/Pranav-chib/End-to-End-Autonomous-Driving) ![](https://img.shields.io/github/stars/Pranav-chib/End-to-End-Autonomous-Driving.svg?style=social&label=Star&maxAge=2592000)
6. [End-to-End Autonomous Driving](https://github.com/OpenDriveLab/End-to-end-Autonomous-Driving) (OpenDriveLab) ![](https://img.shields.io/github/stars/OpenDriveLab/End-to-end-Autonomous-Driving.svg?style=social&label=Star&maxAge=2592000)
7. [Collaborative Perception](https://github.com/Little-Podi/Collaborative_Perception) ![](https://img.shields.io/github/stars/Little-Podi/Collaborative_Perception.svg?style=social&label=Star&maxAge=2592000)
8. [Vision Language Models in Autonomous Driving and ITS](https://github.com/ge25nab/Awesome-VLM-AD-ITS) ![](https://img.shields.io/github/stars/ge25nab/Awesome-VLM-AD-ITS.svg?style=social&label=Star&maxAge=2592000)


## Paper Collection
<!-- Please refer to [this page](./papers.md) for the full list of the papers.

- [Perception](papers.md#perception)
- [Decision-Making](papers.md#decision-making)
- [Planning](papers.md#planning)
- [Communication](papers.md#communication)
- [End-to-End](papers.md#end-to-end)
- [Dataset and Simulator](papers.md#dataset-and-simulator)
- [Security](papers.md#security)

 -->

### Perception
1. `[ICCV'25 Workshop]` Learning 3D Perception from Others' Predictions [[PDF](https://drivex-workshop.github.io/iccv2025/)]
2. `[ICCV'25 Workshop]` RG-Attn: Radian Glue Attention for Multi-modal Multi-agent Cooperative Perception [[PDF](https://drivex-workshop.github.io/iccv2025/)]
3. `[ICCV'25 Workshop]` MIC-BEV: Infrastructure-Based Multi-Camera Bird's-Eye-View Perception Transformer for 3D Object Detection [[PDF](https://drivex-workshop.github.io/iccv2025/)]
4. `[ICCV'25 Workshop]` SlimComm: Doppler-Guided Sparse Queries for Bandwidth-Efficient Cooperative 3-D Perception [[PDF](https://drivex-workshop.github.io/iccv2025/)]
5. `[ICCV'25 Workshop]` D3FNet: A Differential Attention Fusion Network for Fine-Grained Road Structure Extraction in Remote Perception Systems [[PDF](https://drivex-workshop.github.io/iccv2025/)]
6. `[ICCV'25 Workshop]` Understanding What Vision-Language Models See in Traffic: PixelSHAP for Object-Level Attribution in Autonomous Driving [[PDF](https://drivex-workshop.github.io/iccv2025/)]
7. `[ICCV'25 Workshop]` Scene-Aware Location Modeling for Data Augmentation in Automotive Object Detection [[PDF](https://drivex-workshop.github.io/iccv2025/)]
8. `[ICCV'25 Workshop]` Cross-camera Monocular 3D Detection for Autonomous Racing [[PDF](https://drivex-workshop.github.io/iccv2025/)]
9. `[ICRA'25]` CoopDETR: A Unified Cooperative Perception Framework for 3D Detection via Object Query [[PDF](https://arxiv.org/pdf/2502.19313)]
10. `[arXiv'25]` V2V-LLM: Vehicle-to-Vehicle Cooperative Autonomous Driving with Multi-Modal Large Language Models [[PDF](https://arxiv.org/pdf/2502.09980)] [[Code](https://github.com/eddyhkchiu/V2V-LLM)] [[Webpage](https://eddyhkchiu.github.io/v2vllm.github.io/)] ![](https://img.shields.io/github/stars/eddyhkchiu/V2V-LLM.svg?style=social&label=Star&maxAge=2592000)
11. `[arXiv'25]` V2V-GoT: Vehicle-to-Vehicle Cooperative Autonomous Driving with Multimodal Large Language Models and Graph-of-Thoughts [[PDF](https://arxiv.org/abs/2509.18053)]
12. `[Electronics'25]` Vision-Language Models for Autonomous Driving: CLIP-based Dynamic Scene Understanding [[PDF](https://www.mdpi.com/2079-9292/14/7/1282)]
5. `[CVPR'25]` CoSDH: Communication-Efficient Collaborative Perception via Supply-Demand Awareness and Intermediate-Late Hybridization [[PDF](https://arxiv.org/abs/2503.03430)] [[Code](https://github.com/Xu2729/CoSDH)] ![](https://img.shields.io/github/stars/Xu2729/CoSDH.svg?style=social&label=Star&maxAge=2592000)
6. `[CVPR'25]` One is Plenty: A Polymorphic Feature Interpreter for Immutable Heterogeneous Collaborative Perception [[PDF](https://arxiv.org/abs/2411.16799)]
7. `[CVPR'25]` SparseAlign: A Fully Sparse Framework for Cooperative Object Detection [[PDF](https://arxiv.org/pdf/2503.12982)]
8. `[CVPR'25]` V2X-R: Cooperative LiDAR-4D Radar Fusion for 3D Object Detection with Denoising Diffusion [[PDF](https://arxiv.org/abs/2411.08402)] [[Code](https://github.com/ylwhxht/V2X-R)] ![](https://img.shields.io/github/stars/ylwhxht/V2X-R.svg?style=social&label=Star&maxAge=2592000)
9. `[CVPR'25]` Trajectory-aware Feature Alignment for Asynchronous Multi-Agent Perception [[PDF](https://cvpr.thecvf.com/virtual/2025/poster/33270)]
10. `[ICCV'25]` V2XPnP: Vehicle-to-Everything Spatio-Temporal Fusion for Multi-Agent Perception and Prediction [[PDF](https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_V2XPnP_Vehicle-to-Everything_Spatio-Temporal_Fusion_for_Multi-Agent_Perception_and_Prediction_ICCV_2025_paper.html)]
11. `[ICCV'25]` mmCooper: A Multi-agent Multi-stage Communication-efficient and Collaboration-robust Cooperative Perception Framework [[PDF](https://openaccess.thecvf.com/content/ICCV2025/html/Liu_mmCooper_A_Multi-agent_Multi-stage_Communication-efficient_and_Collaboration-robust_Cooperative_Perception_Framework_ICCV_2025_paper.html)]
12. `[AAAI'25]` Privacy-Preserving V2X Collaborative Perception [[PDF](https://ojs.aaai.org/index.php/AAAI/article/view/32619)]
13. `[IEEE'25]` SCOPE++: Robust Multi-Agent Collaborative Perception via Spatio-Temporal Awareness [[PDF](https://ieeexplore.ieee.org/document/10839457/)]
14. `[IEEE VTC'25]` Robust Multi-Agent Collaborative Perception via Triple-Attention and Dynamic Gating [[PDF](https://ieeexplore.ieee.org/document/11174503/)]
15. `[arXiv'25]` FocalComm: Hard Instance-Aware Multi-Agent Perception [[PDF](https://arxiv.org/html/2512.13982v1)]
16. `[TITS'24]` Toward Full-Scene Domain Generalization in Multi-Agent Collaborative Bird's Eye View Segmentation for Connected and Autonomous Driving [[PDF](https://ieeexplore.ieee.org/abstract/document/10779389)]
17. `[CVPR'24]` Collaborative Semantic Occupancy Prediction with Hybrid Feature Fusion in Connected Automated Vehicles [[PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Song_Collaborative_Semantic_Occupancy_Prediction_with_Hybrid_Feature_Fusion_in_Connected_CVPR_2024_paper.pdf)] [[Code](https://github.com/rruisong/CoHFF)] ![](https://img.shields.io/github/stars/rruisong/CoHFF.svg?style=social&label=Star&maxAge=2592000)
18. `[ECCV'24]` Hetecooper: Feature Collaboration Graph for Heterogeneous Collaborative Perception [[PDF](https://eccv.ecva.net/virtual/2024/poster/2467)] 
19. `[ECCV'24]` Rethinking the Role of Infrastructure in Collaborative Perception [[PDF](https://arxiv.org/abs/2410.11259)] 
20. `[NeurIPS'24]` Learning Cooperative Trajectory Representations for Motion Forecasting [[PDF](https://arxiv.org/abs/2311.00371)] [[Code](https://github.com/AIR-THU/V2X-Graph)] ![](https://img.shields.io/github/stars/AIR-THU/V2X-Graph.svg?style=social&label=Star&maxAge=2592000)
21. `[ICLR'24]` An Extensible Framework for Open Heterogeneous Collaborative Perception [[PDF](https://arxiv.org/pdf/2401.13964)] [[Code](https://github.com/yifanlu0227/HEAL)] ![](https://img.shields.io/github/stars/yifanlu0227/HEAL.svg?style=social&label=Star&maxAge=2592000)
22. `[AAAI'24]` What Makes Good Collaborative Views? Contrastive Mutual Information Maximization for Multi-Agent Perception [[PDF](https://arxiv.org/abs/2403.10068)] [[Code](https://github.com/77SWF/CMiMC)] ![](https://img.shields.io/github/stars/77SWF/CMiMC.svg?style=social&label=Star&maxAge=2592000)
23. `[AAAI'24]` DI-V2X: Learning Domain-Invariant Representation for Vehicle-Infrastructure Collaborative 3D Object Detection [[PDF](https://arxiv.org/abs/2312.15742)] [[Code](https://github.com/Serenos/DI-V2X)] ![](https://img.shields.io/github/stars/Serenos/DI-V2X.svg?style=social&label=Star&maxAge=2592000)
24. `[AAAI'24]` DeepAccident: A Motion and Accident Prediction Benchmark for V2X Autonomous Driving [[PDF](https://arxiv.org/abs/2304.01168)] [[Code](https://github.com/tianqi-wang1996/DeepAccident)] ![](https://img.shields.io/github/stars/tianqi-wang1996/DeepAccident.svg?style=social&label=Star&maxAge=2592000)
25. `[WACV'24]` MACP: Efficient Model Adaptation for Cooperative Perception [[PDF](https://arxiv.org/abs/2310.16870)] [[Code](https://github.com/PurdueDigitalTwin/MACP)] ![](https://img.shields.io/github/stars/PurdueDigitalTwin/MACP.svg?style=social&label=Star&maxAge=2592000)
26. `[ICRA'24]` Probabilistic 3D Multi-Object Cooperative Tracking for Autonomous Driving via Differentiable Multi-Sensor Kalman Filter [[PDF](https://arxiv.org/abs/2309.14655)] [[Code](https://github.com/eddyhkchiu/DMSTrack)] ![](https://img.shields.io/github/stars/eddyhkchiu/DMSTrack.svg?style=social&label=Star&maxAge=2592000)
27. `[ICRA'24]` Robust Collaborative Perception without External Localization and Clock Devices [[PDF](https://arxiv.org/abs/2405.02965)] [[Code](https://github.com/MediaBrain-SJTU/FreeAlign)] ![](https://img.shields.io/github/stars/MediaBrain-SJTU/FreeAlign.svg?style=social&label=Star&maxAge=2592000)
28. `[ICCV'23]` Spatio-Temporal Domain Awareness for Multi-Agent Collaborative Perception [[PDF](https://openaccess.thecvf.com/content/ICCV2023/papers/Yang_Spatio-Temporal_Domain_Awareness_for_Multi-Agent_Collaborative_Perception_ICCV_2023_paper.pdf)] [[Webpage](https://ydk122024.github.io/SCOPE/)]
29. `[ICCV'23]` HM-ViT: Hetero-modal Vehicle-to-Vehicle Cooperative Perception with Vision Transformer [[PDF](https://openaccess.thecvf.com/content/ICCV2023/papers/Xiang_HM-ViT_Hetero-Modal_Vehicle-to-Vehicle_Cooperative_Perception_with_Vision_Transformer_ICCV_2023_paper.pdf)] [[Code](https://github.com/XHwind/HM-ViT)] ![](https://img.shields.io/github/stars/XHwind/HM-ViT.svg?style=social&label=Star&maxAge=2592000)
30. `[ICCV'23]` CORE: Cooperative Reconstruction for Multi-Agent Perception [[PDF](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_CORE_Cooperative_Reconstruction_for_Multi-Agent_Perception_ICCV_2023_paper.pdf)] [[Code](https://github.com/zllxot/CORE)] ![](https://img.shields.io/github/stars/zllxot/CORE.svg?style=social&label=Star&maxAge=2592000)
31. `[ICCV'23]` Among Us: Adversarially Robust Collaborative Perception by Consensus [[PDF](https://arxiv.org/abs/2303.09495)] [[Code](https://github.com/coperception/ROBOSAC)]
32. `[ICCV'23]` Spatio-Temporal Domain Awareness for Multi-Agent Collaborative Perception [[PDF](https://arxiv.org/abs/2307.13929)] [[Code](https://github.com/starfdu1418/SCOPE)] ![](https://img.shields.io/github/stars/starfdu1418/SCOPE.svg?style=social&label=Star&maxAge=2592000)
33. `[ICCV'23]` TransIFF: An Instance-Level Feature Fusion Framework for Vehicle-Infrastructure Cooperative 3D Detection with Transformers [[PDF](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_TransIFF_An_Instance-Level_Feature_Fusion_Framework_for_Vehicle-Infrastructure_Cooperative_3D_ICCV_2023_paper.pdf)] 
34. `[ICCV'23]` UMC: A Unified Bandwidth-Efficient and Multi-Resolution Based Collaborative Perception Framework [[PDF](https://arxiv.org/abs/2303.12400)] [[Code](https://github.com/ispc-lab/UMC)] ![](https://img.shields.io/github/stars/ispc-lab/UMC.svg?style=social&label=Star&maxAge=2592000)
35. `[CVPR'23]` Query-Centric Trajectory Prediction [[PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Query-Centric_Trajectory_Prediction_CVPR_2023_paper.pdf)] [[Code](https://github.com/ZikangZhou/QCNet)] ![](https://img.shields.io/github/stars/ZikangZhou/QCNet.svg?style=social&label=Star&maxAge=2592000)
36. `[CVPR'23]` Collaboration Helps Camera Overtake LiDAR in 3D Detection [[PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Hu_Collaboration_Helps_Camera_Overtake_LiDAR_in_3D_Detection_CVPR_2023_paper.pdf)] [[Code](https://github.com/MediaBrain-SJTU/CoCa3D)] ![](https://img.shields.io/github/stars/MediaBrain-SJTU/CoCa3D.svg?style=social&label=Star&maxAge=2592000)
37. `[CVPR'23]` BEVHeight: A Robust Framework for Vision-Based Roadside 3D Object Detection [[PDF](https://arxiv.org/abs/2303.08498)] [[Code](https://github.com/ADLab-AutoDrive/BEVHeight)] ![](https://img.shields.io/github/stars/ADLab-AutoDrive/BEVHeight.svg?style=social&label=Star&maxAge=2592000)
38. `[CVPR'23]` Collaboration Helps Camera Overtake LiDAR in 3D Detection [[PDF](https://arxiv.org/abs/2303.13560)] [[Code](https://github.com/MediaBrain-SJTU/CoCa3D)] ![](https://img.shields.io/github/stars/MediaBrain-SJTU/CoCa3D.svg?style=social&label=Star&maxAge=2592000)
39. `[CVPR'23]` V2X-Seq: The Large-Scale Sequential Dataset for the Vehicle-Infrastructure Cooperative Perception and Forecasting [[PDF](https://arxiv.org/abs/2305.05938)] [[Code](https://github.com/AIR-THU/DAIR-V2X-Seq)] ![](https://img.shields.io/github/stars/AIR-THU/CoCa3D.svg?style=social&label=Star&maxAge=2592000)
40. `[NeurIPS'23]` Robust Asynchronous Collaborative 3D Detection via Bird's Eye View Flow [[PDF](https://arxiv.org/pdf/2309.16940)] [[Code](https://github.com/MediaBrain-SJTU/CoBEVFlow)] ![](https://img.shields.io/github/stars/MediaBrain-SJTU/CoBEVFlow.svg?style=social&label=Star&maxAge=2592000)
41. `[NeurIPS'23]` Flow-Based Feature Fusion for Vehicle-Infrastructure Cooperative 3D Object Detection [[PDF](https://arxiv.org/pdf/2311.01682)] [[Code](https://github.com/haibao-yu/FFNet-VIC3D)] ![](https://img.shields.io/github/stars/haibao-yu/FFNet-VIC3D.svg?style=social&label=Star&maxAge=2592000)
42. `[NeurIPS'23]` How2comm: Communication-Efficient and Collaboration-Pragmatic Multi-Agent Perception [[PDF](https://dl.acm.org/doi/abs/10.5555/3666122.3667215)] [[Code](https://github.com/ydk122024/How2comm)] ![](https://img.shields.io/github/stars/ydk122024/How2comm.svg?style=social&label=Star&maxAge=2592000)
43. `[TIV'23]` HYDRO-3D: Hybrid object detection and tracking for cooperative perception using 3D LiDAR [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10148929)]
44. `[ICLR'23]` CO3: Cooperative Unsupervised 3D Representation Learning for Autonomous Driving [[PDF](https://arxiv.org/pdf/2206.04028)] [[Code](https://github.com/Runjian-Chen/CO3)] ![](https://img.shields.io/github/stars/Runjian-Chen/CO3.svg?style=social&label=Star&maxAge=2592000)
45. `[CoRL'23]` BM2CP: Efficient Collaborative Perception with LiDAR-Camera Modalities [[PDF](https://arxiv.org/pdf/2310.14702)] [[Code](https://github.com/byzhaoAI/BM2CP)] ![](https://img.shields.io/github/stars/byzhaoAI/BM2CP.svg?style=social&label=Star&maxAge=2592000)
46. `[ACMMM'23]` DUSA: Decoupled Unsupervised Sim2Real Adaptation for Vehicle-to-Everything Collaborative Perception [[PDF](https://arxiv.org/abs/2310.08117)] [[Code](https://github.com/refkxh/DUSA)] ![](https://img.shields.io/github/stars/refkxh/DUSA.svg?style=social&label=Star&maxAge=2592000)
47. `[ACMMM'23]` FeaCo: Reaching Robust Feature-Level Consensus in Noisy Pose Conditions [[PDF](https://dl.acm.org/doi/abs/10.1145/3581783.3611880)] [[Code](https://github.com/jmgu0212/FeaCo)] ![](https://img.shields.io/github/stars/jmgu0212/FeaCo.svg?style=social&label=Star&maxAge=2592000)
48. `[ACMMM'23]` What2comm: Towards Communication-Efficient Collaborative Perception via Feature Decoupling  [[PDF](https://dl.acm.org/doi/abs/10.1145/3581783.3611699)] [[Code](https://github.com/MediaBrain-SJTU/where2comm)] ![](https://img.shields.io/github/stars/MediaBrain-SJTU/where2comm.svg?style=social&label=Star&maxAge=2592000)
49. `[WACV'23]` Adaptive Feature Fusion for Cooperative Perception Using LiDAR Point Clouds [[PDF](https://arxiv.org/abs/2208.00116)] [[Code](https://github.com/DonghaoQiao/Adaptive-Feature-Fusion-for-Cooperative-Perception)] ![](https://img.shields.io/github/stars/DonghaoQiao/Adaptive-Feature-Fusion-for-Cooperative-Perception.svg?style=social&label=Star&maxAge=2592000)
50. `[ICRA'23]` Bridging the Domain Gap for Multi-Agent Perception [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10160871)] [[Code](https://github.com/DerrickXuNu/MPDA)] ![](https://img.shields.io/github/stars/DerrickXuNu/MPDA.svg?style=social&label=Star&maxAge=2592000)
51. `[ICRA'23]` Robust Collaborative 3D Object Detection in Presence of Pose Errors [[PDF](https://arxiv.org/abs/2211.07214)] [[Code](https://github.com/yifanlu0227/CoAlign)] ![](https://img.shields.io/github/stars/yifanlu0227/CoAlign.svg?style=social&label=Star&maxAge=2592000)
52. `[ICRA'23]` Deep Masked Graph Matching for Correspondence Identification in Collaborative Perception [[PDF](https://arxiv.org/abs/2303.07555)] [[Code](https://github.com/gaopeng5/DMGM)] ![](https://img.shields.io/github/stars/gaopeng5/DMGM.svg?style=social&label=Star&maxAge=2592000)
53. `[ICRA'23]` Uncertainty Quantification of Collaborative Detection for Self-Driving [[PDF](https://arxiv.org/abs/2209.08162)] [[Code](https://github.com/coperception/double-m-quantification)] ![](https://img.shields.io/github/stars/coperception/double-m-quantification.svg?style=social&label=Star&maxAge=2592000)
54. `[ICRA'23]` Model-Agnostic Multi-Agent Perception Framework [[PDF](https://arxiv.org/abs/2203.13168)] [[Code](https://github.com/DerrickXuNu/model_anostic)] ![](https://img.shields.io/github/stars/coperception/double-m-quantification.svg?style=social&label=Star&maxAge=2592000)
55. `[CoRL'22]` CoBEVT: Cooperative bird's eye view semantic segmentation with sparse transformers [[PDF](https://arxiv.org/pdf/2207.02202)] [[Code](https://github.com/DerrickXuNu/CoBEVT)] ![](https://img.shields.io/github/stars/DerrickXuNu/CoBEVT.svg?style=social&label=Star&maxAge=2592000)
56. `[CVPR'22]` COOPERNAUT: End-to-End Driving with Cooperative Perception for Networked Vehicles [[PDF](https://arxiv.org/abs/2205.02222)] [[Code](https://github.com/UT-Austin-RPL/Coopernaut)] ![](https://img.shields.io/github/stars/UT-Austin-RPL/Coopernaut.svg?style=social&label=Star&maxAge=2592000)
57. `[CVPR'22]` Learning from All Vehicles [[PDF](https://arxiv.org/abs/2203.11934)] [[Code](https://github.com/dotchen/LAV)] ![](https://img.shields.io/github/stars/dotchen/LAV.svg?style=social&label=Star&maxAge=2592000)
58. `[ECCV'22]` Latency-Aware Collaborative Perception [[PDF](https://arxiv.org/abs/2207.08560)] [[Code](https://github.com/MediaBrain-SJTU/SyncNet)] ![](https://img.shields.io/github/stars/MediaBrain-SJTU/SyncNet.svg?style=social&label=Star&maxAge=2592000)
59. `[ECCV'22]` V2X-ViT: Vehicle-to-Everything Cooperative Perception with Vision Transformer [[PDF](https://arxiv.org/abs/2203.10638)] [[Code](https://github.com/DerrickXuNu/v2x-vit)] ![](https://img.shields.io/github/stars/DerrickXuNu/v2x-vit.svg?style=social&label=Star&maxAge=2592000)
60. `[CoRL'22]` Multi-Robot Scene Completion: Towards Task-Agnostic Collaborative Perception [[PDF](https://proceedings.mlr.press/v205/li23e/li23e.pdf)] [[Code](https://github.com/coperception/star)] ![](https://img.shields.io/github/stars/coperception/star.svg?style=social&label=Star&maxAge=2592000)
61. `[ACMMM'22]` Complementarity-Enhanced and Redundancy-Minimized Collaboration Network for Multi-agent Perception [[PDF](https://dl.acm.org/doi/abs/10.1145/3503161.3548197)]
62. `[ICRA'22]` Multi-Robot Collaborative Perception with Graph Neural Networks [[PDF](https://arxiv.org/abs/2201.01760)]




### Decision-Making
1. `[ICCV'25 Workshop]` Drive-R1: Bridging Reasoning and Planning in VLMs for Autonomous Driving with Reinforcement Learning [[PDF](https://drivex-workshop.github.io/iccv2025/)]
2. `[ICCV'25 Workshop]` Contextual-Personalized Adaptive Cruise Control via Fine-Tuned Large Language Models [[PDF](https://drivex-workshop.github.io/iccv2025/)]
3. `[ICCV'25 Workshop]` Multi-modal Large Language Model for Training-free Vision-based Driver State [[PDF](https://drivex-workshop.github.io/iccv2025/)]
4. `[ICCV'25 Workshop]` V2X-based Logical Scenario Understanding with Vision-Language Models [[PDF](https://drivex-workshop.github.io/iccv2025/)]
5. `[TMC'25]` AgentsCoMerge: Large Language Model Empowered Collaborative Decision Making for Ramp Merging [[PDF](https://arxiv.org/pdf/2408.03624)]
6. `[arXiv]` A Vehicle-Infrastructure Multi-layer Cooperative Decision-making Framework [[PDF](https://arxiv.org/pdf/2503.16552)]
7. `[arXiv'24]` CoMAL: Collaborative Multi-Agent Large Language Models for Mixed-Autonomy Traffic [[PDF](https://arxiv.org/pdf/2410.14368)][[Code]](https://github.com/Hyan-Yao/CoMAL)
8. `[arXiv'24]` AGENTSCODRIVER: Large Language Model Empowered Collaborative Driving with Lifelong Learning [[PDF](https://arxiv.org/pdf/2404.06345)]
9. `[arXiv]` Research on Autonomous Driving Decision-making Strategies based Deep Reinforcement Learning [[PDF](https://arxiv.org/pdf/2408.03084)]
10. `[ECCV'24]` MAPPO-PIS: A Multi-Agent Proximal Policy Optimization Method with Prior Intent Sharing for CAVs' Cooperative Decision-Making [[PDF]](https://arxiv.org/abs/2408.06656) [[Code]](https://github.com/CCCC1dhcgd/A-MAPPO-PIS)
11. `[TITS'24]` Cooperative decision-making for cavs at unsignalized intersections: A marl approach with attention and hierarchical game priors [[PDF]](https://ieeexplore.ieee.org/abstract/document/10774177/)
12. `[TITS'24]` A Multi-Agent Reinforcement Learning Approach for Safe and Efficient Behavior Planning of Connected Autonomous Vehicles [[PDF]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10367764)
13. `[TVT'24]` Towards Interactive and Learnable Cooperative Driving Automation: a Large Language Model-Driven Decision-Making Framework [[PDF]](https://ieeexplore.ieee.org/abstract/document/10933798)[[Code]](https://github.com/FanGShiYuu/CoDrivingLLM)
14. `[TIV'24]` KoMA: Knowledge-driven Multi-agent Framework for Autonomous Driving with Large Language Models [[PDF]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10745878) [[Code]](https://github.com/jkmhhh/KoMA_Code)
15. `[ESWA'25]` CCMA: A Framework for Cascading Cooperative Multi-agent in Autonomous Driving Merging using Large Language Models [[PDF](https://www.sciencedirect.com/science/article/pii/S0957417425013399)]
16. `[ICCV'25]` CoLMDriver: LLM-based Negotiation Benefits Cooperative Autonomous Driving [[PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Liu_CoLMDriver_LLM-based_Negotiation_Benefits_Cooperative_Autonomous_Driving_ICCV_2025_paper.pdf)]
17. `[IEEE'25]` LMMCoDrive: Cooperative Driving with Large Multimodal Models [[PDF](https://ieeexplore.ieee.org/abstract/document/11247243/)]
18. `[IEEE'25]` DriVLMe: Enhancing LLM-based Autonomous Driving Agents with Embodied and Social Experiences [[PDF](https://ieeexplore.ieee.org/abstract/document/10802555/)]
19. `[arXiv'23]` LanguageMPC: Large Language Models as Decision Makers for Autonomous Driving [[PDF](https://arxiv.org/pdf/2310.03026)]
20. `[arXiv'25]` Context-aware Decision Making in Autonomous Vehicles [[PDF](https://www.sciencedirect.com/science/article/pii/S2590005625000475)]
21. `[arXiv'25]` Multi-Agent Deep Reinforcement Learning for Safe Autonomous Driving [[PDF](https://arxiv.org/abs/2503.19418)]
22. `[IEEE'25]` Mixed Motivation Driven Social Multi-Agent Reinforcement Learning [[PDF](http://ieeexplore.ieee.org/document/11036678/)]
23. `[Frontiers'25]` Multi-agent Reinforcement Learning Framework for Traffic Flow Management [[PDF](https://www.frontiersin.org/journals/mechanical-engineering/articles/10.3389/fmech.2025.1650918/full)]
24. `[arXiv'25]` Right-of-Way Based Multi-Agent Deep Reinforcement Learning [[PDF](https://www.sciencedirect.com/science/article/abs/pii/S095741742503667X)]
25. `[arXiv'25]` Cooperative Control of Self-Learning Traffic Signal and Connected Automated Vehicles [[PDF](https://www.sciencedirect.com/science/article/abs/pii/S0001457524004354)]
26. `[World Electric Vehicle Journal'24]` A Review of Decision-Making and Planning for Autonomous Vehicles in Intersection Environments [[PDF](https://www.mdpi.com/2032-6653/15/3/99)]
27. `[TVT'24]` Decision-Making for Autonomous Vehicles in Random Task Scenarios at Unsignalized Intersection Using Deep Reinforcement Learning [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10417752)]
28. `[DDCLS'24]` A Brief Survey of Deep Reinforcement Learning for Intersection Navigation of Autonomous Vehicles [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10606719)]
29. `[ICDE'24]` Parameterized Decision-making with Multi-modality Perception for Autonomous Driving [[PDF](https://arxiv.org/pdf/2312.11935)]
30. `[ICDE'24]`  Parameterized Decision-Making with Multi-Modality Perception for Autonomous Driving [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10597785)]
31. `[RAL'24]` Language-driven policy distillation for cooperative driving in multi-agent reinforcement learning [[PDF]](https://ieeexplore.ieee.org/document/10924758)
32. `[IoTML]` Research on Autonomous Driving Decision-making Strategies based Deep Reinforcement Learning [[PDF](https://dl.acm.org/doi/pdf/10.1145/3697467.3697643)]
33. `[ITSC'23]` Curriculum Proximal Policy Optimization with Stage-Decaying Clipping for Self-Driving at Unsignalized Intersections [[PDF](https://arxiv.org/pdf/2308.16445)]
34. `[IV'23]` Hybrid Decision Making for Autonomous Driving in Complex Urban Scenarios [[PDF](https://ieeexplore-ieee-org.sheffield.idm.oclc.org/stamp/stamp.jsp?tp=&arnumber=10186666)]
35. `[TIV'23]` Robust Lane Change Decision Making for Autonomous Vehicles: An Observation Adversarial Reinforcement Learning Approach [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9750867)]
36. `[TITS'23]` Robust Decision Making for Autonomous Vehicles at Highway On-Ramps: A Constrained Adversarial Reinforcement Learning Approach [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9994638)]
37. `[TVT'23]` Exploiting Multi-Modal Fusion for Urban Autonomous Driving Using Latent Deep Reinforcement Learning [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9934803)]
38. `[TIV'23]`  A Multi-Vehicle Game-Theoretic Framework for Decision Making and Planning of Autonomous Vehicles in Mixed Traffic [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10268996)]
39. `[TVT'23]`  Towards Robust Decision-Making for Autonomous Driving on Highway [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10107652)][[Code](https://github.com/Kayne0401/Robust-Decision-Making-Framework)]
40. `[IEEE Transactions on Transportation Electrification'23]` Interaction-Aware Decision-Making for Autonomous Vehicles [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10029923)]
37. `[ICRA'23]` Failure Detection for Motion Prediction of Autonomous Driving: An Uncertainty Perspective [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10160596)]
38. `[TITS'23]` Deep Multi-agent Reinforcement Learning for Highway On-Ramp Merging in Mixed Traffic [[PDF]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10159552)[[Code]](https://github.com/DongChen06/MARL_CAVs)
39. `[arXiv'23]` Spatial-Temporal-Aware Safe Multi-Agent Reinforcement Learning of Connected Autonomous Vehicles in Challenging Scenarios [[PDF]](https://arxiv.org/pdf/2210.02300)
40. `[arXiv'23]` Multi-Agent Reinforcement Learning Guided by Signal Temporal Logic Specifications [[PDF]](https://arxiv.org/pdf/2306.06808)
41. `[TITS'23]` Coordinating CAV Swarms at Intersections With a Deep Learning Model [[PDF]](https://ieeexplore.ieee.org/document/10078727)
42. `[National Conference on Sensors'23]` A Comprehensive Survey on Multi-Agent Reinforcement Learning for Connected and Automated Vehicles [[PDF](https://www.mdpi.com/1424-8220/23/10/4710)]
43. `[arXiv]`  Bringing Diversity to Autonomous Vehicles: An Interpretable Multi-vehicle Decision-making and Planning Framework [[PDF](https://arxiv.org/pdf/2302.06803)]
44. `[ISSN'22]` Reinforcement Learning-Based Autonomous Driving at Intersections in CARLA Simulator [[PDF](https://www.mdpi.com/1424-8220/22/21/8373)]
45. `[Autonomous Intelligent Systems'22]`  Multi-agent Reinforcement Learning for Cooperative Lane Changing of Connected and Autonomous Vehicles in Mixed Traffic [[PDF](https://link.springer.com/article/10.1007/s43684-022-00023-5)]
46. `[TVT'22]`  Highway Decision-Making and Motion Planning for Autonomous Driving via Soft Actor-Critic [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9718218)]
47. `[TITS'22]` PNNUAD: Perception Neural Networks Uncertainty Aware Decision-Making for Autonomous Vehicle [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9858685)]
48. `[CoRL'22]` Socially-Attentive Policy Optimization in Multi-Agent Self-Driving System [[PDF]](https://proceedings.mlr.press/v205/dai23a/dai23a.pdf)
49. `[TITS'22]` Social Coordination and Altruism in Autonomous Driving [[PDF]](https://ieeexplore.ieee.org/document/9905741)
50. `[TITS'22]` Multi-Agent DRL-Based Lane Change With Right-of-Way Collaboration Awareness [[PDF]](https://ieeexplore.ieee.org/document/9932003)
51. `[arXiv'22]` Graph Reinforcement Learning Application to Co-operative Decision-Making in Mixed Autonomy Traffic: Framework, Survey, and Challenges [[PDF]](https://arxiv.org/pdf/2211.03005) [[Code]](https://github.com/Jacklinkk/Graph_CAVs)
52. `[Autonomous Intelligent Systems'22]` Multi-agent reinforcement learning for autonomous vehicles: a survey [[PDF](https://link.springer.com/article/10.1007/s43684-022-00045-z)]
53. `[IROS'21]` Cooperative Autonomous Vehicles that Sympathize with Human Drivers [[PDF]](https://dl.acm.org/doi/10.1109/IROS51168.2021.9636151) [[Code]](https://github.com/BehradToghi/SymCoDrive_IROS2021)
54. `[CoRL'20]` SMARTS: Scalable Multi-Agent Reinforcement Learning Training School for Autonomous Driving [[PDF]](https://proceedings.mlr.press/v155/zhou21a/zhou21a.pdf)[[Code]](https://github.com/huawei-noah/SMARTS)


### Planning
1. `[ICCV'25 Workshop]` V2XPnP: Vehicle-to-Everything Spatio-Temporal Fusion for Multi-Agent Perception and Prediction [[PDF](https://drivex-workshop.github.io/iccv2025/)]
2. `[ICCV'25 Workshop]` MAP: End-to-End Autonomous Driving with Map-Assisted Planning [[PDF](https://drivex-workshop.github.io/iccv2025/)]
3. `[ICCV'25 Workshop]` The Role of Radar in End-to-End Autonomous Driving [[PDF](https://drivex-workshop.github.io/iccv2025/)]
4. `[ICCV'25 Workshop]` Robust Scenario Mining Assisted by Multimodal Semantics [[PDF](https://drivex-workshop.github.io/iccv2025/)]
5. `[ICCV'25 Workshop]` Improving Event-Phase Captions in Multi-View Urban Traffic Videos via Prompt-Aware LoRA Tuning of Vision Language Models [[PDF](https://drivex-workshop.github.io/iccv2025/)]
6. `[arXiv]` CoDriveVLM: VLM-Enhanced Urban Cooperative Dispatching and Motion Planning for Future Autonomous Mobility on Demand Systems [[PDF](https://arxiv.org/pdf/2501.06132)][[Code](https://github.com/henryhcliu/CoDriveVLM)]
7. `[arXiv]` Improved Consensus ADMM for Cooperative Motion Planning of Large-Scale Connected Autonomous Vehicles with Limited Communication [[PDF](https://arxiv.org/pdf/2401.09032)][[Code](https://henryhcliu.github.io/icadmm_cmp_carla/)]
8. `[arXiv]` THOMAS: TRAJECTORY HEATMAP OUTPUT WITH LEARNED MULTI-AGENT SAMPLING [[PDF](https://arxiv.org/pdf/2110.06607)]
9. `[arXiv]` Real-Time Motion Prediction via Heterogeneous Polyline Transformer with Relative Pose Encoding [[PDF]([https://arxiv.org/pdf/2310.12970))][[Code](https://github.com/zhejz/HPTR)]
10. `[TPAMI'24]` MTR++: Multi-Agent Motion Prediction with Symmetric Scene Modeling and Guided Intention Querying  [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10398503)]
11. `[AAAI'24]` EDA: Evolving and Distinct Anchors for Multimodal Motion Prediction [[PDF](https://arxiv.org/pdf/2312.09501)] [[Code](https://github.com/Longzhong-Lin/EDA)]
12. `[arXiv'25]` LeAD: The LLM Enhanced Planning System Converged with End-to-End Autonomous Driving [[PDF](https://arxiv.org/abs/2507.05754)]
13. `[IEEE'25]` LLMDriver: Autonomous Driving Planning Based on Large Language Models [[PDF](https://ieeexplore.ieee.org/document/11174944/)]
14. `[arXiv'25]` An LLM-Powered Cooperative Framework for Large-Scale Multi-Vehicle Navigation [[PDF](https://arxiv.org/abs/2510.07825)]
15. `[ICRA'24]` Parallel Optimization with Hard Safety Constraints for Cooperative Planning of Connected Autonomous Vehicles [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10611158)]
16. `[RAL'24]` SIMPL: A Simple and Efficient Multi-Agent Motion Prediction Baseline for Autonomous Driving [[PDF]](https://ieeexplore.ieee.org/abstract/document/10449378)
17. `[RAL'24]` CMP: Cooperative Motion Prediction With Multi-Agent Communication [[PDF]](https://ieeexplore.ieee.org/abstract/document/10908648)
18. `[NeurIPS'24]` SMART: Scalable Multi-agent Real-time Motion Generation via Next-token Prediction [[PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/cef5c8dec67597b854f0162ad76d92d2-Paper-Conference.pdf) [[Code]](https://github.com/rainmaker22/SMART)
19. `[arXiv'25]` UNCAP: Uncertainty-Guided Neurosymbolic Planning [[PDF](https://arxiv.org/abs/2510.12992)]
20. `[Electronics'25]` Eco-Cooperative Planning and Control of Connected Autonomous Vehicles Considering Energy Consumption Characteristics [[PDF](https://www.mdpi.com/2079-9292/14/8/1646)]
21. `[IEEE'25]` Multiagent Trajectory Prediction With Difficulty-Guided Feature Enhancement [[PDF](https://ieeexplore.ieee.org/iel8/7083369/10849592/10854576.pdf)]
22. `[ICCV'25]` Unified Multi-Agent Trajectory Modeling with Masked Trajectory Diffusion [[PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Yang_Unified_Multi-Agent_Trajectory_Modeling_with_Masked_Trajectory_Diffusion_ICCV_2025_paper.pdf)]
23. `[IEEE Internet of Things Journal'24]` Coordination for Connected and Autonomous Vehicles at Unsignalized Intersections: An Iterative Learning-Based Collision-Free Motion Planning Method [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10224258)]
19. `[ICCV'23]` BiFF: Bi-level Future Fusion with Polyline-based Coordinate for Interactive Trajectory Prediction [[PDF](https://arxiv.org/pdf/2306.14161)]
20. `[CVPR'23]` ProphNet: Efficient Agent-Centric Motion Forecasting with Anchor-Informed Proposals [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10204544)]
21. `[CVPR'23]` FJMP: Factorized Joint Multi-Agent Motion Prediction over Learned Directed Acyclic Interaction Graphs [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10204178)][[Code](https://rluke22.github.io/FJMP/)]
22. `[CVPR'23]` MotionDiffuser: Controllable Multi-Agent Motion Prediction Using Diffusion [[PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Jiang_MotionDiffuser_Controllable_Multi-Agent_Motion_Prediction_Using_Diffusion_CVPR_2023_paper.pdf)
23. `[ICRA'23]` Wayformer: Motion Forecasting via Simple & Efficient Attention Networks [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10160609)]
24. `[ICRA'23]` GoRela: Go Relative for Viewpoint-Invariant Motion Forecasting [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10160984)]
25. `[ICRA'23]` GANet: Goal Area Network for Motion Forecasting [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10160468)][[Code](https://github.com/kingwmk/GANet)]
26. `[TITS'23]` Decentralized iLQR for Cooperative Trajectory Planning of Connected Autonomous Vehicles via Dual Consensus ADMM [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10171831)]
27. `[TIV'23]` Fault-tolerant cooperative driving at signal-free intersections [[PDF]](https://ieeexplore.ieee.org/document/9735412)
28. `[TIV'23]` OpenCDA-ROS: Enabling Seamless Integration of Simulation and Real-World Cooperative Driving Automation [[PDF](https://ieeexplore.ieee.org/document/10192346)
29. `[TIV'23]` Optimal Trajectory Planning for Connected and Automated Vehicles in Lane-Free Traffic With Vehicle Nudging [[PDF]](https://ieeexplore.ieee.org/document/10032638)
30. `[TIV'23]` Multi-Vehicle Conflict Management With Status and Intent Sharing Under Time Delays [[PDF]](https://ieeexplore.ieee.org/document/9998111)
31. `[TIV'23]` Optimizing Vehicle Re-Ordering Events in Coordinated Autonomous Intersection Crossings Under CAVs' Location Uncertainty [[PDF]](https://ieeexplore.ieee.org/document/9976239)
32. `[TIV'23]` Optimizing Vehicle Re-Ordering Events in Coordinated Autonomous Intersection Crossings Under CAVs' Location Uncertainty [[PDF]](https://ieeexplore.ieee.org/document/9735412)
33. `[IEEE Robotics and Automation Letters'23]` MacFormer: Map-Agent Coupled Transformer for Real-Time and Robust Trajectory Prediction [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10238733)]
34. `[ICCV'23]` Forecast-MAE: Self-supervised Pre-training for Motion Forecasting with Masked Autoencoders [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10377996)]
35. `[CoRL'23]` iPLAN: Intent-Aware Planning in Heterogeneous Traffic via Distributed Multi-Agent Reinforcement Learning[[PDF]](https://openreview.net/pdf?id=EvuAJ0wD98)[[Code]](https://github.com/wuxiyang1996/iPLAN)
36. `[NeurIPS'22]` Motion Transformer with Global Intention Localization and Local Movement Refinement  [[PDF](https://proceedings.neurips.cc/paper_files/paper/2022/file/2ab47c960bfee4f86dfc362f26ad066a-Paper-Conference.pdf)]  [[Code](https://github.com/sshaoshuai/MTR)]
37. `[CVPR'22]` HiVT: Hierarchical Vector Transformer for Multi-Agent Motion Prediction [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9878832)][[Code](https://github.com/ZikangZhou/HiVT)]
38. `[ICRA'22]` MultiPath++: Efficient Information Fusion and Trajectory Aggregation for Behavior Prediction [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9812107)]
39. `[ICRA'22]` GOHOME: Graph-Oriented Heatmap Output for future Motion Estimation [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9812253)]
40. `[TPAMI'22]` HDGT: Heterogeneous Driving Graph Transformer for Multi-Agent Trajectory Prediction via Scene Encoding [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10192373)]
41. `[TITS'22]` Cooperative Formation of Autonomous Vehicles in Mixed Traffic Flow: Beyond Platooning [[PDF]](https://ieeexplore.ieee.org/document/9709187)
42. `[TVT'22]` Multi-Lane Unsignalized Intersection Cooperation With Flexible Lane Direction Based on Multi-Vehicle Formation Control [[PDF]](https://ieeexplore.ieee.org/document/9740423)
43. `[ICCV'21]` DenseTNT: End-to-end Trajectory Prediction from Dense Goal Sets [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9710037)]
44. `[ITSC'21]` OpenCDA: An Open Cooperative Driving Automation Framework Integrated with Co-Simulation [[PDF]](https://ieeexplore.ieee.org/abstract/document/9564825) [[Code]](https://github.com/ucla-mobility/OpenCDA)

   
### Communication
1. `[CVPR'24]` Communication-Efficient Collaborative Perception via Information Filling with Codebook [[PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Hu_Communication-Efficient_Collaborative_Perception_via_Information_Filling_with_Codebook_CVPR_2024_paper.pdf)] [[Code](https://github.com/PhyllisH/CodeFilling)] ![](https://img.shields.io/github/stars/PhyllisH/CodeFilling.svg?style=social&label=Star&maxAge=2592000)
2. `[CVPR'24]` ERMVP: Communication-Efficient and Collaboration-Robust Multi-Vehicle Perception in Challenging Environments [[PDF](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_ERMVP_Communication-Efficient_and_Collaboration-Robust_Multi-Vehicle_Perception_in_Challenging_Environments_CVPR_2024_paper.html)] [[Code](https://github.com/Terry9a/ERMVP)] ![](https://img.shields.io/github/stars/Terry9a/ERMVP.svg?style=social&label=Star&maxAge=2592000)
3. `[CVPR'24]` Multi-Agent Collaborative Perception via Motion-Aware Robust Communication Network [[PDF](https://openaccess.thecvf.com/content/CVPR2024/html/Hong_Multi-agent_Collaborative_Perception_via_Motion-aware_Robust_Communication_Network_CVPR_2024_paper.html)] [[Code](https://github.com/IndigoChildren/collaborative-perception-MRCNet)] ![](https://img.shields.io/github/stars/IndigoChildren/collaborative-perception-MRCNet.svg?style=social&label=Star&maxAge=2592000)
4. `[ICRA'23]` Communication-Critical Planning via Multi-Agent Trajectory Exchange [[PDF](https://arxiv.org/abs/2303.06080)]
5. `[ICRA'23]` We Need to Talk: Identifying and Overcoming Communication-Critical Scenarios for Self-Driving [[PDF](https://arxiv.org/abs/2305.04352)] 
6. `[IJCAI'22]` Robust Collaborative Perception against Communication Interruption [[PDF](https://learn-to-race.org/workshop-ai4ad-ijcai2022/papers.html)]
7. `[arXiv'25]` InfoCom: Kilobyte-Scale Communication-Efficient Collaborative Perception with Information Bottleneck [[PDF](https://arxiv.org/abs/2512.10305)]
8. `[arXiv'25]` Map4comm: A Map-Aware Collaborative Perception Framework with Efficient-Bandwidth Information Fusion [[PDF](https://www.sciencedirect.com/science/article/abs/pii/S1566253525006396)]
9. `[ISPRS'25]` MapCooper: A Communication-Efficient Collaborative Perception Framework via Map Alignment [[PDF](https://isprs-annals.copernicus.org/articles/X-G-2025/673/2025/isprs-annals-X-G-2025-673-2025.html)] 


### End-to-End
1. `[ICCV'25 Workshop]` Research Challenges and Progress in the End-to-End V2X Cooperative Autonomous Driving Competition [[PDF](https://drivex-workshop.github.io/iccv2025/)]
2. `[IV'24]` ICOP: Image-based Cooperative Perception for End-to-End Autonomous Driving [[paper](https://ieeexplore.ieee.org/abstract/document/10588825)]
2. `[TIV'23]` End-to-end Autonomous Driving with Semantic Depth Cloud Mapping and Multi-agent [[paper](https://doi.org/10.1109/TIV.2022.3185303)] [[code](https://github.com/oskarnatan/end-to-end-driving)]
3. `[AAAI'22]` CADRE: A Cascade Deep Reinforcement Learning Framework for Vision-Based Autonomous Urban Driving [[paper](https://arxiv.org/abs/2202.08557)] [[code](https://github.com/BIT-MCS/Cadre.git)] ![](https://img.shields.io/github/stars/BIT-MCS/Cadre.svg?style=social&label=Star&maxAge=2592000)
4. `[CVPR'22]` COOPERNAUT: End-to-End Driving with Cooperative Perception for Networked Vehicles [[paper](https://arxiv.org/abs/2205.02222)] [[code](https://github.com/UT-Austin-RPL/Coopernaut.git)] ![](https://img.shields.io/github/stars/UT-Austin-RPL/Coopernaut.svg?style=social&label=Star&maxAge=2592000)
5. `[NeurIPS'21]` Learning to Simulate Self-Driven Particles System with Coordinated Policy Optimization [[paper](https://arxiv.org/pdf/2110.13827)] [[code](https://github.com/decisionforce/CoPO.git)] [[Webpage](https://decisionforce.github.io/CoPO/)]
6. `[arXiv]` End-to-End Autonomous Driving through V2X Cooperation [[paper](https://arxiv.org/abs/2404.00717)] [[code](https://github.com/AIR-THU/UniV2X)] ![](https://img.shields.io/github/stars/AIR-THU/UniV2X.svg?style=social&label=Star&maxAge=2592000)
7. `[arXiv]` Towards Collaborative Autonomous Driving: Simulation Platform and End-to-End System [[paper](https://arxiv.org/abs/2404.09496)] [[code](https://github.com/CollaborativePerception/V2Xverse)] ![](https://img.shields.io/github/stars/CollaborativePerception/V2Xverse.svg?style=social&label=Star&maxAge=2592000)
8. `[arXiv]` AgentsCoMerge: Large Language Model Empowered Collaborative Decision Making for Ramp Merging [[paper](https://arxiv.org/abs/2408.03624)]
9. `[arXiv]` AgentsCoDriver: Large Language Model Empowered Collaborative Driving with Lifelong Learning [[paper](https://arxiv.org/abs/2404.06345)]
10. `[NeurIPS'25]` AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning [[Webpage](https://autovla.github.io/)]





### Dataset and Simulator

#### Dataset
1. `[ICCV'25 Workshop]` HetroD: A High-Fidelity Drone Dataset and Benchmark for Heterogeneous Traffic in Autonomous Driving [[PDF](https://drivex-workshop.github.io/iccv2025/)]
2. `[CoRL'17]` CARLA: An Open Urban Driving Simulator [[PDF](https://arxiv.org/abs/1711.03938)] [[Code](https://github.com/carla-simulator/carla)] [[Webpage](https://carla.org)] ![CARLA](https://img.shields.io/badge/-CARLA-blue)
2. `[ICCV'21]` V2X-Sim: Multi-Agent Collaborative Perception Dataset and Benchmark for Autonomous Driving [[PDF](https://arxiv.org/abs/2202.08449)] [[Code](https://github.com/ai4ce/V2X-Sim)] [[Webpage](https://ai4ce.github.io/V2X-Sim)] ![V2X-Sim](https://img.shields.io/badge/-V2X--Sim-blue)
3. `[ACCV'22]` DOLPHINS: Dataset for Collaborative Perception Enabled Harmonious and Interconnected Self-Driving [[PDF](https://arxiv.org/abs/2207.07609)] [[Code](https://github.com/explosion5/Dolphins)] [[Webpage](https://dolphins-dataset.net)] ![DOLPHINS](https://img.shields.io/badge/-DOLPHINS-blue)
4. `[ICRA'22]` OPV2V: An Open Benchmark Dataset and Fusion Pipeline for Perception with Vehicle-to-Vehicle Communication [[PDF](https://arxiv.org/abs/2109.07644)] [[Code](https://github.com/DerrickXuNu/OpenCOOD)] [[Webpage](https://mobility-lab.seas.ucla.edu/opv2v)] ![OPV2V](https://img.shields.io/badge/-OPV2V-blue)
5. `[ECCV'22]` V2X-ViT: Vehicle-to-Everything Cooperative Perception with Vision Transformer [[PDF](https://arxiv.org/abs/2203.10638)] [[Code](https://github.com/DerrickXuNu/v2x-vit)] [[Webpage](https://drive.google.com/drive/folders/1r5sPiBEvo8Xby-nMaWUTnJIPK6WhY1B6)] ![V2X-ViT](https://img.shields.io/badge/-V2X--ViT-blue)
6. `[CVPR'22]` COOPERNAUT: End-to-End Driving with Cooperative Perception for Networked Vehicles [[PDF](https://ut-austin-rpl.github.io/Coopernaut/)] [[Code](https://github.com/UT-Austin-RPL/Coopernaut)] [[Webpage](https://ut-austin-rpl.github.io/Coopernaut/)] ![AutoCastSim](https://img.shields.io/badge/-AutoCastSim-blue)
7. `[CVPR'22]` DAIR-V2X: A Large-Scale Dataset for Vehicle-Infrastructure Cooperative 3D Object Detection [[PDF](https://arxiv.org/abs/2204.05575)] [[Code](https://github.com/AIR-THU/DAIR-V2X?tab=readme-ov-file)] [[Webpage](https://thudair.baai.ac.cn/index)] ![DAIR-V2X](https://img.shields.io/badge/-DAIR--V2X-blue)
8. `[NeurIPS'22]` Where2comm: Communication-Efficient Collaborative Perception via Spatial Confidence Maps [[PDF&review](https://openreview.net/forum?id=dLL4KXzKUpS)] [[Code](https://github.com/MediaBrain-SJTU/where2comm)] [[Webpage](https://siheng-chen.github.io/dataset/coperception-uav)] ![CoPerception-UAV](https://img.shields.io/badge/-CoPerception--UAV-blue)
9. `[NeurIPS'23]` Robust Asynchronous Collaborative 3D Detection via Bird's Eye View Flow [[PDF&review](https://openreview.net/forum?id=UHIDdtxmVS)] ![IRV2V](https://img.shields.io/badge/-IRV2V-blue)
10. `[CVPR'23]` Collaboration Helps Camera Overtake LiDAR in 3D Detection [[PDF](https://arxiv.org/abs/2303.13560)] [[Code](https://github.com/MediaBrain-SJTU/CoCa3D)] [[Webpage](https://siheng-chen.github.io/dataset/CoPerception+)] ![CoPerception-UAV+](https://img.shields.io/badge/-CoPerception--UAV%2B-blue) ![OPV2V+](https://img.shields.io/badge/-OPV2V%2B-blue)
11. `[CVPR'23]` V2V4Real: A Large-Scale Real-World Dataset for Vehicle-to-Vehicle Cooperative Perception [[PDF](https://arxiv.org/abs/2303.07601)] [[Code](https://github.com/ucla-mobility/V2V4Real)] [[Webpage](https://mobility-lab.seas.ucla.edu/v2v4real)] ![V2V4Real](https://img.shields.io/badge/-V2V4Real-blue)
12. `[CVPR'23]` V2X-Seq: The Large-Scale Sequential Dataset for the Vehicle-Infrastructure Cooperative Perception and Forecasting [[PDF](https://arxiv.org/abs/2305.05938)] [[Code](https://github.com/AIR-THU/DAIR-V2X-Seq)] [[Webpage](https://thudair.baai.ac.cn/index)] ![DAIR-V2X-Seq](https://img.shields.io/badge/-DAIR--V2X--Seq-blue)
13. `[ICRA'23]` Robust Collaborative 3D Object Detection in Presence of Pose Errors [[PDF](https://arxiv.org/abs/2211.07214)] [[Code](https://github.com/yifanlu0227/CoAlign)] [[Webpage](https://siheng-chen.github.io/dataset/dair-v2x-c-complemented)] ![DAIR-V2X-C Complemented](https://img.shields.io/badge/-DAIR--V2X--C-blue)
14. `[ICCV'23]` Optimizing the Placement of Roadside LiDARs for Autonomous Driving [[PDF](https://arxiv.org/abs/2310.07247)] ![Roadside-Opt](https://img.shields.io/badge/-Roadside--Opt-blue)
15. `[AAAI'24]` DeepAccident: A Motion and Accident Prediction Benchmark for V2X Autonomous Driving [[PDF](https://arxiv.org/abs/2304.01168)] [[Code](https://github.com/tianqi-wang1996/DeepAccident)] [[Webpage](https://deepaccident.github.io)] ![DeepAccident](https://img.shields.io/badge/-DeepAccident-blue)
16. `[ICLR'24]` An Extensible Framework for Open Heterogeneous Collaborative Perception [[PDF&review](https://openreview.net/forum?id=KkrDUGIASk)] [[Code](https://github.com/yifanlu0227/HEAL)] [[Webpage](https://huggingface.co/datasets/yifanlu/OPV2V-H)] ![OPV2V-H](https://img.shields.io/badge/-OPV2V--H-blue)
17. `[CVPR'24]` HoloVIC: Large-Scale Dataset and Benchmark for Multi-Sensor Holographic Intersection and Vehicle-Infrastructure Cooperative [[PDF](https://arxiv.org/abs/2403.02640)] [[Webpage](https://holovic.net)] ![HoloVIC](https://img.shields.io/badge/-HoloVIC-blue)
18. `[CVPR'24]` Multiagent Multitraversal Multimodal Self-Driving: Open MARS Dataset [[PDF](https://arxiv.org/abs/2406.09383)] [[Code](https://github.com/ai4ce/MARS)] [[Webpage](https://ai4ce.github.io/MARS)] ![Open Mars Dataset](https://img.shields.io/badge/-Open%20Mars%20Dataset-blue)
19. `[CVPR'24]` RCooper: A Real-World Large-Scale Dataset for Roadside Cooperative Perception [[PDF](https://arxiv.org/abs/2403.10145)] [[Code](https://github.com/AIR-THU/DAIR-RCooper)] [[Webpage](https://www.t3caic.com/qingzhen)] ![RCooper](https://img.shields.io/badge/-RCooper-blue)
20. `[CVPR'24]` TUMTraf V2X Cooperative Perception Dataset [[PDF](https://arxiv.org/abs/2403.01316)] [[Code](https://github.com/tum-traffic-dataset/tum-traffic-dataset-dev-kit)] [[Webpage](https://tum-traffic-dataset.github.io/tumtraf-v2x)] ![TUMTraf-V2X](https://img.shields.io/badge/-TUMTraf--V2X-blue)
21. `[CVPR'24]` Editable Scene Simulation for Autonomous Driving via Collaborative LLM-Agents [[PDF](https://arxiv.org/abs/2402.05746)] [[Code](https://github.com/yifanlu0227/ChatSim)] [[Webpage](https://yifanlu0227.github.io/ChatSim/)] ![ChatSim](https://img.shields.io/badge/-ChatSim-blue)
22. `[ECCV'24]` H-V2X: A Large Scale Highway Dataset for BEV Perception [[PDF](https://eccv2024.ecva.net/virtual/2024/poster/126)] ![H-V2X](https://img.shields.io/badge/-H--V2X-blue)
23. `[NeurIPS'24]` Learning Cooperative Trajectory Representations for Motion Forecasting [[PDF](https://arxiv.org/abs/2311.00371)] [[Code](https://github.com/AIR-THU/V2X-Graph)] [[Webpage](https://thudair.baai.ac.cn/index)] ![DAIR-V2X-Traj](https://img.shields.io/badge/-DAIR--V2X--Traj-blue)
24. `[NeurIPS'24]` SMART: Scalable Multi-agent Real-time Motion Generation via Next-token Prediction [[PDF](https://arxiv.org/abs/2405.15677)] [[Code](https://github.com/rainmaker22/SMART)] [[Webpage](https://smart-motion.github.io/smart/)] ![SMART](https://img.shields.io/badge/-SMART-blue)
25. `[CVPR'25]` Mono3DVLT: Monocular-Video-Based 3D Visual Language Tracking [[PDF](https://cvpr.thecvf.com/Conferences/2025/AcceptedPapers)] ![Mono3DVLT-V2X](https://img.shields.io/badge/-Mono3DVLT--V2X-blue)
26. `[CVPR'25]` RCP-Bench: Benchmarking Robustness for Collaborative Perception Under Diverse Corruptions [[PDF](https://cvpr.thecvf.com/virtual/2025/poster/34639)] ![RCP-Bench](https://img.shields.io/badge/-RCP--Bench-blue)
27. `[CVPR'25]` V2X-R: Cooperative LiDAR-4D Radar Fusion for 3D Object Detection with Denoising Diffusion [[PDF](https://arxiv.org/abs/2411.08402)] [[Code](https://github.com/ylwhxht/V2X-R)] ![V2X-R](https://img.shields.io/badge/-V2X--R-blue)
28. `[arXiv]` Adver-City: Open-Source Multi-Modal Dataset for Collaborative Perception Under Adverse Weather Conditions [[PDF](https://arxiv.org/abs/2410.06380)] [[Code](https://github.com/QUARRG/Adver-City)] [[Webpage](https://labs.cs.queensu.ca/quarrg/datasets/adver-city)] ![Adver-City](https://img.shields.io/badge/-Adver--City-blue)
29. `[arXiv]` CP-Guard+: A New Paradigm for Malicious Agent Detection and Defense in Collaborative Perception [[PDF](https://arxiv.org/abs/2502.07807)] ![CP-GuardBench](https://img.shields.io/badge/-CP--GuardBench-blue)
30. `[arXiv]` DriveGen: Toward Infinite Diverse Traffic Scenarios with Large Models [[PDF](https://arxiv.org/pdf/2503.05808)] ![DriveGen](https://img.shields.io/badge/-DriveGen-blue)
31. `[arXiv]` Griffin: Aerial-Ground Cooperative Detection and Tracking Dataset and Benchmark [[PDF](https://arxiv.org/abs/2503.06983)] [[Code](https://github.com/wang-jh18-SVM/Griffin)] [[Webpage](https://pan.baidu.com/s/1NDgsuHB-QPRiROV73NRU5g)] ![Griffin](https://img.shields.io/badge/-Griffin-blue)
32. `[arXiv]` InScope: A New Real-world 3D Infrastructure-side Collaborative Perception Dataset for Open Traffic Scenarios [[PDF](https://arxiv.org/abs/2407.21581)] [[Code](https://github.com/xf-zh/InScope)] ![InScope](https://img.shields.io/badge/-InScope-blue)
33. `[arXiv]` Mixed Signals: A Diverse Point Cloud Dataset for Heterogeneous LiDAR V2X Collaboration [[PDF](https://arxiv.org/abs/2502.14156)] [[Code](https://github.com/chinitaberrio/Mixed-Signals)] [[Webpage](https://mixedsignalsdataset.cs.cornell.edu)] ![Mixed Signals](https://img.shields.io/badge/-Mixed%20Signals-blue)
34. `[arXiv]` Multi-V2X: A Large Scale Multi-modal Multi-penetration-rate Dataset for Cooperative Perception [[PDF](https://arxiv.org/abs/2409.04980)] [[Code](https://github.com/RadetzkyLi/Multi-V2X)] ![Multi-V2X](https://img.shields.io/badge/-Multi--V2X-blue)
35. `[arXiv]` RCDN: Towards Robust Camera-Insensitivity Collaborative Perception via Dynamic Feature-based 3D Neural Modeling [[PDF](https://arxiv.org/abs/2405.16868)] ![OPV2V-N](https://img.shields.io/badge/-OPV2V--N-blue)
36. `[arXiv]` V2V-LLM: Vehicle-to-Vehicle Cooperative Autonomous Driving with Multi-Modal Large Language Models [[PDF](https://arxiv.org/abs/2502.09980)] [[Code](https://github.com/eddyhkchiu/V2VLLM)] [[Webpage](https://eddyhkchiu.github.io/v2vllm.github.io)] ![V2V-QA](https://img.shields.io/badge/-V2V--QA-blue)
37. `[arXiv]` V2XPnP: Vehicle-to-Everything Spatio-Temporal Fusion for Multi-Agent Perception and Prediction [[PDF](https://arxiv.org/abs/2412.01812)] [[Code](https://github.com/Zewei-Zhou/V2XPnP)] [[Webpage](https://mobility-lab.seas.ucla.edu/v2xpnp)] ![V2XPnP-Seq](https://img.shields.io/badge/-V2XPnP--Seq-blue)
38. `[arXiv]` V2X-Radar: A Multi-Modal Dataset with 4D Radar for Cooperative Perception [[PDF](https://arxiv.org/abs/2411.10962)] [[Webpage](http://openmpd.com/column/V2X-Radar)] ![V2X-Radar](https://img.shields.io/badge/-V2X--Radar-blue)
39. `[arXiv]` V2X-Real: a Large-Scale Dataset for Vehicle-to-Everything Cooperative Perception [[PDF](https://arxiv.org/abs/2403.16034)] [[Webpage](https://mobility-lab.seas.ucla.edu/v2x-real)] ![V2X-Real](https://img.shields.io/badge/-V2X--Real-blue)
40. `[arXiv]` V2X-ReaLO: An Open Online Framework and Dataset for Cooperative Perception in Reality [[PDF](https://arxiv.org/abs/2503.10034)] ![V2X-ReaLO](https://img.shields.io/badge/-V2X--ReaLO-blue)
41. `[arXiv]` WHALES: A Multi-Agent Scheduling Dataset for Enhanced Cooperation in Autonomous Driving [[PDF](https://arxiv.org/abs/2411.13340)] [[Code](https://github.com/chensiweiTHU/WHALES)] [[Webpage](https://pan.baidu.com/s/1dintX-d1T-m2uACqDlAM9A)] ![WHALES](https://img.shields.io/badge/-WHALES-blue)
42. `[arXiv]` DriveGen: Towards Infinite Diverse Traffic Scenarios with Large Models [[PDF](https://arxiv.org/pdf/2503.05808)] ![DriveGen-CS](https://img.shields.io/badge/-DriveGen--CS-blue)
43. `[arXiv]` Towards Collaborative Autonomous Driving: Simulation Platform and End-to-End System [[PDF](https://arxiv.org/abs/2404.09496)] [[Code](https://github.com/CollaborativePerception/V2Xverse)] ![V2Xverse](https://img.shields.io/badge/-V2Xverse-blue)
44. `[NeurIPS'25]` UrbanIng-V2X: A Large-Scale Multi-Vehicle, Multi-Infrastructure Dataset Across Multiple Intersections [[PDF](https://arxiv.org/abs/2510.23478)] [[Code](https://github.com/thi-ad/UrbanIng-V2X)] ![UrbanIng-V2X](https://img.shields.io/badge/-UrbanIng--V2X-blue)

#### Simulator
1. `[CoRL'17]` CARLA: An Open Urban Driving Simulator [[PDF](https://arxiv.org/abs/1711.03938)] [[Code](https://github.com/carla-simulator/carla)] [[Webpage](https://carla.org)] ![CARLA](https://img.shields.io/badge/-CARLA-blue)
2. `[CVPR'24]` Editable Scene Simulation for Autonomous Driving via Collaborative LLM-Agents [[PDF](https://arxiv.org/abs/2402.05746)] [[Code](https://github.com/yifanlu0227/ChatSim)] [[Webpage](https://yifanlu0227.github.io/ChatSim/)] ![ChatSim](https://img.shields.io/badge/-ChatSim-blue)
3. `[NeurIPS'24]` NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking [[PDF](https://arxiv.org/abs/2406.15349)] [[Code](https://github.com/autonomousvision/navsim)] [[Webpage](https://huggingface.co/spaces/AGC2024-P/e2e-driving-navsim)] ![NAVSIM](https://img.shields.io/badge/-NAVSIM-blue)
4. `[arXiv]` DriveGen: Towards Infinite Diverse Traffic Scenarios with Large Models [[PDF](https://arxiv.org/pdf/2503.05808)] ![DriveGen](https://img.shields.io/badge/-DriveGen-blue)
5. `[arXiv]` Towards Collaborative Autonomous Driving: Simulation Platform and End-to-End System [[PDF](https://arxiv.org/abs/2404.09496)] [[Code](https://github.com/CollaborativePerception/V2Xverse)] ![V2Xverse](https://img.shields.io/badge/-V2Xverse-blue)


### Security and Robustness
1. `[TMC'25]` Collaborative Perception Against Data Fabrication Attacks in Vehicular Networks [[PDF](https://ieeexplore.ieee.org/abstract/document/11006384)]
2. `[AAAI'25 Oral]` CP-Guard: Malicious Agent Detection and Defense in Collaborative Bird's Eye View Perception [[PDF](https://arxiv.org/abs/2412.12000), [Code](https://github.com/CP-Security/CP-Guard)]
3. `[IROS'24]` Malicious Agent Detection for Robust Multi-Agent Collaborative Perception [[PDF](http://arxiv.org/abs/2310.11901), [Code](https://github.com/shengyin1224/MADE)] ![](https://img.shields.io/github/stars/shengyin1224/MADE.svg?style=social&label=Star&maxAge=2592000)
4. `[ICRA'24]` AdvGPS: Adversarial GPS for Multi-Agent Perception Attack [[PDF](https://ieeexplore.ieee.org/abstract/document/10610012)] [[Code](https://github.com/jinlong17/AdvGPS)] ![](https://img.shields.io/github/stars/jinlong17/AdvGPS.svg?style=social&label=Star&maxAge=2592000)
5. `[JATS'24]` RAMPART: Reinforcing Autonomous Multi-Agent Protection through Adversarial Resistance in Transportation [[PDF](https://dl.acm.org/doi/full/10.1145/3643137)]
6. `[TITS'24]` A Survey of Multi-Vehicle Consensus in Uncertain Networks for Autonomous Driving [[PDF](https://ieeexplore.ieee.org/abstract/document/10704959)]
7. `[USENIX Security'24]` On Data Fabrication in Collaborative Vehicular Perception: Attacks and Countermeasures [[PDF](http://arxiv.org/abs/2309.12955)]
8. `[AAAI'24]` Robust Communicative Multi-Agent Reinforcement Learning with Active Defense [[PDF](https://ojs.aaai.org/index.php/AAAI/article/view/29708)]
9. `[VehicleSec'23]` Cooperative Perception for Safe Control of Autonomous Vehicles under LiDAR Spoofing Attacks [[PDF](http://arxiv.org/abs/2302.07341)]
10. `[ICCV'23]` Among Us: Adversarially Robust Collaborative Perception by Consensus [[PDF](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Among_Us_Adversarially_Robust_Collaborative_Perception_by_Consensus_ICCV_2023_paper.pdf), [Code](https://github.com/coperception/ROBOSAC)] ![](https://img.shields.io/github/stars/coperception/ROBOSAC.svg?style=social&label=Star&maxAge=2592000)
11. `[TDSC'23]` MARNet: Backdoor Attacks Against Cooperative Multi-Agent Reinforcement Learning [[PDF](https://ieeexplore.ieee.org/abstract/document/9894692)]
12. `[NeurIPS'23]` Efficient Adversarial Attacks on Online Multi-agent Reinforcement Learning [[PDF](https://proceedings.neurips.cc/paper_files/paper/2023/hash/4cddc8fc57039f8fe44e23aba1e4df40-Abstract-Conference.html)]
13. `[TITS'22]` A Survey on Cyber-Security of Connected and Autonomous Vehicles (CAVs) [[PDF](https://ieeexplore.ieee.org/abstract/document/9447840)]
14. `[ICCV'21]` Adversarial Attacks On Multi-Agent Communication [[PDF](https://ieeexplore.ieee.org/document/9711249/?arnumber=9711249)]
15. `[arXiv]` GCP: Guarded Collaborative Perception with Spatial-Temporal Aware Malicious Agent Detection [[PDF](https://arxiv.org/abs/2501.02450)]
16. `[arXiv]` CP-Guard+: A New Paradigm for Malicious Agent Detection and Defense in Collaborative Perception [[PDF](https://arxiv.org/abs/2502.07807v1)]
17. `[arXiv]` A Multi-Agent Security Testbed for the Analysis of Attacks and Defenses in Collaborative Sensor Fusion [[PDF](https://arxiv.org/abs/2401.09387)]
18. `[ACM Computing Surveys]` Adversarial Machine Learning Attacks and Defences in Multi-Agent Reinforcement Learning [[PDF](https://dl.acm.org/doi/full/10.1145/3708320)]
19. `[USENIX Security'25]` From Threat to Trust: Exploiting Attention Mechanisms for Attacks and Defenses in Cooperative Perception (SOMBRA & LUCIA) [[PDF](https://www.usenix.org/conference/usenixsecurity25/presentation/wang-chenyi)]
20. `[ICCV'25]` Pretend Benign: A Stealthy Adversarial Attack by Exploiting Vulnerabilities in Cooperative Perception [[PDF](https://iccv.thecvf.com/virtual/2025/poster/2610)]
21. `[IEEE'25]` Robust Collaborative Perception: Combining Adversarial Training with Consensus Mechanism for Enhanced V2X Security [[PDF](https://ieeexplore.ieee.org/abstract/document/11097632/)]
22. `[arXiv'25]` CP-FREEZER: Latency Attacks against Cooperative Perception [[PDF](https://arxiv.org/abs/2508.01062)]



## Star History


[![Star History Chart](https://api.star-history.com/svg?repos=dl-m9/Multi-Agent-Autonomous-Driving&type=Date)](https://www.star-history.com/#dl-m9/Multi-Agent-Autonomous-Driving&Date)
