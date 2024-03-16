# Awesome-Fine-Grained-Zero-Shot-Learning

A summarization of representative fine-grained zero-shot learning methods, covering publicly available datasets, models, implementations, etc. 

Please feel free to contact us ([jingcai.guo@ieee.org](jingcai.guo@ieee.org)) if you have any advice.

# Datasets  

- CUB: [Caltech-UCSD Birds 200](https://www.florian-schroff.de/publications/CUB-200.pdf) [[Download](https://www.vision.caltech.edu/datasets/cub_200_2011/)]
- FLO: [Automated Flower Classification over a Large Number of Classes](https://ieeexplore.ieee.org/abstract/document/4756141) [[Download](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)]
- SUN: [SUN attribute database: Discovering, annotating, and recognizing scene attributes](https://ieeexplore.ieee.org/abstract/document/6247998) [[Download](https://cs.brown.edu/~gmpatter/sunattributes.html)]
- NABirds: [Building a bird recognition app and large scale dataset with citizen scientists: The fine print in fine-grained dataset collection](https://openaccess.thecvf.com/content_cvpr_2015/papers/Horn_Building_a_Bird_2015_CVPR_paper.pdf) [[Download](https://dl.allaboutbirds.org/nabirds)]
- DeepFashion: [DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations](https://openaccess.thecvf.com/content_cvpr_2016/papers/Liu_DeepFashion_Powering_Robust_CVPR_2016_paper.pdf) [[Download](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)]
- AWA: [Attribute-Based Classification for Zero-Shot Visual Object Categorization](https://ieeexplore.ieee.org/abstract/document/6571196) [[Download](https://cvml.ista.ac.at/AwA/)]
- AWA2: [Zero-Shot Learning—A Comprehensive Evaluation of the Good, the Bad and the Ugly](https://ieeexplore.ieee.org/abstract/document/8413121) [[Download](https://cvml.ista.ac.at/AwA2/)]
- APY: [Describing objects by their attributes](https://ieeexplore.ieee.org/abstract/document/5206772) [[Download](https://vision.cs.uiuc.edu/attributes/)]

--------------------------------------------------------------------------------------

# Attention-Based Methods
      
**Title** | **Venue** | **Backbone** | **FineTune** | **Resolution** | **Datasets** | **Code**   
:-: | :-: | :-  | :-: | :-: | :-: | :-: |
[Discriminative learning of latent features for zero-shot recognition](https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Discriminative_Learning_of_CVPR_2018_paper.pdf) | CVPR'18 | GoogleNet, VGG19 | ✔️ | 224x224 | CUB, AWA | [Code](https://github.com/zbxzc35/Zero_shot_learning_using_LDF_tensorflow)
[Attribute Attention for Semantic Disambiguation in Zero-Shot Learning](https://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Attribute_Attention_for_Semantic_Disambiguation_in_Zero-Shot_Learning_ICCV_2019_paper.pdf) | ICCV'19 | GoogleNet, ResNet101, VGG19 | ✔️ | 224x224 | CUB, SUN, AWA2 | [Code](https://github.com/ZJULearning/AttentionZSL)
[Attentive Region Embedding Network for Zero-shot Learning](https://openaccess.thecvf.com/content_CVPR_2019/papers/Xie_Attentive_Region_Embedding_Network_for_Zero-Shot_Learning_CVPR_2019_paper.pdf) | CVPR'19 | ResNet101 | ✔️ | 224x224 | CUB, SUN, AWA2, APY | [Code](https://github.com/gsx0/Attentive-Region-Embedding-Network-for-Zero-shot-Learning)
[Semantic-Guided Multi-Attention Localization for Zero-Shot Learning](https://proceedings.neurips.cc/paper_files/paper/2019/file/172fd0d638b3282151bd8f3d652cb640-Paper.pdf) | NeurIPS'19 | VGG19 | ✔️ | 448x448 | CUB, FLO, AWA | [Code](https://github.com/wuhuicumt/LearningWhereToLook/tree/master)
[Region Graph Embedding Network for Zero-Shot Learning](https://link.springer.com/chapter/10.1007/978-3-030-58548-8_33) | ECCV'20 | ResNet101 | ✔️ | 224x224 | CUB, SUN, AWA2, APY | Code
[Fine-Grained Generalized Zero-Shot Learning via Dense Attribute-Based Attention](https://openaccess.thecvf.com/content_CVPR_2020/papers/Huynh_Fine-Grained_Generalized_Zero-Shot_Learning_via_Dense_Attribute-Based_Attention_CVPR_2020_paper.pdf) | CVPR'20 | ResNet101 | ❌ | 224x224 | CUB, SUN, DeepFashion, AWA2 | [Code](https://github.com/hbdat/cvpr20_DAZLE)
[Region Semantically Aligned Network for Zero-Shot Learning](https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Discriminative_Learning_of_CVPR_2018_paper.pdf) | CIKM'21 | ResNet101 | - | 448x448 | CUB, SUN, AWA2 | Code
[Goal-Oriented Gaze Estimation for Zero-Shot Learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Goal-Oriented_Gaze_Estimation_for_Zero-Shot_Learning_CVPR_2021_paper.pdf) | CVPR'21 | ResNet101 | ✔️ | 448x448 | CUB, SUN, AWA2 | [Code](https://github.com/osierboy/GEM-ZSL)
[I2DFormer: Learning Image to Document Attention for Zero-Shot Image Classification](https://proceedings.neurips.cc/paper_files/paper/2022/file/4fca3029c9ead4551937ed6987502e5f-Paper-Conference.pdf) | NeurIPS'22 | ViT-B | ✔️ | 224x224 | CUB, FLO, AWA2 | [Code](https://github.com/ferjad/I2DFormer)
[MSDN: Mutually Semantic Distillation Network for Zero-Shot Learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_MSDN_Mutually_Semantic_Distillation_Network_for_Zero-Shot_Learning_CVPR_2022_paper.pdf) | CVPR'22 | ResNet101 | ❌ | 448x448 | CUB, SUN, AWA2 | [Code](https://github.com/shiming-chen/MSDN)
[TransZero: Attribute-Guided Transformer for Zero-Shot Learning](https://ojs.aaai.org/index.php/AAAI/article/view/19909) | AAAI'22 | ResNet101 | ❌ | 448x448 | CUB, SUN, AWA2 | [Code](https://github.com/shiming-chen/TransZero)
[DUET: Cross-Modal Semantic Grounding for Contrastive Zero-Shot Learning](https://ojs.aaai.org/index.php/AAAI/article/view/25114) | AAAI'23 | ViT-B | ✔️ | 224x224 | CUB, SUN, AWA2 | [Code](https://github.com/zjukg/DUET)
[Progressive Semantic-Visual Mutual Adaption for Generalized Zero-Shot Learning](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_Progressive_Semantic-Visual_Mutual_Adaption_for_Generalized_Zero-Shot_Learning_CVPR_2023_paper.pdf) | CVPR'23 | ViT-B | ✔️ | 224x224 | CUB, SUN, AWA2 | [Code](https://github.com/ManLiuCoder/PSVMA)


# Non-Attention Methods

## Prototype Learning
**Title** | **Venue** | **Backbone** | **FineTune** | **Resolution** | **Datasets** | **Code**   
:-: | :-: | :-  | :-: | :-: | :-: | :-: |
[Attribute Prototype Network for Zero-Shot Learning](https://proceedings.neurips.cc/paper_files/paper/2020/file/fa2431bf9d65058fe34e9713e32d60e6-Paper.pdf) | NeurIPS'20 | ResNet101 | ✔️ | 224x224 | CUB, SUN, AWA2 | [Code](https://github.com/wenjiaXu/APN-ZSL/tree/master)
[Dual Progressive Prototype Network for Generalized Zero-Shot Learning](https://proceedings.neurips.cc/paper_files/paper/2021/file/1700002963a49da13542e0726b7bb758-Paper.pdf) | NeurIPS'21 | ResNet101 | ✔️ | 448x448 | CUB, SUN, AWA2, APY | [Code](https://github.com/Roxanne-Wang/DPPN-GZSL)
[Dual Part Discovery Network for Zero-Shot Learning](https://dl.acm.org/doi/abs/10.1145/3503161.3547889) | MM'22 | ResNet101 | ❌ | 448x448 | CUB, SUN, AWA2 | Code
[Boosting Zero-shot Learning via Contrastive Optimization of Attribute Representations](https://arxiv.org/pdf/2207.03824.pdf) | TNNLS'23 | ResNet101, ViT-L | ✔️ | 224x224, 448x448 | CUB, SUN, AWA2 | [Code](https://github.com/dyabel/CoAR-ZSL)

## Data Manipulation
**Title** | **Venue** | **Backbone** | **FineTune** | **Resolution** | **Datasets** | **Code**   
:-: | :-: | :-  | :-: | :-: | :-: | :-: |
[Link the head to the “beak”: Zero Shot Learning from Noisy Text Description at Part Precision](https://openaccess.thecvf.com/content_cvpr_2017/papers/Elhoseiny_Link_the_Head_CVPR_2017_paper.pdf) | CVPR'17 | VGG16 | ❌ | - | CUB, NABirds | [Code](https://github.com/EthanZhu90/ZSL_PP_CVPR17)
[Stacked Semantics-Guided Attention Model for Fine-Grained Zero-Shot Learning](https://proceedings.neurips.cc/paper_files/paper/2018/file/9087b0efc7c7acd1ef7e153678809c77-Paper.pdf) | NeurIPS'18 | VGG16 | ❌ | - | CUB, NABirds | [Code](https://github.com/ylytju/sga/tree/master)
[Semantic-guided Reinforced Region Embedding for Generalized Zero-Shot Learning](https://ojs.aaai.org/index.php/AAAI/article/view/16230) | AAAI'21 | ResNet101 | - | 448x448 | CUB, SUN, AWA2, APY | Code
[VGSE: Visually-Grounded Semantic Embeddings for Zero-Shot Learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_VGSE_Visually-Grounded_Semantic_Embeddings_for_Zero-Shot_Learning_CVPR_2022_paper.pdf) | CVPR'22 | ResNet50 | - | - | CUB, SUN, AWA2 | [Code](https://github.com/wenjiaXu/VGSE)

## Graph Modeling
**Title** | **Venue** | **Backbone** | **FineTune** | **Resolution** | **Datasets** | **Code**   
:-: | :-: | :-  | :-: | :-: | :-: | :-: |
[Attribute Propagation Network for Graph Zero-Shot Learning](https://ojs.aaai.org/index.php/AAAI/article/view/5923) | AAAI'20 | ResNet101 | ❌ | - | CUB, SUN, AWA, AWA2, APY | Code
[GNDAN: Graph Navigated Dual Attention Network for Zero-Shot Learning](https://ieeexplore.ieee.org/abstract/document/9768177) | TNNLS'22 | ResNet101 | ❌ | 448x448 | CUB, SUN, AWA2 | [Code](https://github.com/shiming-chen/GNDAN)
[Graph Knows Unknowns: Reformulate Zero-Shot Learning as Sample-Level Graph Recognition](https://ojs.aaai.org/index.php/AAAI/article/view/25942) | AAAI'23 | ResNet34 | - | - | CUB, NABirds | Code
[Explanatory Object Part Aggregation for Zero-Shot Learning](https://ieeexplore.ieee.org/abstract/document/10287616) | TPAMI'23 | AlexNet, ResNet50 | ✔️ | - | CUB, SUN, FLO, AWA2 | Code

## Generative
**Title** | **Venue** | **Backbone** | **FineTune** | **Resolution** | **Datasets** | **Code**   
:-: | :-: | :-  | :-: | :-: | :-: | :-: |
[A Generative Adversarial Approach for Zero-Shot Learning from Noisy Texts](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhu_A_Generative_Adversarial_CVPR_2018_paper.pdf) | CVPR'18 | VGG16 | ❌ | 224x224 | CUB, NABirds | [Code](https://github.com/EthanZhu90/ZSL_GAN)
[Compositional Zero-Shot Learning via Fine-Grained Dense Feature Composition](https://proceedings.neurips.cc/paper_files/paper/2020/file/e58cc5ca94270acaceed13bc82dfedf7-Paper.pdf) | NeurIPS'20 | ResNet101 | ❌ | 224x224 | CUB, SUN, DeepFashion, AWA2 | [Code](https://github.com/hbdat/neurIPS20_CompositionZSL)
[Zero-Shot Learning With Attentive Region Embedding and Enhanced Semantics](https://ieeexplore.ieee.org/abstract/document/9881214) | TNNLS'22 | ResNet101 | ❌ | 224x224 | CUB, SUN, AWA, AWA2, APY | Code

## Attribute Selection
**Title** | **Venue** | **Backbone** | **FineTune** | **Resolution** | **Datasets** | **Code**   
:-: | :-: | :-  | :-: | :-: | :-: | :-: |
[Multi-Cue Zero-Shot Learning with Strong Supervision](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Akata_Multi-Cue_Zero-Shot_Learning_CVPR_2016_paper.pdf) | CVPR'16 | VGG16 | ❌ | 224x224 | CUB | Code

## Citation

    @article{guo2024fine,
      author    = {Jingcai Guo and
                   Zhijie Rao and
                   Zhi Chen and
                   Jingren Zhou and
                   Dacheng Tao},
      title     = {Fine-Grained Zero-Shot Learning: Advances, Challenges, and Prospects},
      journal   = {arXiv preprint arXiv:2401.17766},
      year      = {2024},
      url       = {https://arxiv.org/abs/2401.17766}
    }
