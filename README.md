# Awesome-Fine-Grained-Zero-Shot-Learning ![](https://img.shields.io/badge/Awesome-FZSL-blue)
A summarization of representative fine-grained zero-shot learning methods, covering publicly available datasets, models, implementations, etc.

--------------------------------------------------------------------------------------

:running: **We will keep updating it. If you find any mistake or have any advice, please feel free to contact us** 

--------------------------------------------------------------------------------------

# Datasets Download    

&#x1F493;: Fine-Grained Dataset
- &#x1F493; **CUB_200_2011 (CUB)** [Paper](https://www.florian-schroff.de/publications/CUB-200.pdf) | [Download Link](https://www.vision.caltech.edu/datasets/cub_200_2011/)
- &#x1F493; **Oxford Flowers (FLO)** [Paper](https://ieeexplore.ieee.org/abstract/document/4756141) | [Download Link](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- &#x1F493; **SUN Attribute (SUN)** [Paper](https://ieeexplore.ieee.org/abstract/document/6247998) | [Download Link](https://cs.brown.edu/~gmpatter/sunattributes.html)
- &#x1F493; **NABirds** [Paper](https://openaccess.thecvf.com/content_cvpr_2015/papers/Horn_Building_a_Bird_2015_CVPR_paper.pdf) | [Download Link](https://dl.allaboutbirds.org/nabirds)
- &#x1F493; **DeepFashion** [Paper](https://openaccess.thecvf.com/content_cvpr_2016/papers/Liu_DeepFashion_Powering_Robust_CVPR_2016_paper.pdf) | [Download Link](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)
- **Animals with Attributes (AWA)** [Paper](https://ieeexplore.ieee.org/abstract/document/6571196) | [Download Link](https://cvml.ista.ac.at/AwA/)
- **Animals with Attributes (2) (AWA2)** [Paper](https://ieeexplore.ieee.org/abstract/document/8413121) | [Download Link](https://cvml.ista.ac.at/AwA2/)
- **Attribute Pascal and Yahoo (APY)** [Paper](https://ieeexplore.ieee.org/abstract/document/5206772) | [Download Link](https://vision.cs.uiuc.edu/attributes/)

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


## 2022       
**:open_file_folder:** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-: 
:scroll: | **arXiv** | Fine-grained Few-shot Recognition by Deep Object Parsing | [Paper](https://arxiv.org/abs/2207.07110)/Code 
:scroll: | **arXiv** | Few-shot Fine-grained Image Classification via Multi-Frequency Neighborhood and Double-cross Modulation | [Paper](https://arxiv.org/abs/2207.08547)/[Code](https://github.com/ChengqingLi/FicNet)   
:triangular_flag_on_post: | **MM** | Learning Cross-Image Object Semantic Relation in Transformer for Few-Shot Fine-Grained Image Classification | [Paper](https://arxiv.org/abs/2207.00784)/[Code](https://github.com/JiakangYuan/HelixFormer) 
:scroll: | **AAAI** | Dual Attention Networks for Few-Shot Fine-Grained Recognition | [Paper](https://www.aaai.org/AAAI22Papers/AAAI-1537.XuSL.pdf)/Code 
:triangular_flag_on_post: | **CVPR** | Task Discrepancy Maximization for Fine-Grained Few-Shot Classification | [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Lee_Task_Discrepancy_Maximization_for_Fine-Grained_Few-Shot_Classification_CVPR_2022_paper.html)/[Code](https://github.com/leesb7426/CVPR2022-Task-Discrepancy-Maximization-for-Fine-grained-Few-Shot-Classification) 
:triangular_flag_on_post: | **PR** | Learning Attention-Guided Pyramidal Features for Few-shot Fine-grained Recognition | [Paper](https://www.sciencedirect.com/science/article/pii/S0031320322002734)/[Code](https://github.com/CSer-Tang-hao/AGPF-FSFG)  
:scroll: | **PR** | Query-Guided Networks for Few-shot Fine-grained Classification and Person Search | [Paper](https://www.sciencedirect.com/science/article/pii/S0031320322005295)/Code  
:triangular_flag_on_post: | **TPAMI** | Reinforcing Generated Images via Meta-learning for One-Shot Fine-Grained Visual Recognition | [Paper](https://ieeexplore.ieee.org/abstract/document/9756906)/[Code](https://github.com/apple2373/MetaIRNet)  

## 2021       
**:open_file_folder:** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-: 
:scroll: | **arXiv** | NDPNet: A novel non-linear data projection network for few-shot fine-grained image classification | [Paper](https://arxiv.org/abs/2106.06988)/Code  
:triangular_flag_on_post: | **arXiv** | Compositional Fine-Grained Low-Shot Learning | [Paper](https://arxiv.org/abs/2105.10438)/Code  
:triangular_flag_on_post: | **ICIP** | Coupled Patch Similarity Network For One-Shot Fine-Grained Image Recognition | [Paper](https://ieeexplore.ieee.org/abstract/document/9506685/)/[Code](https://github.com/CSer-Tang-hao/CPSN-OSFG)  
:triangular_flag_on_post: | **ICME** | Selective, Structural, Subtle: Trilinear Spatial-Awareness for Few-Shot Fine-Grained Visual Recognition | [Paper](https://ieeexplore.ieee.org/abstract/document/9428223/)/[Code](https://github.com/iCVTEAM/S3Net)  
:triangular_flag_on_post: | **CVPR** | Few-Shot Classification With Feature Map Reconstruction Networks | [Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Wertheimer_Few-Shot_Classification_With_Feature_Map_Reconstruction_Networks_CVPR_2021_paper.html)/[Code](https://github.com/Tsingularity/FRN)  
:triangular_flag_on_post: | **MM** | Object-aware long-short-range spatial alignment for few-shot fine-grained image classification | [Paper](https://arxiv.org/abs/2108.13098)/Code
:triangular_flag_on_post: | **ICCV** | Variational Feature Disentangling for Fine-Grained Few-Shot Classification | [Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Xu_Variational_Feature_Disentangling_for_Fine-Grained_Few-Shot_Classification_ICCV_2021_paper.html)/[Code](https://github.com/cvlab-stonybrook/vfd-iccv21)
:scroll: | **NC** | Fine-grained few shot learning with foreground object transformation | [Paper](https://www.sciencedirect.com/science/article/pii/S0925231221013746)/Code  
:scroll: | **KBS** | Few-shot fine-grained classification with Spatial Attentive Comparison | [Paper](https://www.sciencedirect.com/science/article/pii/S0950705121001039)/Code   
:triangular_flag_on_post: | **TPAMI** | Power Normalizations in Fine-Grained Image, Few-Shot Image and Graph Classification | [Paper](https://ieeexplore.ieee.org/abstract/document/9521687/)/Code    

## 2020       
**:open_file_folder:** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-: 
:triangular_flag_on_post: | **CVPR** | Revisiting Pose-Normalization for Fine-Grained Few-Shot Recognition | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Tang_Revisiting_Pose-Normalization_for_Fine-Grained_Few-Shot_Recognition_CVPR_2020_paper.html)/[Code](https://github.com/Tsingularity/PoseNorm_Fewshot)   
:scroll: | **IJCAI** | Multi-attention Meta Learning for Few-shot Fine-grained Image Recognition | [Paper](https://www.ijcai.org/proceedings/2020/0152.pdf)/Code
:scroll: | **ICME** | Knowledge-Based Fine-Grained Classification For Few-Shot Learning | [Paper](https://ieeexplore.ieee.org/abstract/document/9102809)/Code 
:scroll: | **TIE** | Few-Shot Learning for Domain-Specific Fine-Grained Image Classification | [Paper](https://ieeexplore.ieee.org/abstract/document/9027090)/[Code](https://github.com/xhw205/Domain-specific-Fewshot-Learning)   
:scroll: | **TMM** | Low-Rank Pairwise Alignment Bilinear Network For Few-Shot Fine-Grained Image Classification | [Paper](https://ieeexplore.ieee.org/abstract/document/9115215)/Code   
:triangular_flag_on_post: | **TIP** | BSNet: Bi-similarity network for few-shot fine-grained image classification | [Paper](https://ieeexplore.ieee.org/document/9293172)/[Code](https://github.com/PRIS-CV/BSNet)    
:scroll: | **ACL** | Shaping Visual Representations with Language for Few-shot Classification | [Paper](https://arxiv.org/abs/1911.02683)/[Code](https://github.com/jayelm/lsl)  
:triangular_flag_on_post: | **NeurIPS** | Compositional Zero-Shot Learning via Fine-Grained Dense Feature Composition | [Paper](https://proceedings.neurips.cc/paper/2020/file/e58cc5ca94270acaceed13bc82dfedf7-Paper.pdf)/[Code](https://github.com/hbdat/neurIPS20_CompositionZSL) 


## 2019       
**:open_file_folder:** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-: 
:scroll: | **ICME** | Compare More Nuanced: Pairwise Alignment Bilinear Network for Few-Shot Fine-Grained Learning | [Paper](https://ieeexplore.ieee.org/abstract/document/8784745)/Code    
:triangular_flag_on_post: | **TIP** | Piecewise Classifier Mappings: Learning Fine-Grained Learners for Novel Categories With Few Examples | [Paper](https://ieeexplore.ieee.org/abstract/document/8752297)/Code  

# Video Classification
## 2021        
**:open_file_folder:** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-:   
:triangular_flag_on_post: | **MM** | Few-Shot Fine-Grained Action Recognition via Bidirectional Attention and Contrastive Meta-Learning | [Paper](https://dl.acm.org/doi/abs/10.1145/3474085.3475216)/[Code](https://github.com/acewjh/FSFG) 

## 2023        
**:open_file_folder:** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-:   
:triangular_flag_on_post: | **MM** | M$^3$Net: Multi-view Encoding, Matching, and Fusion for Few-shot Fine-grained Action Recognition | [Paper](https://arxiv.org/abs/2308.03063)/Code
