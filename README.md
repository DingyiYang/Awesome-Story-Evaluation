# Awesome-Story-Evaluation
We release a survey paper for story evaluation:[], providing a thorough review of story evaluation and existing evaluation methods that can be proposed or adopted for story evaluation.

In this repository, we introduce the main contents of our survey. We provide detailed collections of story generation/evaluation benchmarks, and correlated evaluation methods.

## Table of Contents
- [Introduction](#introduction)
- [Story Generation](#story-generation)
  - Tasks
  - Benchmarks
- [Story Evaluation](#papers)
  - Criteria
  - Benmarks
- [Metrics](#papers)
  - Traditional
  - LLM-Based
  - Collaborative
  - 

<a name="introduction"></a>
## Introduction
With the development of artificial intelligence, particularly the success of Large Language Models (LLMs), the quantity and quality of automatically generated stories have significantly increased. This has led to the need to explore automatic story evaluation to assess the generative capabilities of computing systems and analyze the quality of both automatic-generated and human-written stories. Evaluating a story can be more challenging than other generated text evaluation tasks. While tasks like language translation primarily focus on assessing the aspects of fluency and accuracy, story evaluation demands complex additional measures such as overall coherence, character development, interestingness, etc. This requires a thorough review of relevant research.
  In this survey, we first summarize existing storytelling tasks, including text-to-text, visual-to-text, and text-to-visual. We highlight their evaluation challenges, identify various human criteria to measure stories, and present existing benchmark datasets. Then, we propose a taxonomy to organize evaluation metrics that have been developed or can be adopted for story evaluation. We also provide descriptions of these metrics, along with the discussion of their merits and limitations. Later, we discuss the human-AI collaboration for story evaluation and  generation. Finally, we suggest potential future research directions, extending from story evaluation to general evaluations.

<a name="story-generation"></a>
## Story Generation
### Tasks
<br>
<div align="left">
  <img src="imgs/tasks.png" alt="LLM evaluation" width="700"><br>
</div>
<br>

### Benchmarks

|  Corpora  | Paper |   Data Source  |  Annotations | Correlated Task |   Domain  |
|:------------|:--------:|:--------:|:--------:|:------:|:--------:|
| Children's Books  | [Paper](http://arxiv.org/abs/1511.02301) | [Download](https://research.facebook.com/downloads/babi) | Story Context, Query→Infilling Entity | Story Completion | Fairy Tale |
| CNN | [Paper](https://proceedings.neurips.cc/paper/2015/hash/afdec7005cc9f14302cd0474fd0f3c96-Abstract.html) |  [Download](https://github.com/abisee/cnn-dailymail) | Story Context, Query→Infilling Entity |  Story Completion |  News |
| Story Cloze Test  | [Paper](https://doi.org/10.18653/v1/n16-1098) |  [Download](https://cs.rochester.edu/nlp/rocstories/) | Story Context→Ending  | Story Completion |  Commonsense |
| RocStories  | [Paper](https://doi.org/10.18653/v1/n16-1098) |  [Download](https://cs.rochester.edu/nlp/rocstories/) | Title→Five-Sentence Story  | Story Generation |  Commonsense |
| NYTimes  | [Paper](https://aclanthology.org/2020.emnlp-main.349.pdf) |  [Download](https://github.com/hrashkin/plotmachines/tree/master/src/preprocessing) | Title→Outline→Story  | Story Generation |  News |
| WritingPrompts |  [Paper](https://aclanthology.org/P18-1082.pdf) | [Download](https://www.kaggle.com/datasets/ratthachat/writing-prompts) | Prompt→Story  | Story Generation  | Real World |
| Mystery  | [Paper](https://arxiv.org/pdf/2001.10161) |  [Download](https://github.com/rajammanabrolu/WorldGeneration) | Outline→Story |  Story Generation |  Fiction |
| Fairy Tales  | [Paper](https://arxiv.org/pdf/2001.10161) |  [Download](https://github.com/rajammanabrolu/WorldGeneration) | Outline→Story  | Story Generation  | Fiction |
| Hippocorpus  |  [Paper](https://aclanthology.org/2020.acl-main.178/) | [Download](http://aka.ms/hippocorpus) | Prompt→Story  | Story Generation  | General |
| STORIUM  |  [Paper](https://aclanthology.org/2020.emnlp-main.525/) | [Download](https://github.com/dojoteef/storium-frontend) | Structural Prompt→Story  | Story Generation |  Fiction |
| TVSTORYGEN |  [Paper](https://arxiv.org/pdf/2109.08833) | [Download](https://github.com/mingdachen/TVRecap) | Character Descriptions, Prompt→Story  | Story Generation  | TV Show |
| LOT | [Paper](https://aclanthology.org/2022.tacl-1.25.pdf) |  [Download](https://github.com/thu-coai/LOT-LongLM) | Title→Outline→Story  | Story Completion/Generation  | Fiction |
| GPT-BOOKSUM | [Paper](https://aclanthology.org/2023.findings-emnlp.723.pdf) | [Download](https://github.com/YichenZW/Pacing) | Outline→Story | Story/Plot Generation | Fiction |
| Image Paragraph | [Paper](https://arxiv.org/pdf/1611.06607) | [Download](https://cs.stanford.edu/people/ranjaykrishna/im2p/index.html) | Image→Story | Image Paragraph Captioning | Real World |
| Travel Blogs | [Paper](https://proceedings.neurips.cc/paper_files/paper/2015/file/17e62166fc8586dfa4d1bc0e1742c08b-Paper.pdf) | [Download](https://github.com/cesc-park/CRCN/tree/master) |  Image→Story |  Visual Storytelling |  Real World |
| VIST | [Paper](https://aclanthology.org/N16-1147v2.pdf) | [Download](https://visionandlanguage.net/VIST/) | Image Sequence→Story |  Visual Storytelling |  Real World |
| AESOP | [Paper](https://ieeexplore.ieee.org/document/9710625) | [Download](https://github.com/adobe-research/aesop) | Image Sequence→Story |  Visual Storytelling |  Real World |
| Video Storytelling | [Paper](https://arxiv.org/pdf/1807.09418v3) | [Download](https://zenodo.org/records/2383739) | Video→Story |  Video Storytelling  | Real World |
| VWP | [Paper](https://aclanthology.org/2023.tacl-1.33.pdf) | [Download](https://vwprompt.github.io/) | Image Sequence→Story |  Visual Storytelling |  Movie |
| Album Storytelling | [Paper](https://arxiv.org/pdf/2305.12943) | - | Image Sequence→Story  | Visual Storytelling |  Real World |
| MUGEN | [Paper](https://arxiv.org/pdf/2204.08058) | [Download](https://mugen-org.github.io/data) | Story→Video |  Story Visualization |  Game |
| PororoSV | [Paper]() | [Download]() | Story→Image Sequence  | Story Visualization  | Cartoon |
| FlintstonesSV| [Paper]() | [Download]() | Story→Image Sequence  | Story Visualization  | Cartoon |
| DiDeMoSV | [Paper]() | [Download]() | Story→Image Sequence |  Story Visualization  | Real World |
| StorySalon | [Paper]() | [Download]() | Story→Image Sequence  | Story Visualization |  Animation |
| MovieNet-TeViS | [Paper]() | [Download]() | Story→Image Sequence  | Story Illustration  | Movie |
| CMD | [Paper]() | [Download]() | Story→Video Clip Sequence  | Story Illustration  | Movie |
| CVSV | [Paper]() | [Download]() | Story→Video Clip Sequence |  Story Illustration  | Movie |
| StoryBench | [Paper]() | [Download]() | Textual Story, Video Prompt→Video  | Continuous Story Visualization  | Real World |

<a name="papers"></a>
## Papers
