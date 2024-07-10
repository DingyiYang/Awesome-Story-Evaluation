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
  <img src="imgs/tasks.pdf" alt="LLM evaluation" width="700"><br>
</div>
<br>
### Benchmarks
|  Corpora  |    Data Source   |  Annotations | Correlated Task   |   Domain  |
|:--------|:--------:|:--------:|:--------:|:--------:|
| [**Children's Books**]()  | [Download]() | Story Context, Query→Infilling Entity | Story Completion | Fairy Tale |
| [**CNN**]() |  [Download]() | Story Context, Query→Infilling Entity |  Story Completion |  News |
| [**Story Cloze Test**]()  |  [Download]() | Story Context→Ending  | Story Completion |  Commonsense |
| [**RocStories**]()  |  [Download]() | Title→Five-Sentence Story  | Story Generation |  Commonsense |
| [**NYTimes**]()  |  [Download]() | Title→Outline[143]→Story  | Story Generation |  News |
| [**WritingPrompts**]() |  [Download]() | Prompt→Outline→Story  | Story Generation  | Real World |
| [**Mystery**]()  |  [Download]() | Outline→Story |  Story Generation |  Fiction |
| [**Fairy Tales**]()  |  [Download]() | Outline→Story  | Story Generation  | Fiction |
| [**Hippocorpus**]()  |  [Download]() | Prompt→Story  | Story Generation  | General |
| [**STORIUM**]()  |  [Download]() | Structural Prompt→Story  | Story Generation |  Fiction |
| [**TVSTORYGEN**]() |  [Download]() | Prompt, Character Descriptions→Story  | Story Generation  | TV Show |
| [**LOT**]()  |  [Download]() | Title→Outline→Story  | Story Completion/Generation  | Fiction |

<a name="papers"></a>
## Papers
