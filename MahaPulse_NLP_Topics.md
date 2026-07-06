# महापल्स — NLP Topics Reference

## Project

**महापल्स: AI-Powered Marathi Sentiment & Opinion Analysis System**

---

## Problem Statement

With the rapid growth of digital platforms, millions of Marathi speakers
express their opinions through customer reviews, social media posts, news
comments, surveys, and feedback portals every day. Despite Marathi being
one of the most widely spoken languages in India (spoken by over 83 million
people), most existing sentiment analysis solutions are designed primarily
for English and offer limited or inaccurate support for Marathi [1].
Consequently, businesses, government organizations, media agencies, and
researchers struggle to efficiently understand public opinion, identify
emerging issues, and make data-driven decisions from large volumes of
Marathi text [2]. Manual analysis of such data is time-consuming,
inconsistent, and impractical at scale. Therefore, there is a need for an
intelligent NLP-based system capable of automatically processing Marathi
text, classifying sentiment, extracting meaningful insights, and presenting
analytical visualizations through an easy-to-use web platform. The proposed
system, **महापल्स**, aims to bridge this gap by providing an AI-powered
Marathi sentiment analysis platform that enables efficient opinion mining
and decision support for regional language data [3].

---

## NLP Topics Used in महापल्स

The following topics represent the complete, ordered NLP pipeline that
महापल्स uses — from raw text to sentiment classification. Every topic
below is directly applied in the system; no background or optional topics
are included.

---

## 1. Text Preprocessing

Text preprocessing is the foundational step for any NLP system. Because
Marathi is written in the Devanagari script and is morphologically rich,
specialized preprocessing is required before any model can process it [4].
महापल्स applies the following six steps in sequence.

### 1.1 Text Cleaning

Text cleaning removes content that carries no linguistic meaning for
sentiment analysis. For social media and review data in Marathi, this
includes removing URLs, HTML tags, and non-Devanagari characters (such as
stray Roman or Arabic script mixed into Marathi text) [5]. The
**mahaNLP** library provides built-in utilities for this step [6].

### 1.2 Noise Removal

Noise removal is the process of stripping out elements that are present in
user-generated content but are not part of the language itself — such as
emojis, punctuation sequences, hashtags, and special symbols [5]. In
Marathi social media data, mixing of scripts and irregular punctuation is
particularly common, making noise removal critical for maintaining data
quality [4].

### 1.3 Sentence Segmentation

Sentence segmentation identifies the boundaries between sentences in a
block of Marathi text. Marathi uses the Devanagari *danda* (।) as a
sentence-ending marker in addition to standard punctuation, which requires
language-specific segmentation rules [6]. Correct segmentation ensures
that the model processes each sentence as a self-contained unit of meaning.

### 1.4 Tokenization

Tokenization splits a Marathi sentence into individual meaningful units
(tokens), which serve as the primary input for downstream model processing.
Because Marathi is morphologically rich, simple whitespace splitting is
insufficient; subword tokenization methods such as **WordPiece** (used
internally by BERT-based models) and **SentencePiece** are more effective
for handling out-of-vocabulary (OOV) words and complex word formations [4].
The mahaNLP library provides a dedicated Marathi tokenizer [6].

### 1.5 Stop-Word Removal

Stop words are high-frequency function words (e.g., आणि, तो, ते) that
appear frequently but contribute little to sentiment. Removing them reduces
dimensionality and helps the model focus on semantically meaningful content
[7]. A curated list of 400 Marathi stop words derived from the L3Cube
MahaCorpus (24.8 million sentences) is available for this purpose [7].

### 1.6 Lemmatization

Lemmatization reduces an inflected Marathi word to its dictionary base form
(lemma), preserving actual linguistic meaning — unlike stemming, which
merely strips suffixes and may produce non-words [8]. Because Marathi is
highly inflectional, lemmatization is preferred over stemming to ensure
semantic accuracy during feature extraction and model training [8].

> **Note on Case Normalization:** Case normalization (uppercase/lowercase
> conversion) is **not applicable** to Marathi. The Devanagari script has no
> concept of letter case, so this step is skipped entirely.

---

## 2. Language Representation

After preprocessing, text must be converted into numerical representations
that a machine learning model can process. महापल्स uses contextual
embeddings — the state-of-the-art approach for transformer-based systems.

### 2.1 Contextual Embeddings

Unlike static word embeddings (e.g., Word2Vec, GloVe), contextual
embeddings generate a unique vector for each word *based on its surrounding
context* in a sentence [9]. This means the same word can have different
representations depending on how it is used, which is critical for
capturing nuanced sentiment in Marathi text. Contextual embeddings are
produced by the encoder layers of transformer models such as BERT [9].

---

## 3. Language Model

महापल्स uses a pre-trained transformer-based language model as its core,
fine-tuned on the Marathi sentiment analysis task.

### 3.1 Transformer Architecture

The Transformer is a deep learning architecture based entirely on
self-attention mechanisms, dispensing with recurrence and convolutions [10].
Its encoder produces rich, bidirectional contextual representations of
input text. All BERT-family models (including IndicBERT and MahaBERT) are
built on this architecture.

### 3.2 MahaBERT / IndicBERT (Fine-tuned for महापल्स)

महापल्स uses one of the following two pre-trained models (selected based
on evaluation performance):

- **MahaBERT** — A monolingual BERT model pre-trained on the L3Cube
  MahaCorpus, a large Marathi-specific corpus. As a dedicated Marathi
  model, it consistently outperforms multilingual alternatives on
  Marathi downstream tasks such as sentiment analysis, NER, and text
  classification [3].

- **IndicBERT** — A multilingual ALBERT model covering 12 Indian languages
  including Marathi, developed by AI4Bharat. It is suitable for
  cross-lingual transfer and general Indic NLP tasks [11].

The selected model is fine-tuned on a labelled Marathi sentiment dataset
using the Hugging Face Transformers library.

---

## महापल्स NLP Pipeline (Summary)

The complete pipeline in execution order:

| Step | Topic | Purpose |
|------|-------|---------|
| 1 | Text Cleaning | Remove URLs, HTML, non-Devanagari characters |
| 2 | Noise Removal | Remove emojis, hashtags, special symbols |
| 3 | Sentence Segmentation | Identify sentence boundaries |
| 4 | Tokenization | Split sentences into subword tokens |
| 5 | Stop-Word Removal | Filter semantically insignificant words |
| 6 | Lemmatization | Reduce words to their base form |
| 7 | Contextual Embeddings | Generate context-aware word vectors |
| 8 | Transformer (MahaBERT / IndicBERT) | Classify sentiment |

---

## References

[1] D. Kakwani et al., "IndicNLPSuite: Monolingual Corpora, Evaluation
Benchmarks and Pre-trained Multilingual Language Models for Indian
Languages," in *Findings of EMNLP 2020*, 2020.

[2] G. Kale et al., "Robust Sentiment Analysis for Low Resource Languages
Using Data Augmentation Approaches: A Case Study in Marathi," *arXiv
preprint arXiv:2311.xxxxx*, 2023.

[3] R. Joshi, "L3Cube-MahaCorpus and MahaBERT: Marathi Monolingual Corpus,
Marathi BERT Language Models, and Resources," in *Proc. WILDRE-6 Workshop,
LREC 2022*, Marseille, France: ELRA, 2022, pp. 97–101.

[4] R. Joshi, "L3Cube-MahaNLP: Marathi Natural Language Processing
Datasets, Models, and Library," *arXiv preprint arXiv:2205.14728*, 2022.

[5] V. Magdum, O. J. Dhekane, S. S. Hiwarkhedkar, S. S. Mittal, and R.
Joshi, "mahaNLP: A Marathi Natural Language Processing Library," in *Proc.
IJCNLP-AACL 2023 (Demo Track)*, 2023.

[6] V. Magdum et al., "mahaNLP: A Marathi Natural Language Processing
Library," *ACL Anthology*, 2023. [Online]. Available:
https://aclanthology.org/2023.aacl-demo.12/

[7] R. Joshi, "L3Cube-MahaNLP: Marathi Natural Language Processing
Datasets, Models, and Library," *arXiv preprint arXiv:2205.14728*, 2022.
*(Stop-word list derived from MahaCorpus, 24.8 M sentences.)*

[8] R. J. Sutar and K. R. Desai, "Sentiment Analysis for Transliterated
Hindi and Marathi Language using Machine Learning Approach," *International
Journal of Computer and Engineering in Science and Engineering*, 2025.

[9] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training
of Deep Bidirectional Transformers for Language Understanding," in *Proc.
NAACL-HLT 2019*, Minneapolis, MN: ACL, 2019, pp. 4171–4186.

[10] A. Vaswani et al., "Attention is all you need," in *Advances in Neural
Information Processing Systems (NeurIPS)*, 2017, pp. 5998–6008.

[11] D. Kakwani et al., "IndicNLPSuite: Monolingual Corpora, Evaluation
Benchmarks and Pre-trained Multilingual Language Models for Indian
Languages," in *Findings of EMNLP 2020*, 2020.
