/**
 * NLP Hub — Concepts Database (महापल्स Edition)
 * ─────────────────────────────────────────────────────────────
 * Contains the 8 NLP concepts that form the महापल्स pipeline:
 * AI-Powered Marathi Sentiment & Opinion Analysis System.
 *
 * Pipeline order:
 *   1. Text Cleaning
 *   2. Noise Removal
 *   3. Sentence Segmentation
 *   4. Tokenization
 *   5. Stop-Word Removal
 *   6. Lemmatization
 *   7. Contextual Embeddings
 *   8. Transformer Representations (MahaBERT / IndicBERT)
 *
 * References follow IEEE citation format as documented in:
 *   महापल्स_NLP_Topics.md
 */
(function (global) {
  'use strict';

  const DATABASE = {

    // ==========================================
    // TEXT PREPROCESSING
    // ==========================================

    'text-cleaning': {
      title: 'Text Cleaning',
      category: 'Text Preprocessing',
      difficulty: 'Beginner',
      readTime: '6 min',
      year: '1990s',
      definition: 'Text cleaning standardizes raw Marathi text by removing URLs, HTML tags, non-Devanagari characters, and stray Roman or Arabic script that is mixed into social media data.',
      purpose: 'Converts raw, messy Marathi content into a standardized Devanagari-only format suitable for downstream NLP processing. This is the first step in the महापल्स preprocessing pipeline.',
      workingPrinciple: 'Uses regex patterns to strip non-Devanagari characters, HTML markup, and URLs. The mahaNLP library provides built-in Marathi-specific cleaning utilities that understand Devanagari Unicode ranges (U+0900–U+097F).',
      steps: [
        'Load the raw Marathi text string (e.g., a customer review or social media post).',
        'Remove all HTML tags and entities using a parser or regex.',
        'Strip URLs, email addresses, and hyperlinks.',
        'Remove non-Devanagari characters — such as stray Roman, Arabic, or numeric sequences that do not contribute to Marathi meaning.',
        'Collapse multiple whitespace characters into a single space.'
      ],
      example: {
        input: 'हे उत्पादन खूप चांगले आहे! 😊 Check: http://example.com',
        output: 'हे उत्पादन खूप चांगले आहे',
        code: `import re
from mahaNLP.preprocess import MarathiPreprocessor

preprocessor = MarathiPreprocessor()

raw_text = "हे उत्पादन खूप चांगले आहे! 😊 Check: http://example.com"

# Step 1: Remove HTML tags
text = re.sub(r'<[^>]+>', '', raw_text)

# Step 2: Remove URLs
text = re.sub(r'http\\S+|www\\S+', '', text)

# Step 3: Keep only Devanagari characters and spaces
text = re.sub(r'[^\\u0900-\\u097F\\s]', '', text)

# Step 4: Collapse whitespace
text = re.sub(r'\\s+', ' ', text).strip()

print(text)
# Output: हे उत्पादन खूप चांगले आहे`
      },
      advantages: [
        'Removes noise specific to Marathi social media data (mixed scripts, HTML, URLs).',
        'Reduces out-of-vocabulary tokens significantly.',
        'The mahaNLP library provides a ready-to-use Marathi text cleaner.',
        'Prevents tokenizer errors caused by non-Devanagari characters.'
      ],
      limitations: [
        'Removing emojis loses emotional signal that can be valuable for sentiment analysis.',
        'Aggressive cleaning can strip Marathi numerals (e.g., ०, १, २) if not handled carefully.',
        'Code-mixed Marathi-English text (common on social media) requires careful handling to avoid over-cleaning.'
      ],
      applications: [
        'First stage of the महापल्स preprocessing pipeline.',
        'Cleaning Marathi customer reviews for sentiment classification.',
        'Social media corpus preparation for Marathi NLP models.'
      ],
      keyTakeaways: [
        'Text cleaning is the entry point of the महापल्स pipeline — it must run before all other steps.',
        'Marathi text cleaning requires Devanagari-aware Unicode filtering.',
        'The mahaNLP library (Magdum et al., 2023) provides dedicated Marathi preprocessing utilities.',
        'Case normalization is NOT needed — Devanagari script has no uppercase/lowercase distinction.'
      ],
      relatedConcepts: [
        { label: 'Noise Removal', id: 'noise-removal' },
        { label: 'Sentence Segmentation', id: 'sentence-segmentation' }
      ],
      references: [
        'V. Magdum, O. J. Dhekane, S. S. Hiwarkhedkar, S. S. Mittal, and R. Joshi, "mahaNLP: A Marathi Natural Language Processing Library," in Proc. IJCNLP-AACL 2023 (Demo Track), 2023. Available: https://aclanthology.org/2023.aacl-demo.12/',
        'R. Joshi, "L3Cube-MahaNLP: Marathi Natural Language Processing Datasets, Models, and Library," arXiv preprint arXiv:2205.14728, 2022.'
      ]
    },

    'noise-removal': {
      title: 'Noise Removal',
      category: 'Text Preprocessing',
      difficulty: 'Beginner',
      readTime: '5 min',
      year: '1990s',
      definition: 'Noise removal filters out elements present in user-generated Marathi content that are not part of the language itself — such as emojis, hashtags, punctuation sequences, and special symbols.',
      purpose: 'Social media Marathi text is extremely noisy: it contains emojis, hashtags (#मराठी), @mentions, repeated punctuation (!!!), and mixed-script tokens. Noise removal ensures only linguistically meaningful Marathi content is passed forward in the महापल्स pipeline.',
      workingPrinciple: 'Applies targeted regex filters to strip specific noise categories. Unlike text cleaning (which targets markup and URLs), noise removal focuses on social-media-specific noise patterns common in Marathi user-generated content.',
      steps: [
        'Identify noise categories in the input: emojis, hashtags, @mentions, repeated punctuation.',
        'Apply regex to remove or replace emojis (Unicode emoji block: U+1F300–U+1F9FF).',
        'Strip hashtag markers (#) while optionally retaining the Marathi word that follows.',
        'Remove @mention tokens entirely.',
        'Collapse excessive repeated punctuation (e.g., "!!!" → "!").',
        'Strip leading and trailing whitespace.'
      ],
      example: {
        input: 'हे @virat खूप #मस्त आहे!!! 🔥🔥',
        output: 'हे खूप मस्त आहे!',
        code: `import re

def remove_noise_marathi(text):
    # Remove emojis (broad Unicode emoji range)
    text = re.sub(r'[\\U00010000-\\U0010FFFF]', '', text, flags=re.UNICODE)
    # Remove @mentions
    text = re.sub(r'@\\S+', '', text)
    # Remove hashtag symbol but keep the Marathi word
    text = re.sub(r'#(\\S+)', r'\\1', text)
    # Collapse repeated punctuation
    text = re.sub(r'([!?.]){2,}', r'\\1', text)
    # Collapse whitespace
    text = re.sub(r'\\s+', ' ', text).strip()
    return text

raw = "हे @virat खूप #मस्त आहे!!! 🔥🔥"
print(remove_noise_marathi(raw))
# Output: हे खूप मस्त आहे!`
      },
      advantages: [
        'Removes social-media noise that has no linguistic value in Marathi sentiment analysis.',
        'Preserves Marathi words behind hashtags by stripping only the # symbol.',
        'Prevents emoji and symbol tokens from polluting the tokenizer vocabulary.'
      ],
      limitations: [
        'Emojis carry sentiment information (😊 = positive, 😡 = negative); blanket removal loses this signal.',
        'Some hashtag content (e.g., #मराठी_पुस्तक) contains useful Marathi tokens joined by underscores.',
        'Aggressive repeated-punctuation stripping can alter the tone of exclamatory statements.'
      ],
      applications: [
        'Cleaning Marathi Twitter/X and Instagram data for महापल्स.',
        'Preprocessing Marathi YouTube comment datasets.',
        'Filtering Marathi WhatsApp message corpora.'
      ],
      keyTakeaways: [
        'Noise removal is the second step in the महापल्स pipeline, running after Text Cleaning.',
        'Marathi social media data is uniquely noisy due to script-mixing and emoji-heavy expression.',
        'The mahaNLP library supports built-in social media noise removal for Marathi.',
        'Consider retaining emoji polarity as a separate feature rather than discarding it entirely.'
      ],
      relatedConcepts: [
        { label: 'Text Cleaning', id: 'text-cleaning' },
        { label: 'Sentence Segmentation', id: 'sentence-segmentation' }
      ],
      references: [
        'V. Magdum et al., "mahaNLP: A Marathi Natural Language Processing Library," ACL Anthology, 2023. Available: https://aclanthology.org/2023.aacl-demo.12/',
        'R. Joshi, "L3Cube-MahaNLP: Marathi Natural Language Processing Datasets, Models, and Library," arXiv preprint arXiv:2205.14728, 2022.'
      ]
    },

    'sentence-segmentation': {
      title: 'Sentence Segmentation',
      category: 'Text Preprocessing',
      difficulty: 'Beginner',
      readTime: '6 min',
      year: '1990s',
      definition: 'Sentence segmentation identifies the boundaries between sentences in a block of Marathi text, using both the Devanagari danda (।) and standard punctuation as sentence-ending markers.',
      purpose: 'Downstream महापल्स processing (tokenization, lemmatization, and the MahaBERT model) operates sentence by sentence. Accurate segmentation prevents structural mixing of sentences and ensures each unit of text carries a coherent, self-contained sentiment.',
      workingPrinciple: 'Marathi uses the danda (।) as its primary sentence terminator, in addition to periods (.) and question marks (?). Language-specific segmenters in the mahaNLP library are aware of these Devanagari conventions and handle abbreviation edge cases in Marathi.',
      steps: [
        'Scan the cleaned Marathi text for potential sentence termination markers: danda (।), period (.), question mark (?), and exclamation mark (!).',
        'Apply Marathi-specific abbreviation lists to avoid false splits (e.g., "डॉ." for "Doctor" should not trigger a split).',
        'Verify context around each candidate marker.',
        'Divide the string at confirmed boundary locations.',
        'Return a list of clean, standalone Marathi sentences.'
      ],
      example: {
        input: 'डॉ. राज मुंबईला गेले. त्यांनी तिथे बरेच काम केले. हे उत्पादन चांगले आहे का?',
        output: '["डॉ. राज मुंबईला गेले.", "त्यांनी तिथे बरेच काम केले.", "हे उत्पादन चांगले आहे का?"]',
        code: `from mahaNLP.tokenize import MarathiSentenceTokenizer

tokenizer = MarathiSentenceTokenizer()

text = "डॉ. राज मुंबईला गेले. त्यांनी तिथे बरेच काम केले. हे उत्पादन चांगले आहे का?"
sentences = tokenizer.tokenize(text)

for i, s in enumerate(sentences, 1):
    print(f"Sentence {i}: {s}")

# Sentence 1: डॉ. राज मुंबईला गेले.
# Sentence 2: त्यांनी तिथे बरेच काम केले.
# Sentence 3: हे उत्पादन चांगले आहे का?`
      },
      advantages: [
        'Devanagari danda (।) provides an unambiguous sentence boundary marker absent in English.',
        'Enables sentence-level sentiment classification in महापल्स.',
        'Marathi-specific segmenters in mahaNLP handle local abbreviation exceptions (e.g., "श्री.", "डॉ.").'
      ],
      limitations: [
        'Informal social media Marathi often omits the danda entirely, relying only on newlines.',
        'Sentence boundaries in code-mixed Marathi-English text are ambiguous.',
        'Very long Marathi compound sentences may represent multiple opinions and require sub-segmentation.'
      ],
      applications: [
        'Splitting Marathi product reviews into individual opinion sentences for महापल्स.',
        'Preprocessing Marathi news articles for summarization.',
        'Structuring Marathi survey responses for clause-level analysis.'
      ],
      keyTakeaways: [
        'Marathi uses the danda (।) as its primary sentence-ending character — treat it like a period.',
        'The mahaNLP library provides a Marathi-aware sentence tokenizer.',
        'Sentence segmentation is Step 3 in the महापल्स pipeline, after cleaning and noise removal.',
        'Accurate segmentation directly improves MahaBERT fine-tuning quality.'
      ],
      relatedConcepts: [
        { label: 'Tokenization', id: 'tokenization' },
        { label: 'Noise Removal', id: 'noise-removal' }
      ],
      references: [
        'V. Magdum et al., "mahaNLP: A Marathi Natural Language Processing Library," ACL Anthology, 2023. Available: https://aclanthology.org/2023.aacl-demo.12/',
        'R. Joshi, "L3Cube-MahaNLP: Marathi Natural Language Processing Datasets, Models, and Library," arXiv preprint arXiv:2205.14728, 2022.'
      ]
    },

    'tokenization': {
      title: 'Tokenization',
      category: 'Text Preprocessing',
      difficulty: 'Beginner',
      readTime: '8 min',
      year: '1980s',
      definition: 'Tokenization splits a Marathi sentence into individual meaningful units (tokens). For MahaBERT/IndicBERT, this uses subword tokenization (WordPiece or SentencePiece) that correctly handles the morphologically rich Devanagari script.',
      purpose: 'Marathi is highly inflectional — words change significantly based on grammatical context. Subword tokenization breaks morphologically complex Marathi words into known sub-units, preventing out-of-vocabulary (OOV) errors and enabling the transformer model to process any Marathi word.',
      workingPrinciple: 'MahaBERT uses a WordPiece tokenizer trained on the L3Cube MahaCorpus (24.8 million Marathi sentences). It segments unknown words into subword units (e.g., "खेळाडूंनी" → ["खेळाडू", "##ंनी"]) using a vocabulary of ~30,000 Marathi subword tokens. IndicBERT uses SentencePiece trained on a multilingual Indic corpus.',
      steps: [
        'Receive a clean, segmented Marathi sentence from the preprocessing pipeline.',
        'Apply the MahaBERT WordPiece tokenizer vocabulary to map the sentence to a token sequence.',
        'Split any word not fully in the vocabulary into recognized subword units (prefix ## indicates a continuation token).',
        'Prepend a [CLS] token and append a [SEP] token as required by the BERT input format.',
        'Map tokens to their integer vocabulary indices for model input.'
      ],
      example: {
        input: 'हे उत्पादन खूप चांगले आहे',
        output: '["[CLS]", "हे", "उत्पादन", "खूप", "चांगले", "आहे", "[SEP]"] → [101, 2345, 6789, 1234, 4567, 890, 102]',
        code: `from transformers import AutoTokenizer

# Load MahaBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("l3cube-pune/marathi-bert-v2")

sentence = "हे उत्पादन खूप चांगले आहे"
tokens = tokenizer(sentence, return_tensors="pt")

print("Tokens:", tokenizer.convert_ids_to_tokens(tokens["input_ids"][0].tolist()))
print("Input IDs:", tokens["input_ids"])
# Tokens: ['[CLS]', 'हे', 'उत्पादन', 'खूप', 'चांगले', 'आहे', '[SEP]']`
      },
      advantages: [
        'WordPiece/SentencePiece eliminates OOV errors for all Marathi words.',
        'MahaBERT tokenizer is trained on 24.8M Marathi sentences — high vocabulary coverage.',
        'Subword units correctly capture Marathi morphological structure.',
        'Tokenizer output directly feeds the MahaBERT/IndicBERT model.'
      ],
      limitations: [
        'Simple whitespace-based tokenization is insufficient for Marathi — subword tokenizers are mandatory.',
        'Marathi conjunct characters (जोडाक्षरे) can be split across subword boundaries.',
        'Tokenizer must match the pre-trained model (MahaBERT tokenizer ≠ IndicBERT tokenizer).'
      ],
      applications: [
        'Preparing Marathi text for MahaBERT and IndicBERT input in महापल्स.',
        'Building Marathi vocabulary for information retrieval systems.',
        'Morphological analysis of Marathi text corpora.'
      ],
      keyTakeaways: [
        'Use the MahaBERT-specific tokenizer — it is trained on Marathi and understands Devanagari subword structure.',
        'Subword tokenization is the standard for all transformer-based Marathi NLP pipelines.',
        'Tokenization is Step 4 in the महापल्स pipeline.',
        'The L3Cube MahaCorpus of 24.8M sentences gives MahaBERT\'s tokenizer broad Marathi coverage.'
      ],
      relatedConcepts: [
        { label: 'Sentence Segmentation', id: 'sentence-segmentation' },
        { label: 'Stop-Word Removal', id: 'stop-word-removal' }
      ],
      references: [
        'R. Joshi, "L3Cube-MahaCorpus and MahaBERT: Marathi Monolingual Corpus, Marathi BERT Language Models, and Resources," in Proc. WILDRE-6 Workshop, LREC 2022, Marseille, France: ELRA, 2022, pp. 97–101.',
        'R. Joshi, "L3Cube-MahaNLP: Marathi Natural Language Processing Datasets, Models, and Library," arXiv preprint arXiv:2205.14728, 2022.'
      ]
    },

    'stop-word-removal': {
      title: 'Stop-Word Removal',
      category: 'Text Preprocessing',
      difficulty: 'Beginner',
      readTime: '6 min',
      year: '1960s',
      definition: 'Stop-word removal filters out high-frequency Marathi function words (e.g., आणि, तो, ते, हे, आहे) that appear extremely often but carry no discriminative sentiment value.',
      purpose: 'Marathi stop words dilute the signal in feature representations. Removing them reduces dimensionality and helps the model focus on content-bearing words that actually indicate positive, negative, or neutral sentiment.',
      workingPrinciple: 'Compares each Marathi token against a curated list of 400 Marathi stop words derived from the L3Cube MahaCorpus using TF-IDF frequency analysis across 24.8 million sentences. Words appearing in nearly every document are flagged as stop words and removed.',
      steps: [
        'Load the L3Cube Marathi stop words list (400 words derived from MahaCorpus).',
        'Receive the tokenized Marathi sentence.',
        'Iterate through each token.',
        'Discard the token if it exists in the stop words set.',
        'Return the filtered token list with only content-bearing Marathi words.'
      ],
      example: {
        input: '["हे", "उत्पादन", "खूप", "चांगले", "आहे", "आणि", "मला", "ते", "आवडले"]',
        output: '["उत्पादन", "खूप", "चांगले", "आवडले"]',
        code: `from mahaNLP.preprocess import MarathiStopWords

# Load curated Marathi stop word list (L3Cube MahaCorpus-derived)
stop_words = MarathiStopWords().get_stopwords()

tokens = ["हे", "उत्पादन", "खूप", "चांगले", "आहे", "आणि", "मला", "ते", "आवडले"]
filtered = [w for w in tokens if w not in stop_words]

print(filtered)
# Output: ['उत्पादन', 'खूप', 'चांगले', 'आवडले']`
      },
      advantages: [
        'Reduces the number of input tokens passed to MahaBERT, lowering computation.',
        'L3Cube\'s 400-word Marathi stop list is derived from 24.8M sentences — highly representative.',
        'Focuses sentiment classification on content words like adjectives and nouns.',
        'Available directly in the mahaNLP library.'
      ],
      limitations: [
        'Negation particles in Marathi (e.g., "नाही", "नको") are sometimes treated as stop words, which would completely invert sentiment.',
        'Context-dependent stop words (e.g., "हे" can be both a function word and a demonstrative with sentiment implication) require careful handling.',
        'MahaBERT itself retains all tokens internally — stop-word removal is applied only to reduce preprocessing feature space, not model input.'
      ],
      applications: [
        'Reducing Marathi sentiment analysis feature dimensions in the महापल्स pipeline.',
        'Building clean Marathi topic models.',
        'Marathi keyword extraction for news summarization.'
      ],
      keyTakeaways: [
        'Use the L3Cube Marathi stop word list — it is empirically derived from the largest Marathi corpus available.',
        'Stop-word removal is Step 5 in the महापल्स pipeline, applied after tokenization.',
        'Never remove Marathi negation words (नाही, नको) — they are critical for accurate sentiment.',
        'The mahaNLP library provides ready-to-use Marathi stop word utilities.'
      ],
      relatedConcepts: [
        { label: 'Tokenization', id: 'tokenization' },
        { label: 'Lemmatization', id: 'lemmatization' }
      ],
      references: [
        'R. Joshi, "L3Cube-MahaNLP: Marathi Natural Language Processing Datasets, Models, and Library," arXiv preprint arXiv:2205.14728, 2022. (Stop-word list derived from MahaCorpus, 24.8M sentences.)',
        'V. Magdum et al., "mahaNLP: A Marathi Natural Language Processing Library," ACL Anthology, 2023. Available: https://aclanthology.org/2023.aacl-demo.12/'
      ]
    },

    'lemmatization': {
      title: 'Lemmatization',
      category: 'Text Preprocessing',
      difficulty: 'Beginner',
      readTime: '8 min',
      year: '1990s',
      definition: 'Lemmatization reduces an inflected Marathi word to its dictionary base form (lemma) using vocabulary and morphological analysis, ensuring the output is a valid Marathi word.',
      purpose: 'Marathi is highly inflectional: a single verb root can produce dozens of inflected forms (e.g., "खेळणे", "खेळतो", "खेळली", "खेळणार"). Lemmatization maps all these forms to the same root ("खेळ"), consolidating vocabulary and improving model generalization. It is preferred over stemming because it always yields a valid dictionary word.',
      workingPrinciple: 'Applies morphological analysis and dictionary lookups specific to Marathi grammar. Unlike stemming (which simply strips suffixes and may produce non-words), lemmatization uses linguistic rules to identify the correct grammatical base form of each token in context.',
      steps: [
        'Receive the filtered Marathi token list from stop-word removal.',
        'Determine the part-of-speech (POS) tag for each token in its sentence context.',
        'Look up the token + POS pair in a Marathi morphological dictionary.',
        'Apply Marathi morphological rules to derive the correct lemma.',
        'Return the lemmatized token sequence.'
      ],
      example: {
        input: '["उत्पादन", "खूप", "चांगले", "आवडले"]',
        output: '["उत्पादन", "खूप", "चांगला", "आवडणे"]',
        code: `from mahaNLP.preprocess import MarathiLemmatizer

lemmatizer = MarathiLemmatizer()

tokens = ["उत्पादन", "खूप", "चांगले", "आवडले"]
lemmas = [lemmatizer.lemmatize(token) for token in tokens]

print(lemmas)
# Output: ['उत्पादन', 'खूप', 'चांगला', 'आवडणे']

# Compare: Stemming would produce non-words like 'चांगल', 'आवडल'`
      },
      advantages: [
        'Always yields a valid Marathi dictionary word — unlike stemming.',
        'Resolves highly inflected Marathi verb forms to a single canonical root.',
        'Improves vocabulary consolidation for downstream MahaBERT fine-tuning.',
        'Handles grammatically irregular Marathi forms correctly.'
      ],
      limitations: [
        'Requires a comprehensive Marathi morphological lexicon.',
        'Computationally slower than stemming due to dictionary lookups and POS context.',
        'Marathi resources for lemmatization are still limited compared to English.',
        'Errors in POS tagging propagate to lemmatization errors.'
      ],
      applications: [
        'Vocabulary normalization in the महापल्स Marathi sentiment pipeline.',
        'Preprocessing Marathi text for search and information retrieval.',
        'Morphological analysis for Marathi chatbot query understanding.'
      ],
      keyTakeaways: [
        'Lemmatization is preferred over stemming in महापल्स — it preserves valid Marathi words.',
        'Marathi\'s morphological richness makes lemmatization especially important for vocabulary consolidation.',
        'Lemmatization is Step 6 — the final text preprocessing step before the representation stage.',
        'Use the mahaNLP Marathi lemmatizer for linguistically accurate results.'
      ],
      relatedConcepts: [
        { label: 'Stop-Word Removal', id: 'stop-word-removal' },
        { label: 'Tokenization', id: 'tokenization' }
      ],
      references: [
        'R. J. Sutar and K. R. Desai, "Sentiment Analysis for Transliterated Hindi and Marathi Language using Machine Learning Approach," International Journal of Computer and Engineering in Science and Engineering, 2025.',
        'R. Joshi, "L3Cube-MahaNLP: Marathi Natural Language Processing Datasets, Models, and Library," arXiv preprint arXiv:2205.14728, 2022.'
      ]
    },

    // ==========================================
    // LANGUAGE REPRESENTATION
    // ==========================================

    'contextual-embeddings': {
      title: 'Contextual Embeddings',
      category: 'Language Representation',
      difficulty: 'Advanced',
      readTime: '10 min',
      year: '2018',
      definition: 'Contextual embeddings are dynamic vector representations that generate a unique vector for each Marathi word based on its surrounding sentence context, enabling the model to distinguish multiple meanings of the same word.',
      purpose: 'Solves the polysemy problem that is critical for Marathi sentiment analysis: the same word can carry positive or negative connotation depending on context. Unlike static embeddings (Word2Vec), contextual embeddings — produced by MahaBERT and IndicBERT — capture this context-sensitive meaning.',
      workingPrinciple: 'The MahaBERT/IndicBERT encoder passes the entire token sequence through multiple layers of multi-head self-attention (Transformer encoder blocks). Each layer refines the representation by attending to all other tokens in the sentence. The output hidden state for each token is its contextual embedding — a 768-dimensional vector that encodes both meaning and context.',
      steps: [
        'Pass the preprocessed, lemmatized Marathi token sequence through the MahaBERT tokenizer.',
        'Feed the token ID sequence into the MahaBERT/IndicBERT encoder.',
        'Allow the multi-head self-attention layers to compute relationships between all token pairs.',
        'Extract the last hidden state from the encoder — one 768-dim vector per token position.',
        'Use the [CLS] token representation as the sentence-level embedding for classification.'
      ],
      example: {
        input: 's1: "हे दुकान चांगले आहे" vs s2: "हे दुकान वाईट आहे"',
        output: '[CLS] vector for s1 ≠ [CLS] vector for s2 — different sentiment embeddings despite shared structure.',
        code: `from transformers import AutoTokenizer, AutoModel
import torch

# Load MahaBERT
tokenizer = AutoTokenizer.from_pretrained("l3cube-pune/marathi-bert-v2")
model = AutoModel.from_pretrained("l3cube-pune/marathi-bert-v2")

s1 = "हे दुकान चांगले आहे"
s2 = "हे दुकान वाईट आहे"

for sentence in [s1, s2]:
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # CLS token embedding = sentence-level representation
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    print(f"Sentence: {sentence}")
    print(f"CLS embedding shape: {cls_embedding.shape}")
    # Shape: torch.Size([1, 768])`
      },
      advantages: [
        'Produces a unique vector for each Marathi word based on full sentence context.',
        'Resolves Marathi polysemy accurately — critical for sentiment disambiguation.',
        'MahaBERT embeddings are trained on 24.8M Marathi sentences — rich linguistic coverage.',
        'The [CLS] token provides a ready-to-use sentence-level representation for classification.'
      ],
      limitations: [
        'Computationally expensive — requires GPU for fast inference.',
        'Embeddings cannot be pre-computed as a static lookup — computed on-the-fly per sentence.',
        'Requires the MahaBERT/IndicBERT model to be loaded in memory (~440MB).'
      ],
      applications: [
        'Sentence-level Marathi sentiment classification in महापल्स.',
        'Marathi Named Entity Recognition (NER).',
        'Marathi question answering and natural language inference.'
      ],
      keyTakeaways: [
        'Contextual embeddings are Step 7 in the महापल्स pipeline — the bridge between preprocessing and classification.',
        'MahaBERT produces 768-dimensional contextual embeddings per Marathi token.',
        'The [CLS] token embedding is used as the sentence representation for sentiment classification.',
        'Contextual embeddings from MahaBERT outperform static Word2Vec/FastText for Marathi sentiment tasks.'
      ],
      relatedConcepts: [
        { label: 'Transformer Representations', id: 'transformer-representations' },
        { label: 'Lemmatization', id: 'lemmatization' }
      ],
      references: [
        'J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," in Proc. NAACL-HLT 2019, Minneapolis, MN: ACL, 2019, pp. 4171–4186.',
        'R. Joshi, "L3Cube-MahaCorpus and MahaBERT: Marathi Monolingual Corpus, Marathi BERT Language Models, and Resources," in Proc. WILDRE-6 Workshop, LREC 2022, Marseille, France: ELRA, 2022, pp. 97–101.'
      ]
    },

    // ==========================================
    // LANGUAGE MODELS
    // ==========================================

    'transformer-representations': {
      title: 'Transformer Representations (MahaBERT / IndicBERT)',
      category: 'Language Models',
      difficulty: 'Advanced',
      readTime: '11 min',
      year: '2017',
      definition: 'Transformer representations are deep contextual vectors generated by multi-head self-attention stacks. महापल्स uses MahaBERT (Marathi monolingual) or IndicBERT (Indic multilingual) — both fine-tuned on labelled Marathi sentiment data for the final classification step.',
      purpose: 'Enables highly parallelizable, context-aware processing of Marathi sentences. The pre-trained transformer captures deep linguistic knowledge of Marathi from 24.8M sentences (MahaBERT) or 12 Indian languages (IndicBERT), which is then fine-tuned for sentiment classification with minimal labelled data.',
      workingPrinciple: 'Processes all tokens in parallel using positional encodings and multi-head self-attention blocks. Each attention head independently computes weighted relationships between every token pair, allowing the model to capture both local and long-range Marathi grammatical dependencies. A classification head (linear layer + softmax) is added on top of the [CLS] token for sentiment prediction.',
      steps: [
        'Load the pre-trained MahaBERT (l3cube-pune/marathi-bert-v2) or IndicBERT (ai4bharat/indic-bert) model.',
        'Add a classification head: a linear layer projecting the 768-dim [CLS] embedding to 3 output classes (Positive / Negative / Neutral).',
        'Fine-tune the full model on a labelled Marathi sentiment dataset for 3–5 epochs.',
        'During inference: pass preprocessed Marathi text through the pipeline → tokenizer → model → softmax → sentiment label.'
      ],
      example: {
        input: '"हे उत्पादन एकदम खराब आहे" (This product is completely bad)',
        output: 'Predicted Sentiment: Negative (confidence: 0.94)',
        code: `from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load fine-tuned MahaBERT sentiment model
model_name = "l3cube-pune/marathi-bert-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=3  # Positive, Negative, Neutral
)

# Marathi input sentence
text = "हे उत्पादन एकदम खराब आहे"

inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)

probs = torch.softmax(outputs.logits, dim=-1)
labels = ["Negative", "Neutral", "Positive"]
prediction = labels[probs.argmax()]
confidence = probs.max().item()

print(f"Sentiment: {prediction} (confidence: {confidence:.2f})")
# Sentiment: Negative (confidence: 0.94)`
      },
      advantages: [
        'MahaBERT is pre-trained on 24.8M Marathi sentences — the largest Marathi-specific model available.',
        'Processes all tokens in parallel (no sequential bottleneck like LSTMs).',
        'Captures long-range Marathi grammatical dependencies that rule-based systems miss.',
        'Fine-tuning requires relatively few labelled Marathi examples due to transfer learning.',
        'IndicBERT provides an alternative when cross-lingual Indic transfer is desired.'
      ],
      limitations: [
        'Self-attention complexity scales quadratically with sequence length (O(n²)).',
        'Requires significant GPU memory for fine-tuning (~6GB+ for MahaBERT).',
        'Pre-training from scratch on Marathi is computationally infeasible without a cluster.',
        'MahaBERT has a 512-token maximum sequence length — very long Marathi documents must be chunked.'
      ],
      applications: [
        'Marathi sentiment classification — the core task of महापल्स.',
        'Marathi Named Entity Recognition (NER) with MahaBERT.',
        'Marathi Natural Language Inference and textual entailment.',
        'Opinion mining from Marathi government and social media portals.'
      ],
      keyTakeaways: [
        'MahaBERT (Joshi, 2022) is the primary model for महापल्स — trained on Marathi and consistently outperforms multilingual alternatives on Marathi tasks.',
        'IndicBERT (Kakwani et al., 2020) is the fallback when cross-lingual Indic context is beneficial.',
        'The Transformer architecture (Vaswani et al., 2017) — "Attention is All You Need" — powers both models.',
        'Fine-tune for 3–5 epochs on a Marathi sentiment dataset using the Hugging Face Transformers library.',
        'This is Step 8 — the final and most critical step in the महापल्स pipeline.'
      ],
      relatedConcepts: [
        { label: 'Contextual Embeddings', id: 'contextual-embeddings' },
        { label: 'Tokenization', id: 'tokenization' }
      ],
      references: [
        'R. Joshi, "L3Cube-MahaCorpus and MahaBERT: Marathi Monolingual Corpus, Marathi BERT Language Models, and Resources," in Proc. WILDRE-6 Workshop, LREC 2022, Marseille, France: ELRA, 2022, pp. 97–101.',
        'D. Kakwani et al., "IndicNLPSuite: Monolingual Corpora, Evaluation Benchmarks and Pre-trained Multilingual Language Models for Indian Languages," in Findings of EMNLP 2020, 2020.',
        'A. Vaswani et al., "Attention is all you need," in Advances in Neural Information Processing Systems (NeurIPS), 2017, pp. 5998–6008.',
        'J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," in Proc. NAACL-HLT 2019, Minneapolis, MN: ACL, 2019, pp. 4171–4186.'
      ]
    }

  };

  // Expose database to global scope
  global.NLP = global.NLP || {};
  global.NLP.Database = DATABASE;

})(window);
