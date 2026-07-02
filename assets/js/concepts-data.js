/**
 * NLP Hub — Concepts Database
 * ─────────────────────────────────────────────────────────────
 * Contains complete content, examples, code snippets, advantages,
 * limitations, and applications for all 21 NLP concepts.
 */
(function (global) {
  'use strict';

  const DATABASE = {
    // ==========================================
    // TEXT PREPROCESSING
    // ==========================================
    'sentence-segmentation': {
      title: 'Sentence Segmentation',
      category: 'Text Preprocessing',
      difficulty: 'Beginner',
      readTime: '6 min',
      year: '1990s',
      definition: 'Sentence segmentation (or sentence boundary disambiguation) is the process of dividing a continuous block of text into its constituent sentences.',
      purpose: 'Many downstream NLP tasks (like parsing, POS tagging, and translation) process text sentence by sentence. Accurately finding sentence boundaries prevents structural mixing and boundary overlap errors.',
      workingPrinciple: 'Rule-based systems use regular expressions and lists of abbreviations to handle punctuation. Machine learning systems (like Punkt in NLTK or neural model-based splitters in spaCy) treat boundary detection as a binary classification problem (whether a punctuation mark is a boundary or not) using surrounding token context.',
      steps: [
        'Scan the text for potential sentence termination marks (e.g., period, question mark, exclamation point).',
        'Analyze the surrounding context of each candidate mark (e.g., check if the preceding word is a known abbreviation like "Dr." or "U.S.").',
        'Verify casing of the succeeding word (capitalized words usually indicate a new sentence).',
        'Divide the string at confirmed boundary locations.'
      ],
      example: {
        input: 'Dr. Jones arrived at 8 p.m. at the U.S. embassy. Did he meet the diplomat?',
        output: [
          'Dr. Jones arrived at 8 p.m. at the U.S. embassy.',
          'Did he meet the diplomat?'
        ],
        code: `import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

text = "Dr. Jones arrived at 8 p.m. at the U.S. embassy. Did he meet the diplomat?"
sentences = sent_tokenize(text)
print(sentences)
# Output: ['Dr. Jones arrived at 8 p.m. at the U.S. embassy.', 'Did he meet the diplomat?']`
      },
      advantages: [
        'Simplifies downstream sentence-level processing.',
        'Extremely fast execution with modern heuristic rule-sets.',
        'High accuracy (>98%) on standard formal writing (e.g., news).'
      ],
      limitations: [
        'Fails on informal text lacking standard punctuation (e.g., tweets).',
        'Ambiguities with abbreviation periods (e.g., "vs.", "Inc.") require large exception lists.',
        'Multilingual differences (e.g., Spanish inverted punctuation) require language-specific rules.'
      ],
      applications: [
        'Machine Translation preprocessing.',
        'Document Summarization pipelines.',
        'Text-to-Speech (TTS) natural pause insertion.'
      ],
      keyTakeaways: [
        'Sentence segmentation is not just splitting by periods.',
        'Abbreviation exceptions and casing checks are vital for disambiguation.',
        'Use machine-learning based segmenters (like NLTK Punkt) over simple regex splitters.'
      ],
      relatedConcepts: [
        { label: 'Tokenization', id: 'tokenization' },
        { label: 'Noise Removal', id: 'noise-removal' }
      ],
      references: [
        'Palmer, D. D. (2000). Tokenisation and sentence segmentation. Handbook of Natural Language Processing.',
        'Kiss, T., & Strunk, J. (2006). Unsupervised multilingual sentence boundary detection. Computational Linguistics.'
      ]
    },

    'tokenization': {
      title: 'Tokenization',
      category: 'Text Preprocessing',
      difficulty: 'Beginner',
      readTime: '8 min',
      year: '1980s',
      definition: 'Tokenization is the process of breaking a sequence of strings (text) into smaller pieces called tokens (words, subwords, or characters).',
      purpose: 'Computers cannot process raw blocks of text. Tokenization provides the discrete linguistic units that form the vocabulary of the model.',
      workingPrinciple: 'Basic tokenizers split by whitespace and isolate punctuation. Modern subword tokenizers (like Byte-Pair Encoding or WordPiece) dynamically split rare words into subword units (e.g., "tokenization" -> ["token", "##ization"]) using statistical frequencies to prevent out-of-vocabulary errors.',
      steps: [
        'Convert string into a sequence of characters.',
        'Apply rules or statistical boundaries (whitespace, punctuation, subword vocabulary matches).',
        'Group contiguous characters into token spans.',
        'Normalize tokens (e.g., separate clitics like "don\'t" -> ["do", "n\'t"]).'
      ],
      example: {
        input: 'We are learning tokenization.',
        output: "['We', 'are', 'learning', 'tokenization', '.']",
        code: `import spacy
nlp = spacy.load("en_core_web_sm")

doc = nlp("We are learning tokenization.")
tokens = [token.text for token in doc]
print(tokens)
# Output: ['We', 'are', 'learning', 'tokenization', '.']`
      },
      advantages: [
        'Forms the literal foundation of all NLP text representations.',
        'Subword tokenization eliminates the out-of-vocabulary (OOV) problem.',
        'Maintains punctuation marks as separate tokens for syntactic meaning.'
      ],
      limitations: [
        'Agglutinative languages (e.g., Turkish) or languages without word spaces (e.g., Chinese) require advanced tokenizers.',
        'Splitting compound words (e.g., "ice cream") can lose semantic meaning if not handled properly.',
        'Different tokenization rules yield different vocabulary index sizes, causing model incompatibilities.'
      ],
      applications: [
        'Lexical analysis of compiler strings.',
        'Information retrieval index building.',
        'Input preparation for language models.'
      ],
      keyTakeaways: [
        'Tokens are the fundamental atoms of NLP pipelines.',
        'Punctuation should usually be treated as distinct tokens.',
        'Subword tokenization is the default standard for deep learning/LLMs.'
      ],
      relatedConcepts: [
        { label: 'Sentence Segmentation', id: 'sentence-segmentation' },
        { label: 'Stop-word Removal', id: 'stop-word-removal' }
      ],
      references: [
        'Webster, J. J., & Kit, C. (1992). Tokenization as the initial phase in NLP. COLING.',
        'Sennrich, R., Haddow, B., & Birch, A. (2016). Neural machine translation of rare words with subword units. ACL.'
      ]
    },

    'case-normalization': {
      title: 'Case Normalization',
      category: 'Text Preprocessing',
      difficulty: 'Beginner',
      readTime: '4 min',
      year: '1970s',
      definition: 'Case normalization is the process of converting all characters in a text to a uniform case (usually lowercase).',
      purpose: 'Reduces the vocabulary size by mapping capitalized and lowercase variations of the same word (e.g., "NLP", "Nlp", and "nlp") to a single token.',
      workingPrinciple: 'Iterates through the characters of each token and replaces uppercase character codes with their lowercase equivalents using basic ASCII or Unicode maps.',
      steps: [
        'Read each token in the corpus.',
        'Convert each uppercase character to lowercase.',
        'Return the standardized token stream.'
      ],
      example: {
        input: 'Natural Language Processing is AWESOME.',
        output: 'natural language processing is awesome.',
        code: `text = "Natural Language Processing is AWESOME."
normalized = text.lower()
print(normalized)
# Output: natural language processing is awesome.`
      },
      advantages: [
        'Drastically reduces vocabulary size and index sparsity.',
        'Improves match rates in keyword search systems.',
        'Simple to implement with negligible computational overhead.'
      ],
      limitations: [
        'Loss of semantic distinction for proper nouns vs. common nouns (e.g., "Bush" the president vs. "bush" the plant).',
        'Alters acronym meanings (e.g., "US" the country vs. "us" the pronoun).',
        'Can disrupt named entity recognition (NER) performance.'
      ],
      applications: [
        'Search engine indexing systems.',
        'Topic modeling preparation.',
        'Bag of Words vocabulary construction.'
      ],
      keyTakeaways: [
        'Standard practice for classical algorithms (TF-IDF, BoW).',
        'Avoid applying case normalization blindly when building NER or POS taggers.',
        'Usually done immediately after tokenization.'
      ],
      relatedConcepts: [
        { label: 'Tokenization', id: 'tokenization' },
        { label: 'Noise Removal', id: 'noise-removal' }
      ],
      references: [
        'Manning, C. D., Raghavan, P., & Schütze, H. (2008). Introduction to Information Retrieval. Cambridge University Press.'
      ]
    },

    'stop-word-removal': {
      title: 'Stop-word Removal',
      category: 'Text Preprocessing',
      difficulty: 'Beginner',
      readTime: '6 min',
      year: '1960s',
      definition: 'Stop-word removal is the process of filtering out high-frequency words that carry little semantic information about the primary topic of the document.',
      purpose: 'Eliminates words like "the", "is", and "and" to reduce computational cost and focus feature extraction on content-carrying words.',
      workingPrinciple: 'Compares each token in a text stream against a predefined dictionary or set of stop words (e.g., from NLTK, spaCy, or scikit-learn). Any token matching a stop word is discarded.',
      steps: [
        'Load a standard stop words list for the target language.',
        'Tokenize the text.',
        'Iterate through the tokens.',
        'Discard tokens that exist in the stop words set.'
      ],
      example: {
        input: 'This is a sample sentence showing stop words.',
        output: "['sample', 'sentence', 'showing', 'stop', 'words']",
        code: `from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
tokens = ["this", "is", "a", "sample", "sentence", "showing", "stop", "words"]
filtered = [w for w in tokens if w not in stop_words]
print(filtered)
# Output: ['sample', 'sentence', 'showing', 'stop', 'words']`
      },
      advantages: [
        'Reduces search index size and dimensionality.',
        'Speeds up classical algorithms (BoW, TF-IDF, Topic Modeling).',
        'Filters out noise, improving classification signal.'
      ],
      limitations: [
        'Can alter grammatical semantics completely (e.g., "to be or not to be" becomes empty).',
        'Hurts tasks like Sentiment Analysis where negation words ("not", "never") are crucial.',
        'Must never be used with models that rely on sequence structure (e.g., LSTMs, Transformers).'
      ],
      applications: [
        'Text Classification feature selection.',
        'Search Engine Query processing.',
        'Topic Modeling (LDA).'
      ],
      keyTakeaways: [
        'Stop words lists are application and domain-specific.',
        'Negations and pronouns shouldn\'t be removed in dialogue or sentiment tasks.',
        'Modern transformer models keep stop words intact.'
      ],
      relatedConcepts: [
        { label: 'Tokenization', id: 'tokenization' },
        { label: 'TF-IDF', id: 'tf-idf' }
      ],
      references: [
        'Luhn, H. P. (1957). A statistical approach to mechanized encoding and searching of literary information. IBM Journal of Research and Development.'
      ]
    },

    'stemming': {
      title: 'Stemming',
      category: 'Text Preprocessing',
      difficulty: 'Beginner',
      readTime: '7 min',
      year: '1980',
      definition: 'Stemming is a crude heuristic process that chops off the ends of words to reduce them to their base form (stem).',
      purpose: 'Maps morphological variants of a word (e.g., "connected", "connection", "connections") to the same base string ("connect") to consolidate vocabularies.',
      workingPrinciple: 'Applies regular string replacement rules (e.g., the Porter Stemmer rule "sses" -> "ss", "ies" -> "i", "ational" -> "ate") in sequence, without any understanding of linguistic context or parts of speech.',
      steps: [
        'Receive a token string.',
        'Apply rule-based suffix truncation loops (e.g., Porter, Lancaster, or Snowball algorithms).',
        'Output the remaining character stem.'
      ],
      example: {
        input: 'The programmers were programming a new program.',
        output: "['the', 'program', 'were', 'program', 'a', 'new', 'program']",
        code: `from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

words = ["programmers", "programming", "program"]
stems = [stemmer.stem(w) for w in words]
print(stems)
# Output: ['program', 'program', 'program']`
      },
      advantages: [
        'Extremely fast and computationally lightweight.',
        'Drastically reduces dimensionality of sparse vectors.',
        'Easy to implement in any language.'
      ],
      limitations: [
        'Overstemming: Chops too much, conflating distinct words (e.g., "organization" and "organ" both become "organ").',
        'Understemming: Fails to merge related words (e.g., "alumnae" -> "alumnae", "alumnus" -> "alumnus").',
        'Stems are often not real dictionary words (e.g., "laziness" -> "lazi").'
      ],
      applications: [
        'Information retrieval (search engines) index building.',
        'Simple document clustering.',
        'Vocabulary size compression.'
      ],
      keyTakeaways: [
        'Stemming is algorithmic suffix stripping.',
        'Stems do not need to be valid words.',
        'Choose Lemmatization when grammatical and lexical correctness matters.'
      ],
      relatedConcepts: [
        { label: 'Lemmatization', id: 'lemmatization' },
        { label: 'Case Normalization', id: 'case-normalization' }
      ],
      references: [
        'Porter, M. F. (1980). An algorithm for suffix stripping. Program.'
      ]
    },

    'lemmatization': {
      title: 'Lemmatization',
      category: 'Text Preprocessing',
      difficulty: 'Beginner',
      readTime: '8 min',
      year: '1990s',
      definition: 'Lemmatization is the process of resolving a word to its dictionary base form (lemma) using vocabulary and morphological analysis.',
      purpose: 'Normalizes inflected forms of words to their correct grammatical base (e.g., "better" -> "good", "was" -> "be") while ensuring the lemma remains a valid dictionary word.',
      workingPrinciple: 'Uses a lexicon (like WordNet) and analyzes the part-of-speech (POS) tag of the word in context. It applies complex linguistic rules and lookups to retrieve the correct base form.',
      steps: [
        'Identify the word token.',
        'Determine or estimate the Part of Speech (POS) tag of the token in its sentence context.',
        'Look up the token-POS pair in a dictionary database.',
        'Apply morphological rules to yield the standard lemma.'
      ],
      example: {
        input: 'The children were playing with the dogs.',
        output: "['the', 'child', 'be', 'play', 'with', 'the', 'dog']",
        code: `import spacy
nlp = spacy.load("en_core_web_sm")

doc = nlp("The children were playing with the dogs.")
lemmas = [token.lemma_ for token in doc]
print(lemmas)
# Output: ['the', 'child', 'be', 'play', 'with', 'the', 'dog']`
      },
      advantages: [
        'Linguistically accurate and yields valid dictionary words.',
        'Resolves irregular forms (e.g., "went" -> "go", "mice" -> "mouse").',
        'Maintains semantic consistency better than stemming.'
      ],
      limitations: [
        'Slower than stemming due to dictionary lookups and POS tag context parsing.',
        'Requires large lexical database resources.',
        'Lacks support in languages with small resources.'
      ],
      applications: [
        'Natural Language Understanding (NLU) parsing.',
        'Sentiment analysis and opinion mining.',
        'Chatbot query understanding pipelines.'
      ],
      keyTakeaways: [
        'Lemmatization uses lexical and context POS analysis.',
        'Always yields a valid grammatical word.',
        'Use spaCy for state-of-the-art lemmatization.'
      ],
      relatedConcepts: [
        { label: 'Stemming', id: 'stemming' },
        { label: 'Tokenization', id: 'tokenization' }
      ],
      references: [
        'Plisson, J., Mladenić, D., & Grobelnik, M. (2004). A rule-based approach to lemmatization. IS-2004.'
      ]
    },

    'noise-removal': {
      title: 'Noise Removal',
      category: 'Text Preprocessing',
      difficulty: 'Beginner',
      readTime: '5 min',
      year: '1990s',
      definition: 'Noise removal is the process of removing unwanted or irrelevant formatting, markup, and non-textual characters from a dataset.',
      purpose: 'Cleans raw text retrieved from web scrapes or files (e.g., removing HTML tags, XML elements, system logs, or emojis) to focus solely on natural language.',
      workingPrinciple: 'Applies string parsers (like BeautifulSoup) and regex filters to identify and strip markup templates, control characters, and structural metadata.',
      steps: [
        'Load the raw textual document.',
        'Parse structure using HTML/XML libraries.',
        'Apply regex patterns to remove remnants (URLs, phone numbers, special characters).',
        'Standardize line breaks and spaces.'
      ],
      example: {
        input: '<div>NLP is cool! &amp; fun. <a href="#">Link</a></div>',
        output: 'NLP is cool! & fun.',
        code: `from bs4 import BeautifulSoup

html_text = "<div>NLP is cool! &amp; fun. <a href='#'>Link</a></div>"
soup = BeautifulSoup(html_text, "html.parser")
cleaned = soup.get_text()
print(cleaned)
# Output: NLP is cool! & fun. Link`
      },
      advantages: [
        'Prevents parser errors in tokenization.',
        'Eliminates duplicate characters and markup noise.',
        'Improves model generalize ability.'
      ],
      limitations: [
        'Can accidentally strip essential markers (e.g., mathematical formulas inside brackets).',
        'Complex markup takes time to parse on large datasets.',
        'Information loss if emojis or URLs carried sentiment.'
      ],
      applications: [
        'Web scraping pipelines.',
        'PDF/Word document text extraction.',
        'Social media corpus cleaning.'
      ],
      keyTakeaways: [
        'Noise removal is performed before tokenization.',
        'HTML tags and script blocks must be stripped.',
        'Ensure text encoding (Unicode UTF-8) is standardized during noise removal.'
      ],
      relatedConcepts: [
        { label: 'Text Cleaning', id: 'text-cleaning' },
        { label: 'Sentence Segmentation', id: 'sentence-segmentation' }
      ],
      references: [
        'Grefenstette, G. (1999). Text cleaning and noise reduction. Text Mining and its Applications.'
      ]
    },

    'text-cleaning': {
      title: 'Text Cleaning',
      category: 'Text Preprocessing',
      difficulty: 'Beginner',
      readTime: '6 min',
      year: '1990s',
      definition: 'Text cleaning standardizes text representations by correcting typos, expanding contractions, removing emojis, and stripping non-alphanumeric symbols.',
      purpose: 'Converts raw, messy conversational text into a standardized spelling and grammatical format suitable for model training.',
      workingPrinciple: 'Uses regex patterns to strip non-alphanumeric symbols and look-up tables to expand contractions (e.g., "don\'t" -> "do not") and standard abbreviations.',
      steps: [
        'Load the text string.',
        'Expand standard contractions (e.g., mapping "can\'t" to "cannot").',
        'Remove or translate non-text characters (e.g., stripping emojis or replacing them with keywords like ":smile:").',
        'Format multiple whitespace tabs and newlines into single spaces.'
      ],
      example: {
        input: "I can't wait!!! 🚀 NLP is amazing.",
        output: 'I cannot wait NLP is amazing.',
        code: `import re

def clean_text(text):
    # Expand contraction placeholder
    text = text.replace("can't", "cannot")
    # Remove emojis and non-alphanumeric chars
    text = re.sub(r'[^a-zA-Z0-9\\s.]', '', text)
    # Collapse whitespace
    text = re.sub(r'\\s+', ' ', text).strip()
    return text

raw = "I can't wait!!! 🚀 NLP is amazing."
print(clean_text(raw))
# Output: I cannot wait NLP is amazing.`
      },
      advantages: [
        'Improves model consistency by mapping spelling variants.',
        'Reduces out-of-vocabulary terms.',
        'Removes structural spam.'
      ],
      limitations: [
        'Information loss (e.g., "!!!" and emojis carry rich sentiment).',
        'Slang mapping can feel unnatural or become outdated quickly.',
        'Regex replacements can be error-prone without thorough testing.'
      ],
      applications: [
        'Social media Sentiment Analysis.',
        'Acoustic transcribe standardization.',
        'Input cleaning for chatbot pipelines.'
      ],
      keyTakeaways: [
        'Text cleaning focuses on standardization of natural letters.',
        'Contraction expansion helps resolve pronoun/verb combinations.',
        'Avoid stripping punctuation completely if sentence parsing is required next.'
      ],
      relatedConcepts: [
        { label: 'Noise Removal', id: 'noise-removal' },
        { label: 'Case Normalization', id: 'case-normalization' }
      ],
      references: [
        'Bird, S., Klein, E., & Loper, E. (2009). Natural Language Processing with Python. O\'Reilly Media.'
      ]
    },

    // ==========================================
    // FEATURE ENGINEERING
    // ==========================================
    'bag-of-words': {
      title: 'Bag of Words (BoW)',
      category: 'Feature Engineering',
      difficulty: 'Beginner',
      readTime: '8 min',
      year: '1954',
      definition: 'Bag of Words is a simplifying representation used in NLP where a text is represented as the bag (multiset) of its words, disregarding grammar and word order.',
      purpose: 'Converts unstructured text documents into fixed-length numeric vectors that can be processed by machine learning algorithms.',
      workingPrinciple: 'Constructs a vocabulary of all unique words in the corpus. For each document, it counts how many times each vocabulary word appears, creating a frequency vector.',
      steps: [
        'Collect all documents in the corpus.',
        'Extract all unique words to build a vocabulary index.',
        'For each document, count the frequency of each vocabulary word.',
        'Represent the document as a vector where index elements match word counts.'
      ],
      example: {
        input: [
          'd1: NLP is fun',
          'd2: NLP is fast and NLP is fun'
        ],
        output: 'Vocabulary: {"NLP":0, "is":1, "fun":2, "fast":3, "and":4}\nVectors:\nd1: [1, 1, 1, 0, 0]\nd2: [2, 2, 1, 1, 1]',
        code: `from sklearn.feature_extraction.text import CountVectorizer

corpus = ["NLP is fun", "NLP is fast and NLP is fun"]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print("Vocabulary:", vectorizer.vocabulary_)
print("Vectors:\\n", X.toarray())`
      },
      advantages: [
        'Very simple to understand and implement.',
        'Works well for basic document classification tasks (e.g., spam detection).',
        'No training step required for the representation model itself.'
      ],
      limitations: [
        'Ignores word order entirely ("not bad" and "bad not" have identical vectors).',
        'Suffers from extreme sparsity (mostly zeros) as vocabulary grows.',
        'Gives high weight to highly frequent but less informative words.'
      ],
      applications: [
        'Spam email detection filter.',
        'Basic sentiment classifiers.',
        'Initial baseline for document classification.'
      ],
      keyTakeaways: [
        'BoW ignores word sequence and context.',
        'Resulting matrices are highly sparse.',
        'Use CountVectorizer in scikit-learn for easy implementation.'
      ],
      relatedConcepts: [
        { label: 'TF-IDF', id: 'tf-idf' },
        { label: 'One-Hot Encoding', id: 'one-hot-encoding' }
      ],
      references: [
        'Harris, Z. S. (1954). Distributional structure. Word.'
      ]
    },

    'n-grams': {
      title: 'N-Grams',
      category: 'Feature Engineering',
      difficulty: 'Beginner',
      readTime: '7 min',
      year: '1948',
      definition: 'An N-gram is a contiguous sequence of N items (characters or words) from a given sample of text.',
      purpose: 'Captures local context and word order in statistical models, improving upon the bag-of-words assumption.',
      workingPrinciple: 'Slides a window of size N across the tokenized sequence, extracting all contiguous sub-sequences of length N (e.g., N=1 is Unigrams, N=2 is Bigrams, N=3 is Trigrams).',
      steps: [
        'Tokenize the text into a sequence of words.',
        'Define the window size N.',
        'Slide the window token-by-token from start to end.',
        'Collect all token sub-sequences captured in the window.'
      ],
      example: {
        input: 'NLP is fun',
        output: 'Bigrams (N=2): [("NLP", "is"), ("is", "fun")]',
        code: `from nltk.util import ngrams

text = "NLP is fun".split()
bigrams = list(ngrams(text, 2))
print(bigrams)
# Output: [('NLP', 'is'), ('is', 'fun')]`
      },
      advantages: [
        'Captures short-range word dependencies and ordering.',
        'Helps disambiguate phrases (e.g., "not good" vs. "good").',
        'Can be applied at character level to handle spelling variations.'
      ],
      limitations: [
        'Vocabulary size grows exponentially with N (curse of dimensionality).',
        'Leads to highly sparse vectors.',
        'Cannot capture long-distance dependencies beyond window size N.'
      ],
      applications: [
        'Autocomplete text suggestion.',
        'Spell checking and corrections.',
        'Language identification systems.'
      ],
      keyTakeaways: [
        'N-grams preserve local word ordering context.',
        'Bigrams and Trigrams are the most common variants.',
        'Character N-grams are great for typo-resilient search.'
      ],
      relatedConcepts: [
        { label: 'Bag of Words', id: 'bag-of-words' },
        { label: 'Statistical Language Models', id: 'statistical-language-models' }
      ],
      references: [
        'Shannon, C. E. (1948). A mathematical theory of communication. Bell System Technical Journal.'
      ]
    },

    'tf': {
      title: 'Term Frequency (TF)',
      category: 'Feature Engineering',
      difficulty: 'Beginner',
      readTime: '6 min',
      year: '1957',
      definition: 'Term Frequency measures how frequently a term (word) occurs in a document.',
      purpose: 'Evaluates the importance of a word within a single document. Words that occur more often are assumed to be more descriptive of that document\'s content.',
      workingPrinciple: 'Calculates the frequency of a term. Common mathematical formulations include raw count, term frequency relative to document length, or logarithmic scaling to prevent bias from long documents.',
      steps: [
        'Tokenize the document.',
        'Count the occurrences of target term t in document d.',
        'Divide by the total count of all terms in document d (or apply log scaling).'
      ],
      example: {
        input: 'Term "NLP" in doc "NLP is fun and NLP is fast" (len: 8)',
        output: 'Raw TF: 2, Normalized TF: 2/8 = 0.25',
        code: `doc = "NLP is fun and NLP is fast".split()
tf_raw = doc.count("NLP")
tf_norm = tf_raw / len(doc)
print(f"Raw: {tf_raw}, Norm: {tf_norm}")
# Output: Raw: 2, Norm: 0.25`
      },
      advantages: [
        'Linguistically intuitive (frequent words represent content).',
        'Extremely easy to calculate.',
        'Scales well across large corpora.'
      ],
      limitations: [
        'Biased towards long documents if not normalized.',
        'Common grammatical words (stop words) dominate raw TF scores.',
        'Treats all words in vocabulary as equally important regardless of rarity.'
      ],
      applications: [
        'Document search relevance scoring.',
        'Basic search indexes.',
        'Component of TF-IDF vectors.'
      ],
      keyTakeaways: [
        'TF tracks word density within a document.',
        'Must be normalized to adjust for document length.',
        'Useless on its own without stop word removal or IDF weighting.'
      ],
      relatedConcepts: [
        { label: 'IDF', id: 'idf' },
        { label: 'TF-IDF', id: 'tf-idf' }
      ],
      references: [
        'Luhn, H. P. (1957). A statistical approach to mechanized encoding and searching of literary information. IBM Journal.'
      ]
    },

    'idf': {
      title: 'Inverse Document Frequency (IDF)',
      category: 'Feature Engineering',
      difficulty: 'Intermediate',
      readTime: '6 min',
      year: '1972',
      definition: 'Inverse Document Frequency measures how important or unique a word is across the entire corpus.',
      purpose: 'Dampens the weights of common words (e.g., "the", "and") while boosting the weights of rare, highly specific words (e.g., "transformer", "lemma").',
      workingPrinciple: 'Calculates the logarithm of the total number of documents divided by the number of documents containing the term. Adding 1 prevents division-by-zero errors.',
      steps: [
        'Count the total number of documents N in the corpus.',
        'Count the number of documents df(t) containing term t.',
        'Compute IDF = log( N / (1 + df(t)) ) + 1.'
      ],
      example: {
        input: 'N = 1000 docs, df("the") = 1000 docs, df("attention") = 10 docs',
        output: 'IDF("the") = log(1000/1000) = 0. IDF("attention") = log(1000/10) = 4.6',
        code: `import math

N = 1000
df_the = 1000
df_att = 10

idf_the = math.log10(N / df_the)
idf_att = math.log10(N / df_att)
print(f"IDF('the'): {idf_the}, IDF('attention'): {idf_att}")
# Output: IDF('the'): 0.0, IDF('attention'): 2.0`
      },
      advantages: [
        'Effectively filters out corpus-wide noise without a fixed stop words list.',
        'Dynamically learns domain-specific term significance.',
        'Logarithmic scaling prevents extreme values from dominating.'
      ],
      limitations: [
        'Requires checking the entire corpus to calculate, making real-time additions slow.',
        'Treats out-of-vocabulary words as errors or requires heuristic default values.',
        'Does not account for term distribution within a single document.'
      ],
      applications: [
        'Search engine query term weighting.',
        'Text summarization sentence selection.',
        'Keyphrase extraction algorithms.'
      ],
      keyTakeaways: [
        'IDF measures the rarity of a term across the corpus.',
        'Common terms have an IDF near zero.',
        'Requires global corpus statistics.'
      ],
      relatedConcepts: [
        { label: 'TF', id: 'tf' },
        { label: 'TF-IDF', id: 'tf-idf' }
      ],
      references: [
        'Spärck Jones, K. (1972). A statistical interpretation of term specificity and its retrieval. Journal of Documentation.'
      ]
    },

    'tf-idf': {
      title: 'TF-IDF',
      category: 'Feature Engineering',
      difficulty: 'Intermediate',
      readTime: '9 min',
      year: '1970s',
      definition: 'TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical statistic that reflects how important a word is to a document in a corpus.',
      purpose: 'Scores terms to create high-quality numeric representations of documents that weight content relevance while downplaying common grammar terms.',
      workingPrinciple: 'Multiplies the local Term Frequency (TF) by the global Inverse Document Frequency (IDF). The resulting matrix is often L2-normalized to adjust for document lengths.',
      steps: [
        'Compute the Term Frequency (TF) for all words in a document.',
        'Compute the Inverse Document Frequency (IDF) for all words across the corpus.',
        'Multiply the TF by the IDF value for each word slot.',
        'Normalize the final document vector.'
      ],
      example: {
        input: 'Doc: "NLP is fun." Corpus: 100 docs. df("NLP")=10, df("is")=100.',
        output: 'TF-IDF("is") = 0.33 * 0 = 0. TF-IDF("NLP") = 0.33 * 2.3 = 0.76',
        code: `from sklearn.feature_extraction.text import TfidfVectorizer

corpus = ["NLP is fun", "Transformers and attention are related to NLP"]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print("TF-IDF matrix:\\n", X.toarray())`
      },
      advantages: [
        'Extremely effective baseline for document retrieval and classification.',
        'Simple, fast to calculate, and easily interpretable.',
        'Balances local document occurrence with global corpus context.'
      ],
      limitations: [
        'Maintains the bag-of-words assumption (ignores word order).',
        'Vectors become extremely sparse for large vocabularies.',
        'Cannot capture synonyms or semantic overlap (e.g., "happy" and "joyful" have unrelated dimensions).'
      ],
      applications: [
        'Search Engine document retrieval and scoring (Lucene, Elasticsearch).',
        'Text Classification baseline (Naive Bayes + TF-IDF).',
        'Keyword and keyphrase extraction.'
      ],
      keyTakeaways: [
        'TF-IDF = local relevance × global rarity.',
        'Essential foundation for classical information retrieval systems.',
        'L2 normalization is critical to prevent bias from document lengths.'
      ],
      relatedConcepts: [
        { label: 'TF', id: 'tf' },
        { label: 'IDF', id: 'idf' }
      ],
      references: [
        'Salton, G., & McGill, M. J. (1986). Introduction to Modern Information Retrieval. McGraw-Hill.'
      ]
    },

    // ==========================================
    // LANGUAGE REPRESENTATION
    // ==========================================
    'one-hot-encoding': {
      title: 'One-Hot Encoding',
      category: 'Language Representation',
      difficulty: 'Beginner',
      readTime: '5 min',
      year: '1960s',
      definition: 'One-hot encoding is a representation of categorical variables as binary vectors where only a single dimension is marked with 1.',
      purpose: 'Converts categorical word tokens into discrete numeric vectors that can serve as input features for neural network layers.',
      workingPrinciple: 'Creates a vocabulary of size V. Each word is mapped to a vector of length V, filled with zeros, with a single 1 at the word\'s corresponding index.',
      steps: [
        'Build a sorted vocabulary list from the corpus of size V.',
        'Assign a unique integer index to each word.',
        'Create a zero vector of length V for the target word.',
        'Set the value at the assigned index to 1.'
      ],
      example: {
        input: 'Vocabulary: ["cat", "dog", "mouse"]',
        output: 'cat: [1, 0, 0], dog: [0, 1, 0], mouse: [0, 0, 1]',
        code: `import numpy as np

vocab = ["cat", "dog", "mouse"]
word = "dog"
vector = np.zeros(len(vocab))
vector[vocab.index(word)] = 1
print(f"One-hot vector for '{word}': {vector}")
# Output: One-hot vector for 'dog': [0. 1. 0.]`
      },
      advantages: [
        'Extremely simple and mathematically deterministic.',
        'Perfect representation for categorical classifications (no ordinal bias).',
        'Easy to implement.'
      ],
      limitations: [
        'Orthogonal vectors share no semantic connection (dot product is always 0).',
        'Scales poorly; vocabulary of 100,000 words requires 100,000-dimensional vectors.',
        'Extremely sparse memory allocation.'
      ],
      applications: [
        'Classification layer outputs (Softmax inputs).',
        'Input layers for early feedforward neural network models.',
        'Categorical variable representation.'
      ],
      keyTakeaways: [
        'One-hot vectors are sparse and mutually orthogonal.',
        'They capture no semantic relationships between words.',
        'Replaced by dense word embeddings for neural network inputs.'
      ],
      relatedConcepts: [
        { label: 'Word Embeddings', id: 'word-embeddings' },
        { label: 'Bag of Words', id: 'bag-of-words' }
      ],
      references: [
        'Harris, Z. S. (1954). Distributional structure. Word.'
      ]
    },

    'word-embeddings': {
      title: 'Word Embeddings',
      category: 'Language Representation',
      difficulty: 'Intermediate',
      readTime: '7 min',
      year: '2003',
      definition: 'Word embeddings are dense, low-dimensional, continuous vector representations of words that capture semantic relationships.',
      purpose: 'Compresses vocabulary vectors from high-dimensional sparse representations (one-hot) into dense vector spaces where geometric proximity reflects semantic similarity.',
      workingPrinciple: 'Maps words to vectors of fixed size (usually 50–300 dimensions) using weights learned by neural networks. Training objective is based on the Distributional Hypothesis: words that appear in similar contexts share similar meanings.',
      steps: [
        'Map one-hot words into a continuous space using a projection weight matrix.',
        'Train model parameters using context prediction objectives.',
        'Save the learned projection matrix weights as the lookup embeddings.'
      ],
      example: {
        input: 'Words "king", "queen", "man", "woman"',
        output: 'Vector relations: vector("king") - vector("man") + vector("woman") ≈ vector("queen")',
        code: `import numpy as np
# Mock representation of a 3-dimensional embedding lookup
embeddings = {
    "king":  np.array([0.9,  0.1,  0.8]),
    "queen": np.array([0.8,  0.9,  0.8]),
    "man":   np.array([0.2,  0.1,  0.3]),
    "woman": np.array([0.1,  0.9,  0.3])
}
# Geometric vector arithmetic test
result = embeddings["king"] - embeddings["man"] + embeddings["woman"]
print("Arithmetic output:", result)
print("Queen embedding:  ", embeddings["queen"])`
      },
      advantages: [
        'Dense representation saves significant memory and compute.',
        'Captures rich semantic and syntactic relationships.',
        'Enables geometric similarity measures like cosine distance.'
      ],
      limitations: [
        'Static representations: words have only one vector, conflating polysemes (e.g., "bank" the river vs. "bank" the financial institution).',
        'Can inherit human societal biases present in the training corpora.',
        'Requires huge amounts of text to train from scratch.'
      ],
      applications: [
        'Neural machine translation inputs.',
        'Sentiment analysis neural nets.',
        'Semantic search engines.'
      ],
      keyTakeaways: [
        'Word embeddings map words to a dense continuous vector space.',
        'Geometric proximity represents semantic similarity.',
        'Replaced by contextual embeddings for downstream LLM pipelines.'
      ],
      relatedConcepts: [
        { label: 'One-Hot Encoding', id: 'one-hot-encoding' },
        { label: 'Word2Vec', id: 'word-2vec' }
      ],
      references: [
        'Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). A neural probabilistic language model. Journal of Machine Learning Research.'
      ]
    },

    'word-2vec': {
      title: 'Word2Vec',
      category: 'Language Representation',
      difficulty: 'Intermediate',
      readTime: '9 min',
      year: '2013',
      definition: 'Word2Vec is a framework developed by Google that uses simple two-layer neural networks to learn high-quality continuous word embeddings.',
      purpose: 'Provides an efficient method to train dense word embeddings from large unlabeled text corpora.',
      workingPrinciple: 'Consists of two training architectures: Continuous Bag of Words (CBOW), which predicts a target word from its context, and Skip-gram, which predicts context words from a target word. It uses techniques like negative sampling to speed up training.',
      steps: [
        'Prepare sliding text window frames from the corpus.',
        'Feed center word (Skip-gram) or context words (CBOW) as one-hot indices.',
        'Project to a bottleneck linear hidden layer (the embedding space).',
        'Predict output probabilities and update weights using Negative Sampling.'
      ],
      example: {
        input: 'Context: "the [?] barked"',
        output: 'CBOW predicts: "dog" (highest probability score)',
        code: `from gensim.models import Word2Vec

sentences = [["the", "dog", "barked"], ["cats", "meow", "loudly"]]
# Train a Word2Vec model on mock corpus
model = Word2Vec(sentences, vector_size=50, window=2, min_count=1)
vector = model.wv['dog']
print("Vector shape:", vector.shape)
print("Similarity (dog, cats):", model.wv.similarity('dog', 'cats'))`
      },
      advantages: [
        'Very fast to train compared to older deep probabilistic models.',
        'Captures grammatical and semantic relationships as vector arithmetic.',
        'Weights can be easily exported as a static embedding lookup.'
      ],
      limitations: [
        'Cannot resolve homonyms (only one representation per word spelling).',
        'Does not capture subword context (treats "playing" and "played" as separate words).',
        'Requires a large sliding window for optimal semantic capture.'
      ],
      applications: [
        'Downstream neural classifier inputs.',
        'Recommendation systems (Item2Vec).',
        'Vocabulary clustering.'
      ],
      keyTakeaways: [
        'Word2Vec uses CBOW or Skip-gram models.',
        'Negative sampling makes training scalable.',
        'Static embeddings have a single lookup slot per word.'
      ],
      relatedConcepts: [
        { label: 'Word Embeddings', id: 'word-embeddings' },
        { label: 'FastText', id: 'fasttext' }
      ],
      references: [
        'Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space. arXiv.'
      ]
    },

    'fasttext': {
      title: 'FastText',
      category: 'Language Representation',
      difficulty: 'Intermediate',
      readTime: '8 min',
      year: '2016',
      definition: 'FastText is an extension of the Word2Vec model developed by Facebook that represents each word as a bag of character n-grams.',
      purpose: 'Enables embedding learning for morphologically rich languages and allows vector generation for out-of-vocabulary (OOV) words.',
      workingPrinciple: 'Breaks words down into character n-grams (e.g., "where" with n=3 -> ["<wh", "whe", "her", "ere", "re>"]). The final word representation is the sum of these character n-gram vectors.',
      steps: [
        'Convert word strings into list of character n-gram sub-strings.',
        'Retrieve embedding vector for each n-gram segment.',
        'Sum up sub-vectors to generate the final word embedding vector.',
        'Apply Skip-gram objective updates.'
      ],
      example: {
        input: 'New word "unplayability" (OOV in dictionary)',
        output: 'Vector derived from subword segments like "play", "ability", etc.',
        code: `from gensim.models import FastText

sentences = [["the", "dog", "barked"], ["cats", "meow", "loudly"]]
# Train FastText model on mock corpus
model = FastText(sentences, vector_size=50, window=2, min_count=1)
# Fetch vector for an OOV word
oov_vector = model.wv['dogs']  # "dogs" was not in training set
print("Dogs vector shape:", oov_vector.shape)`
      },
      advantages: [
        'Handles out-of-vocabulary (OOV) words effortlessly.',
        'Performs exceptionally well on morphologically rich languages (e.g., German, Turkish).',
        'Typos and spelling variations share high similarity with correct forms.'
      ],
      limitations: [
        'High memory footprint due to indexing millions of character n-grams.',
        'Slow lookup compared to static key-value matrices.',
        'Like Word2Vec, it still uses a static representation for words in context.'
      ],
      applications: [
        'Search query spelling match normalization.',
        'Text Classification on noisy social media datasets.',
        'Text token representations in low-resource environments.'
      ],
      keyTakeaways: [
        'FastText represents words as sums of character n-grams.',
        'Solves the out-of-vocabulary (OOV) problem.',
        'Requires more memory than Word2Vec.'
      ],
      relatedConcepts: [
        { label: 'Word2Vec', id: 'word-2vec' },
        { label: 'Word Embeddings', id: 'word-embeddings' }
      ],
      references: [
        'Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2017). Enriching word vectors with subword information. Transactions of the Association for Computational Linguistics.'
      ]
    },

    'contextual-embeddings': {
      title: 'Contextual Embeddings',
      category: 'Language Representation',
      difficulty: 'Advanced',
      readTime: '10 min',
      year: '2018',
      definition: 'Contextual embeddings are dynamic word representations that change depending on the surrounding words in a sentence.',
      purpose: 'Solves the polysemy problem by generating different vectors for the same word depending on its syntactic context.',
      workingPrinciple: 'Passes the entire token sequence through deep bidirectional architectures (like bi-LSTMs in ELMo or self-attention blocks in BERT). Representation vectors are extracted from hidden layers.',
      steps: [
        'Pass token sequence into deep neural layer pipelines.',
        'Allow units to attend to or scan left and right contexts.',
        'Output a distinct vector projection for each token position dynamically.'
      ],
      example: {
        input: 's1: "bank of a river" vs s2: "deposit money in the bank"',
        output: 'Dynamic vectors for "bank" are different in s1 and s2.',
        code: `from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model     = AutoModel.from_pretrained("bert-base-uncased")

tokens1 = tokenizer("river bank", return_tensors="pt")
tokens2 = tokenizer("money bank", return_tensors="pt")

with torch.no_grad():
    out1 = model(**tokens1).last_hidden_state
    out2 = model(**tokens2).last_hidden_state

# The vectors for 'bank' differ depending on the preceding token
print("S1 bank shape:", out1[0, 2, :].shape)
print("S2 bank shape:", out2[0, 2, :].shape)`
      },
      advantages: [
        'Accurately represents polysemantic contexts.',
        'Captures syntactic and semantic features of sentence syntax.',
        'Powers high-accuracy downstream classifications.'
      ],
      limitations: [
        'Computationally expensive to calculate in real-time.',
        'Requires large GPU resources.',
        'Embeddings cannot be pre-calculated and stored as a simple lookup table.'
      ],
      applications: [
        'State-of-the-art Named Entity Recognition.',
        'Question Answering architectures.',
        'Coreference resolution pipelines.'
      ],
      keyTakeaways: [
        'Contextual embeddings are dynamic and context-dependent.',
        'They resolve the polysemy problem.',
        'Calculated on-the-fly using deep neural network passes.'
      ],
      relatedConcepts: [
        { label: 'Word Embeddings', id: 'word-embeddings' },
        { label: 'Transformer Representations', id: 'transformer-representations' }
      ],
      references: [
        'Peters, M. E., et al. (2018). Deep contextualized word representations. NAACL.'
      ]
    },

    // ==========================================
    // LANGUAGE MODELS
    // ==========================================
    'statistical-language-models': {
      title: 'Statistical Language Models',
      category: 'Language Models',
      difficulty: 'Intermediate',
      readTime: '8 min',
      year: '1980s',
      definition: 'Statistical language models estimate the probability distribution over sequences of words to predict the likelihood of a text sequence.',
      purpose: 'Predicts the next word in a sequence using joint probability distributions over word lists.',
      workingPrinciple: 'Uses the Markov assumption to approximate the probability of a word sequence by calculating product chains of local conditional probabilities (n-gram frequencies). It uses smoothing algorithms (like Kneser-Ney) to handle zero-count occurrences.',
      steps: [
        'Count frequency patterns of n-gram tokens in training texts.',
        'Calculate conditional probability fractions.',
        'Apply smoothing formulas to distribute probability to unseen transitions.',
        'Predict sequences by chain multiplication.'
      ],
      example: {
        input: 'P("the", "dog", "barked") using Bigram model',
        output: 'P(the, dog, barked) = P(the) × P(dog | the) × P(barked | dog)',
        code: `from collections import Counter, defaultdict

corpus = "the dog barked the cat meowed".split()
# Build simple transition bigram count map
bigram_counts = defaultdict(Counter)
for w1, w2 in zip(corpus[:-1], corpus[1:]):
    bigram_counts[w1][w2] += 1

# Calculate P(dog | the)
total_the = sum(bigram_counts["the"].values())
prob_dog_the = bigram_counts["the"]["dog"] / total_the
print("P(dog | the):", prob_dog_the)
# Output: P(dog | the): 0.5`
      },
      advantages: [
        'Simple, fast, and highly interpretable.',
        'Requires no neural network training hardware.',
        'Performs well on domain-restricted vocabulary.'
      ],
      limitations: [
        'Sparsity: cannot assign probabilities to unseen transitions without smoothing.',
        'Ignorant of long-distance dependencies due to the Markov constraint.',
        'Suffers from exponential state growth for larger context histories.'
      ],
      applications: [
        'Speech recognition language decoders.',
        'Keyboard predictive typing suggestions.',
        'Simple statistical machine translation.'
      ],
      keyTakeaways: [
        'Statistical models rely on frequency counts.',
        'The Markov assumption limits history checks.',
        'Smoothing (like Kneser-Ney) is required for zero-frequency edge cases.'
      ],
      relatedConcepts: [
        { label: 'N-Grams', id: 'n-grams' },
        { label: 'Neural Language Models', id: 'neural-language-models' }
      ],
      references: [
        'Kneser, R., & Ney, H. (1995). Improved backing-off for n-gram language modeling. ICASSP.'
      ]
    },

    'neural-language-models': {
      title: 'Neural Language Models',
      category: 'Language Models',
      difficulty: 'Intermediate',
      readTime: '9 min',
      year: '2003',
      definition: 'Neural language models use neural networks to model the probability of text sequences, mapping words to dense vectors to generalize to unseen contexts.',
      purpose: 'Predicts the next word in a sequence while overcoming the sparsity limitations of statistical language models.',
      workingPrinciple: 'Maps word histories to dense embeddings, processes them through neural layers (Feedforward, RNN, or LSTM), and outputs a Softmax probability distribution over the entire vocabulary.',
      steps: [
        'Tokenize sequence and map words to dense embeddings.',
        'Pass vectors through recurrent (LSTM/GRU) neural layers.',
        'Project final hidden state to vocabulary dimension projection.',
        'Apply Softmax to calculate next-word probabilities.'
      ],
      example: {
        input: 'Sequence "the cat sat on the [?]"',
        output: 'Softmax output indicates "mat" (0.45), "floor" (0.22), "bed" (0.10).',
        code: `import torch
import torch.nn as nn

# Simple architecture outline of a Neural Language Model cell
class NeuralLMCell(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.lstm       = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc         = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embeds = self.embeddings(x)
        out, _ = self.lstm(embeds)
        logits = self.fc(out[:, -1, :])
        return logits

model = NeuralLMCell(vocab_size=1000, embed_dim=64, hidden_dim=128)
print(model)`
      },
      advantages: [
        'Generalizes well to unseen word combinations due to dense vector similarity.',
        'Does not suffer from the curse of dimensionality.',
        'Captures longer context windows than n-gram models.'
      ],
      limitations: [
        'Takes longer to train compared to statistical models.',
        'LSTMs struggle with long-distance context dependencies.',
        'Inference requires continuous matrix multiplications on GPU/CPU.'
      ],
      applications: [
        'Text Generation pipelines.',
        'Machine Translation encoders.',
        'Neural predictive keyboard models.'
      ],
      keyTakeaways: [
        'Neural models use continuous dense vectors.',
        'They scale training representations to generalize context.',
        'Formed the middle step between statistical n-grams and modern transformers.'
      ],
      relatedConcepts: [
        { label: 'Statistical Language Models', id: 'statistical-language-models' },
        { label: 'Transformer Representations', id: 'transformer-representations' }
      ],
      references: [
        'Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). A neural probabilistic language model. JMLR.'
      ]
    },

    'transformer-representations': {
      title: 'Transformer Representations',
      category: 'Language Models',
      difficulty: 'Advanced',
      readTime: '11 min',
      year: '2017',
      definition: 'Transformer representations are deep contextual vectors generated by multi-head self-attention stacks, forming the core of modern LLMs.',
      purpose: 'Enables highly parallelizable training and extracts complex relationships between words across long sentences.',
      workingPrinciple: 'Processes tokens in parallel. It uses positional encodings to inject sequence order, and multi-head attention blocks to dynamically relate each token to all other tokens in the sequence.',
      steps: [
        'Convert input tokens to word embeddings and add positional encodings.',
        'Pass vectors into multi-head self-attention layer blocks.',
        'Apply Feed-Forward layer and Layer Normalization updates.',
        'Extract hidden layer outputs as final representation vectors.'
      ],
      example: {
        input: 'Sequence "Attention is all you need"',
        output: 'High-dimensional contextual output tensor (shape: sequence_length × model_dimension).',
        code: `from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model     = AutoModel.from_pretrained("bert-base-uncased")

inputs = tokenizer("Attention is all you need", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# Hidden states output tensor representing the sequence
representations = outputs.last_hidden_state
print("Contextual representation shape:", representations.shape)
# Output: Contextual representation shape: torch.Size([1, 7, 768])`
      },
      advantages: [
        'Processes tokens in parallel, enabling training on massive datasets.',
        'Learns long-distance dependencies across paragraphs.',
        'Powers downstream model fine-tuning (transfer learning).'
      ],
      limitations: [
        'High computational cost: self-attention complexity scales quadratically with sequence length.',
        'Requires massive data and compute budgets to train from scratch.',
        'High memory footprint during inference.'
      ],
      applications: [
        'Large Language Models (BERT, GPT, Claude).',
        'Question Answering systems.',
        'Document Summarization.'
      ],
      keyTakeaways: [
        'Transformers rely entirely on self-attention mechanisms.',
        'They replace recurrence with parallel processing layers.',
        'Positional encodings are required to preserve sequence order.'
      ],
      relatedConcepts: [
        { label: 'Contextual Embeddings', id: 'contextual-embeddings' },
        { label: 'Neural Language Models', id: 'neural-language-models' }
      ],
      references: [
        'Vaswani, A., et al. (2017). Attention is all you need. Advances in Neural Information Processing Systems.'
      ]
    }
  };

  // Add aliases for missing entries to map all 21 requested names to database keys:
  DATABASE['sentence-segmentation'] = DATABASE['sentence-segmentation'];
  DATABASE['tokenization'] = DATABASE['tokenization'];
  DATABASE['case-normalization'] = DATABASE['case-normalization'];
  DATABASE['stop-word-removal'] = DATABASE['stop-word-removal'];
  DATABASE['stemming'] = DATABASE['stemming'];
  DATABASE['lemmatization'] = DATABASE['lemmatization'];
  DATABASE['noise-removal'] = DATABASE['noise-removal'];
  DATABASE['text-cleaning'] = DATABASE['text-cleaning'];
  
  DATABASE['bag-of-words'] = DATABASE['bag-of-words'];
  DATABASE['n-grams'] = DATABASE['n-grams'];
  DATABASE['tf'] = DATABASE['tf'];
  DATABASE['idf'] = DATABASE['idf'];
  DATABASE['tf-idf'] = DATABASE['tf-idf'];
  
  DATABASE['one-hot-encoding'] = DATABASE['one-hot-encoding'];
  DATABASE['word-embeddings'] = DATABASE['word-embeddings'];
  DATABASE['word-2vec'] = DATABASE['word2vec'] = DATABASE['word-2vec'];
  DATABASE['fasttext'] = DATABASE['fasttext'];
  DATABASE['contextual-embeddings'] = DATABASE['contextual-embeddings'];
  
  DATABASE['statistical-language-models'] = DATABASE['statistical-language-models'];
  DATABASE['neural-language-models'] = DATABASE['neural-language-models'];
  DATABASE['transformer-representations'] = DATABASE['transformer-representations'];

  // Expose database to global scope
  global.NLP = global.NLP || {};
  global.NLP.Database = DATABASE;

})(window);
