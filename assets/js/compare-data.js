/**
 * महापल्स — Comparison Database
 * ─────────────────────────────────────────────────────────────
 * Contains complete detailed side-by-side comparisons for the
 * 5 requested comparisons.
 */
(function (global) {
  'use strict';

  const COMPARE_DATABASE = {
    'stemming-vs-lemmatization': {
      title: 'Stemming vs Lemmatization',
      category: 'Text Preprocessing',
      difficulty: 'Beginner',
      readTime: '6 min',
      subjects: ['Stemming', 'Lemmatization'],
      overview: 'Stemming and Lemmatization are two key normalization techniques in NLP used to reduce inflectional forms (and sometimes derivationally related forms) of a word to a common base representation.',
      mechanism: 'Stemming uses crude heuristic rules to chop off suffixes from words (e.g. Porter Stemmer rules). It operates purely on string transformations. Lemmatization uses lexical databases (like WordNet) and morphological analysis, parsing the word\'s Part of Speech (POS) in sentence context to look up its true dictionary root (lemma).',
      complexity: 'Stemming runs in O(N) time where N is the number of characters, requiring negligible memory. Lemmatization is significantly slower, O(N + D) where D is the dictionary database lookup time, and requires a POS tagger model and lookup maps loaded into memory.',
      advantages: {
        sub1: [
          'Extremely fast execution with minimal compute.',
          'Very simple to implement in any language without dictionary files.',
          'Consistently reduces vocabulary size for search indexing.'
        ],
        sub2: [
          'Linguistically correct, returning valid dictionary words.',
          'Resolves irregular verb and noun forms (e.g., "went" -> "go", "mice" -> "mouse").',
          'Preserves context and POS semantic distinctions.'
        ]
      },
      disadvantages: {
        sub1: [
          'Overstemming: cuts off too much, grouping unrelated words (e.g., "university" and "universe" -> "univers").',
          'Understemming: fails to group related words (e.g., "criteria" and "criterion").',
          'Results in non-words (stems) which makes interpretation hard.'
        ],
        sub2: [
          'Slow and computationally expensive due to dictionary lookups.',
          'Requires large memory footprint to store the lexicon and POS models.',
          'Lacks support for low-resource languages without rich dictionaries.'
        ]
      },
      applications: {
        sub1: [
          'Search engines (Lucene/Elasticsearch index stemmers).',
          'Large-scale keyword extraction.',
          'Fast document clustering.'
        ],
        sub2: [
          'Question Answering & Chatbots.',
          'Named Entity Recognition pipelines.',
          'Semantic search and conversational AI.'
        ]
      },
      table: [
        { trait: 'Approach', sub1: 'Heuristic suffix stripping', sub2: 'Contextual dictionary lookup' },
        { trait: 'Output Form', sub1: 'Often a non-word stem (e.g., "lazi")', sub2: 'Always a valid dictionary word ("lazy")' },
        { trait: 'POS Tag Aware', sub1: 'No (ignores sentence grammar)', sub2: 'Yes (requires POS tagging for accuracy)' },
        { trait: 'Speed', sub1: 'Extremely Fast (O(N))', sub2: 'Slower (depends on POS and DB lookup)' },
        { trait: 'Resources Needed', sub1: 'None (only small algorithm code)', sub2: 'Large dictionary database (e.g. WordNet)' }
      ],
      summary: 'Stemming is a fast, crude normalization method that strips word suffixes. Lemmatization is a slower, linguistically precise approach that maps inflected words to their true base lemma using contextual POS tags.',
      recommendation: 'Use **Stemming** if you are processing massive amounts of text for search indexing where speed is paramount and readability of the base form doesn\'t matter. Use **Lemmatization** for conversational agents, question answering, and tasks where grammatical correctness and precise semantics are required.'
    },

    'tf-vs-tfidf': {
      title: 'TF vs TF-IDF',
      category: 'Feature Engineering',
      difficulty: 'Beginner',
      readTime: '6 min',
      subjects: ['Term Frequency (TF)', 'TF-IDF'],
      overview: 'Term Frequency (TF) and TF-IDF are statistical vectorization methods used to represent text documents as numeric feature vectors for machine learning.',
      mechanism: 'TF counts the density of a word within a single document (either raw count, normalized by document length, or log-scaled). TF-IDF weights this local frequency by the Inverse Document Frequency (IDF) of the word across the whole corpus, penalizing words that appear everywhere (like "the") and boosting words that are unique.',
      complexity: 'TF runs in O(L) time where L is the document length. TF-IDF requires O(L) for the local term counts, plus O(V) to lookup the vocabulary IDF weights computed across the entire corpus of size D during training.',
      advantages: {
        sub1: [
          'Extremely simple and fast to calculate.',
          'Directly represents term density within a single document.',
          'No global corpus statistics needed; can be calculated on a single string.'
        ],
        sub2: [
          'Attenuates highly frequent but low-information words (stop words) dynamically.',
          'Highlights rare, domain-specific keywords.',
          'Generates more descriptive features for classification.'
        ]
      },
      disadvantages: {
        sub1: [
          'Dominated by stop words ("the", "is") if not heavily pre-filtered.',
          'Biased towards long documents (more words mean higher counts).',
          'Does not represent word rarity in the wider language context.'
        ],
        sub2: [
          'Slightly more complex to calculate and maintain in production.',
          'Requires training over a representative corpus to calculate IDF scores.',
          'Out-of-vocabulary terms in new documents receive zero or default IDF weight.'
        ]
      },
      applications: {
        sub1: [
          'Initial text analysis and word count reports.',
          'Inputs to models that already have built-in weighting layers.',
          'Simple text summarization sentence scoring.'
        ],
        sub2: [
          'Search engine ranking algorithms (e.g. vector space model).',
          'Text classification (e.g. Naive Bayes classification).',
          'Automated keyword extraction.'
        ]
      },
      table: [
        { trait: 'Measurement', sub1: 'Local term occurrences in a document', sub2: 'Local occurrences weighted by global rarity' },
        { trait: 'Stop Words Handling', sub1: 'Vulnerable (common words score highest)', sub2: 'Inherent (common words receive IDF close to 0)' },
        { trait: 'Corpus Dependency', sub1: 'None (calculated per document)', sub2: 'High (requires global document counts)' },
        { trait: 'Vector Quality', sub1: 'Low (captures density, not significance)', sub2: 'High (captures unique topic descriptors)' },
        { trait: 'Memory Footprint', sub1: 'Minimal', sub2: 'Requires storing IDF weight dictionary' }
      ],
      summary: 'Term Frequency measures local occurrence density of words, while TF-IDF adjusts this count by term document frequency across a corpus to emphasize unique words.',
      recommendation: 'Use **TF** only if you are doing simple local text statistics or if your model already incorporates external weighting. For standard search indexing, classification baselines, or document retrieval, always prefer **TF-IDF** as it naturally suppresses stop words and captures topic significance.'
    },

    'word2vec-vs-fasttext': {
      title: 'Word2Vec vs FastText',
      category: 'Language Representation',
      difficulty: 'Intermediate',
      readTime: '7 min',
      subjects: ['Word2Vec', 'FastText'],
      overview: 'Word2Vec and FastText are static word embedding frameworks that represent words as dense vectors in continuous vector spaces.',
      mechanism: 'Word2Vec assigns a single vector to each word in its vocabulary list using neural context prediction (CBOW or Skip-gram). FastText extends Word2Vec by treating each word as a bag of character n-grams, summing up n-gram vectors to construct the final word representation.',
      complexity: 'Word2Vec is lightweight, requiring memory proportional only to vocabulary size V × dimensions. FastText has a much higher training and memory cost because it must store and update vectors for millions of character n-gram segments.',
      advantages: {
        sub1: [
          'Small model file size and fast lookup times.',
          'Excellent at capturing semantic relations (vector analogies).',
          'Lighter compute requirements during training.'
        ],
        sub2: [
          'Handles Out-of-Vocabulary (OOV) words by summing their subwords.',
          'Captures internal word structure, excelling at prefix/suffix patterns.',
          'Resilient to spelling typos and grammatical errors.'
        ]
      },
      disadvantages: {
        sub1: [
          'Cannot generate vectors for words not seen during training.',
          'Ignorant of internal word structure (e.g. treats "play" and "player" as unrelated).',
          'Poor performance on morphologically rich languages.'
        ],
        sub2: [
          'Very large model file sizes due to n-grams.',
          'Slower lookup speeds (vector summation required on-the-fly).',
          'High computational overhead.'
        ]
      },
      applications: {
        sub1: [
          'Downstream neural nets for standard text classification.',
          'Semantic search with clean vocabularies.',
          'Item recommendation systems (Item2Vec).'
        ],
        sub2: [
          'Text processing for social media (with high typos/slang).',
          'Morphologically rich languages (German, Turkish, Finnish).',
          'Search engines handling arbitrary query terms.'
        ]
      },
      table: [
        { trait: 'Atom Unit', sub1: 'Individual whole words', sub2: 'Bag of character n-grams' },
        { trait: 'OOV Words', sub1: 'Fails (returns error or unknown token)', sub2: 'Succeeds (builds vector from subword n-grams)' },
        { trait: 'Morphology Aware', sub1: 'No', sub2: 'Yes (groups root/suffix variations easily)' },
        { trait: 'Model Size', sub1: 'Compact (typically 100MB - 300MB)', sub2: 'Large (typically 1GB - 5GB+ due to n-grams)' },
        { trait: 'Typos Resilience', sub1: 'Low', sub2: 'High (partially matches character segments)' }
      ],
      summary: 'Word2Vec embeds whole words as atomic units. FastText extends this by decomposing words into character n-grams, enabling out-of-vocabulary support and morphology representation.',
      recommendation: 'Use **Word2Vec** if you are memory-constrained or deploying models to standard devices where search terms are well-defined. Use **FastText** if you are working with languages like German, Turkish, or Finnish, or if your dataset contains conversational text (like tweets) filled with spelling mistakes and novel compounds.'
    },

    'statistical-vs-neural': {
      title: 'Statistical vs Neural Language Models',
      category: 'Language Models',
      difficulty: 'Intermediate',
      readTime: '8 min',
      subjects: ['Statistical LMs (N-gram)', 'Neural LMs (RNN/LSTM)'],
      overview: 'Language Models calculate probabilities over word sequences to predict future words, forming the core of autocomplete, translation, and text generation systems.',
      mechanism: 'Statistical language models estimate transition probabilities using frequency counts of n-grams (e.g. Bigrams or Trigrams) in training corpora, using smoothing to handle unseen sequences. Neural language models map history words to continuous dense vectors and project them through neural layers (e.g. RNN, LSTM) using a softmax layer to output probabilities.',
      complexity: 'Statistical models require O(V^N) space theoretically, though stored sparsely. Lookup is an O(1) hash check. Neural models have constant memory space based on weights, but require O(V × H) compute operations (where H is hidden size) for matrix multiplications at each step.',
      advantages: {
        sub1: [
          'Extremely fast, requiring simple count lookups at runtime.',
          'Completely interpretable (probabilities derived directly from frequencies).',
          'Easy to train (requires only simple counting, no backpropagation).'
        ],
        sub2: [
          'Generalizes to unseen contexts using dense vector similarity.',
          'Compact memory footprint (scales with parameters, not vocabulary combinations).',
          'Captures longer context histories than n-gram statistical models.'
        ]
      },
      disadvantages: {
        sub1: [
          'Sparsity problem: cannot evaluate unseen n-gram combinations.',
          'Memory usage scales exponentially with context length N.',
          'Ignores long-range sentence constraints due to the Markov limit.'
        ],
        sub2: [
          'Slow and computationally expensive to train and run inference on.',
          'Acts as a black box (hard to trace probability sources).',
          'Vulnerable to vanishing/exploding gradients during LSTM training.'
        ]
      },
      applications: {
        sub1: [
          'Speech recognition acoustic decoders.',
          'Basic predictive keyboard systems.',
          'Simple query completion.'
        ],
        sub2: [
          'Neural Machine Translation.',
          'Advanced conversational agents.',
          'Creative text generation.'
        ]
      },
      table: [
        { trait: 'Probability Engine', sub1: 'Corpus frequency ratio tables', sub2: 'Softmax activation weights' },
        { trait: 'Generalization', sub1: 'None (unseen n-grams evaluate to 0 or smooth values)', sub2: 'High (similar words share representation spaces)' },
        { trait: 'Context History', sub1: 'Limited (typically N = 3 to 5)', sub2: 'Longer (handled by recurrent hidden states)' },
        { trait: 'Training Compute', sub1: 'Negligible (counting passes)', sub2: 'High (requires GPU backpropagation iterations)' },
        { trait: 'Model Size', sub1: 'Grows with corpus unique n-grams', sub2: 'Fixed by network weight parameters' }
      ],
      summary: 'Statistical models count text patterns to predict sequence probability. Neural models project context into dense layers, capturing long-distance dependencies and generalizing to unseen contexts.',
      recommendation: 'Use **Statistical Models** if you need an ultra-fast, lightweight decoder for restricted grammar domains (like local IVR phone voice menus). Use **Neural Models** (and modern transformers) if you are building semantic systems that require writing generation, deep reasoning, or contextual translations.'
    },

    'onehot-vs-embeddings': {
      title: 'One-Hot Encoding vs Word Embeddings',
      category: 'Language Representation',
      difficulty: 'Beginner',
      readTime: '6 min',
      subjects: ['One-Hot Encoding', 'Word Embeddings'],
      overview: 'One-hot encoding and word embeddings are representation methods used to translate text tokens into numerical formats for neural network layers.',
      mechanism: 'One-hot encoding maps words to sparse binary vectors of vocabulary size V, with a single 1 at the word\'s vocabulary index. Word embeddings project words into a dense, low-dimensional continuous space (usually 50–300 dimensions) where the coordinate values are learned during model training.',
      complexity: 'One-hot encoding requires O(V) space per vector (mostly zeros) and captures no semantic information. Word embeddings require O(D) space where D << V, and lookup is an O(1) projection offset from a weights matrix.',
      advantages: {
        sub1: [
          'Deterministic and completely simple.',
          'No training or parameter fitting required.',
          'Perfect categorical classification representation.'
        ],
        sub2: [
          'Dense representation saves significant memory and execution compute.',
          'Captures rich semantic and syntactic relationships (geometric proximity).',
          'Enables vector arithmetic (e.g. king - man + woman = queen).'
        ]
      },
      disadvantages: {
        sub1: [
          'Dimension grows linearly with vocabulary size V (highly inefficient).',
          'Orthogonal vectors capture no semantic relationships.',
          'Extremely sparse memory allocations.'
        ],
        sub2: [
          'Requires massive datasets to train meaningful embedding spaces.',
          'Black-box coordinates (hard to interpret what dimension 42 represents).',
          'Static embeddings cannot handle polysemy (words with multiple meanings).'
        ]
      },
      applications: {
        sub1: [
          'Classification output target categories (Softmax labels).',
          'Small vocabulary categorical representations.',
          'Initial inputs to local bag-of-words vectors.'
        ],
        sub2: [
          'Inputs to neural networks and deep sequence models.',
          'Semantic search engines and word similarity analysis.',
          'Transfer learning base weights.'
        ]
      },
      table: [
        { trait: 'Vector Density', sub1: 'Sparse (mostly 0s, single 1)', sub2: 'Dense (real continuous numbers)' },
        { trait: 'Dimensionality', sub1: 'Equal to vocabulary size V (large)', sub2: 'Fixed hyperparameter D (usually 50-300)' },
        { trait: 'Semantic Capture', sub1: 'None (all words are equidistant)', sub2: 'High (similar meanings have small cosine distance)' },
        { trait: 'Computability', sub1: 'Poor for deep networks (causes matrix bloat)', sub2: 'Excellent (optimized for neural matrix layers)' },
        { trait: 'Training Step', sub1: 'None', sub2: 'Required (weights learned during training)' }
      ],
      summary: 'One-hot encoding represents words as high-dimensional, sparse, orthogonal vectors. Word embeddings project tokens into low-dimensional, dense, continuous vector spaces representing semantic relationships.',
      recommendation: 'Use **One-Hot Encoding** for targets and categorical outputs (e.g. output class mapping). Always use **Word Embeddings** for neural network input representation layers, as sparse one-hot vectors will exhaust system memory and lack the semantic connections necessary for generalization.'
    }
  };

  // Expose database to global scope
  global.NLP = global.NLP || {};
  global.NLP.CompareDatabase = COMPARE_DATABASE;

})(window);
