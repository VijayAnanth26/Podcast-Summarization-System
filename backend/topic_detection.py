import logging
import re
from typing import List, Dict, Union, Optional
from pathlib import Path
import numpy as np
import os

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check NLTK data availability
try:
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
        
    NLTK_AVAILABLE = True
except:
    logger.warning("NLTK not properly installed; using regex fallbacks")
    NLTK_AVAILABLE = False

# Import needed libraries
try:
    # Import huggingface_hub
    from huggingface_hub import hf_hub_download
    
    # Import sentence transformers first to avoid import errors
    import sentence_transformers
    from sentence_transformers import SentenceTransformer
    
    # Then import BERTopic dependencies
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer
    import umap
    import hdbscan
    
    BERTOPIC_AVAILABLE = True
    logger.info("BERTopic and dependencies available")
except ImportError as e:
    logger.warning(f"BERTopic or its dependencies not properly installed: {e}")
    logger.warning("Topic detection will use fallback keyword extraction")
    BERTOPIC_AVAILABLE = False

class TopicDetector:
    """
    Topic detection using BERTopic to identify main topics in transcript.
    """
    
    def __init__(self, use_best_model=False, fast_mode=False):
        """
        Initialize the topic detector.
        
        Args:
            use_best_model: Whether to use the best (but slower) model
            fast_mode: Whether to use faster processing settings
        """
        self._topic_model = None
        self._embedding_model = None
        self._fallback_mode = not BERTOPIC_AVAILABLE
        self._fast_mode = fast_mode
        
        # Set embedding model based on use case
        if use_best_model and not fast_mode:
            self.embedding_model_name = "all-mpnet-base-v2"  # Better but slower
        elif fast_mode:
            # Set a reliable fast model that works locally
            self.embedding_model_name = "paraphrase-MiniLM-L3-v2"
            # Fallback if fast mode is needed
            self._fast_embedding_model_name = "all-MiniLM-L6-v2"
        else:
            self.embedding_model_name = "all-MiniLM-L12-v2"  # Balanced performance
        
        # Extended stopwords for podcast/audio content
        self._podcast_stopwords = {
            'um', 'uh', 'like', 'you know', 'actually', 'basically', 'literally',
            'sort of', 'kind of', 'obviously', 'yeah', 'right', 'okay', 'oh',
            'hmm', 'huh', 'mm', 'mm-hmm', 'wow', 'ah', 'eh', 'er', 'anyway',
            'thing', 'stuff', 'thing is', 'going', 'going to', 'gonna', 'wanna',
            'want to', 'got to', 'gotta', 'said', 'say', 'says', 'saying',
            'just', 'really', 'very', 'quite', 'people', 'person', 'know', 'think',
            'thought', 'talk', 'talking', 'talked', 'talks', 'tell', 'telling',
            'told', 'tells', 'lot', 'lots', 'little bit', 'mean', 'means', 'meant',
            'actually', 'basically', 'essentially', 'simply', 'perhaps', 'maybe',
            'sure', 'certainly', 'definitely', 'probably', 'possibly', 'likely',
            'look', 'looks', 'looking', 'looked', 'see', 'sees', 'seeing', 'saw',
            'comes', 'come', 'coming', 'came', 'goes', 'go', 'going', 'went', 'gone'
        }
        
        # Try to initialize BERTopic dependencies
        if not self._fallback_mode:
            try:
                self._init_models()
            except Exception as e:
                logger.warning(f"Failed to initialize BERTopic models: {e}")
                logger.warning("Falling back to basic keyword extraction")
                self._fallback_mode = True
    
    def _init_models(self):
        """Initialize the embedding and topic models"""
        if not BERTOPIC_AVAILABLE:
            self._fallback_mode = True
            return
            
        try:
            # For fast mode, always try local models first that don't require authentication
            local_models = [
                "all-MiniLM-L6-v2",
                "paraphrase-MiniLM-L3-v2",
                "distilbert-base-nli-mean-tokens",
                "all-MiniLM-L12-v2"
            ]
            
            # Try loading models in order until one succeeds
            for model_name in local_models:
                try:
                    logger.info(f"Trying to load model: {model_name}")
                    self._embedding_model = SentenceTransformer(model_name)
                    logger.info(f"Successfully loaded model: {model_name}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load model {model_name}: {str(e)}")
                    continue
            
            # If no model has been loaded yet, try the default model
            if self._embedding_model is None:
                logger.info("Using default model: all-MiniLM-L6-v2")
                try:
                    self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                except Exception as e:
                    logger.error(f"Failed to load default model: {str(e)}")
                    self._fallback_mode = True
                    return
            
            # Determine model mode
            is_best_model = self.embedding_model_name == "all-mpnet-base-v2"
            is_fast_mode = self._fast_mode
            
            # Get standard English stopwords
            if NLTK_AVAILABLE:
                try:
                    from nltk.corpus import stopwords
                    nltk_stopwords = set(stopwords.words('english'))
                except:
                    nltk_stopwords = set()
            else:
                nltk_stopwords = set()
            
            # Combine standard stopwords with podcast-specific ones
            extended_stopwords = list(nltk_stopwords.union(self._podcast_stopwords))
            
            # Configure UMAP settings based on mode
            if is_fast_mode:
                # Super fast UMAP settings
                umap_model = umap.UMAP(
                    n_neighbors=5,
                    n_components=3,
                    min_dist=0.0,
                    metric='cosine',
                    low_memory=True,
                    random_state=42
                )
            elif is_best_model:
                # High quality settings
                umap_model = umap.UMAP(
                    n_neighbors=20,
                    n_components=8,
                    min_dist=0.0,
                    metric='cosine',
                    random_state=42
                )
            else:
                # Balanced settings
                umap_model = umap.UMAP(
                    n_neighbors=15,
                    n_components=5,
                    min_dist=0.0,
                    metric='cosine',
                    random_state=42
                )
            
            # Configure HDBSCAN settings based on mode
            if is_fast_mode:
                # Super fast clustering
                hdbscan_model = hdbscan.HDBSCAN(
                    min_cluster_size=3,
                    min_samples=2,
                    metric='euclidean',
                    cluster_selection_method='eom',
                    prediction_data=True
                )
            elif is_best_model:
                # High quality clustering
                hdbscan_model = hdbscan.HDBSCAN(
                    min_cluster_size=4,
                    min_samples=3,
                    metric='euclidean',
                    cluster_selection_method='eom',
                    prediction_data=True
                )
            else:
                # Balanced settings
                hdbscan_model = hdbscan.HDBSCAN(
                    min_cluster_size=5,
                    min_samples=2,
                    metric='euclidean',
                    cluster_selection_method='eom',
                    prediction_data=True
                )
            
            # Configure CountVectorizer based on mode
            if is_fast_mode:
                # Super fast, limited vocabulary
                vectorizer_model = CountVectorizer(
                    stop_words=extended_stopwords,
                    ngram_range=(1, 2),  # Limited to bigrams for speed
                    max_features=5000
                )
            elif is_best_model:
                # Comprehensive vocabulary
                vectorizer_model = CountVectorizer(
                    stop_words=extended_stopwords,
                    ngram_range=(1, 3),
                    max_features=20000
                )
            else:
                # Balanced vocabulary
                vectorizer_model = CountVectorizer(
                    stop_words=extended_stopwords,
                    ngram_range=(1, 3),
                    max_features=10000
                )
            
            # Create the BERTopic model with settings based on selected mode
            self._topic_model = BERTopic(
                # Embedding model
                embedding_model=self._embedding_model,
                
                # Dimensionality reduction
                umap_model=umap_model,
                
                # Clustering
                hdbscan_model=hdbscan_model,
                
                # Topic representation
                vectorizer_model=vectorizer_model,
                
                # Other parameters - simpler for fast mode
                nr_topics="auto",
                calculate_probabilities=not is_fast_mode,  # Skip probability calc in fast mode
                verbose=False
            )
            
            logger.info(f"BERTopic model initialized successfully using {self.embedding_model_name} (fast_mode={is_fast_mode})")
            
        except Exception as e:
            logger.error(f"Error initializing BERTopic models: {str(e)}")
            self._fallback_mode = True
            raise
    
    def _improve_topic_names(self, keywords):
        """
        Improve topic names by using first keyword or finding collocations.
        
        Args:
            keywords: List of keywords for a topic
            
        Returns:
            Tuple of (topic_name, description)
        """
        if not keywords or len(keywords) == 0:
            return "Unknown Topic", "No keywords available for this topic"
        
        # If first keyword is a proper noun or compound noun, use it directly
        first_word = keywords[0].strip()
        if first_word[0].isupper() or "_" in first_word or len(first_word.split()) > 1:
            # It's likely a name or a proper entity
            cleaned_name = first_word.replace("_", " ").strip()
            return cleaned_name.title(), f"Topics related to {cleaned_name}"
        
        # If we have at least 2 keywords, try to create a more descriptive name
        if len(keywords) >= 2:
            # Check for meaningful pairs (adjective + noun)
            for i, word1 in enumerate(keywords[:3]):  # Try with top 3 keywords
                for word2 in keywords[i+1:i+3]:  # Try the next 2 words
                    combined = f"{word1} {word2}"
                    if len(combined) < 25:  # Not too long
                        return combined.title(), "Topics including " + ", ".join(keywords[:3])
        
        # Fallback to first keyword + "Topics"
        return f"{first_word.title()} Topics", "Topics including " + ", ".join(keywords[:3])
    
    def detect_topics(self, text: str, min_topics: int = 3, max_topics: int = 10, use_best_model: bool = False, fast_mode: bool = False) -> List[Dict]:
        """
        Detect topics in the transcript.
        
        Args:
            text: The transcript text
            min_topics: Minimum number of topics to return
            max_topics: Maximum number of topics to return
            use_best_model: Whether to use the best (but slower) model
            fast_mode: Whether to use the fastest possible processing
            
        Returns:
            List of topics with keywords and relevance scores
        """
        logger.info(f"Starting topic detection (mode={'fast' if fast_mode else 'best' if use_best_model else 'standard'})")
        
        # Use fallback mode if BERTopic is not available or failed to initialize
        if self._fallback_mode:
            logger.info("Using fallback keyword extraction instead of BERTopic")
            return self._fallback_keyword_extraction(text)
        
        # Check if requested mode doesn't match current initialization
        current_is_best = self.embedding_model_name == "all-mpnet-base-v2"
        current_is_fast = self._fast_mode
        
        if (fast_mode and not current_is_fast) or (use_best_model and not current_is_best and not fast_mode):
            logger.info(f"Switching topic detection model to {'fast' if fast_mode else 'best'} mode")
            self._fast_mode = fast_mode
            self.embedding_model_name = "paraphrase-MiniLM-L3-v2" if fast_mode else "all-mpnet-base-v2" if use_best_model else "all-MiniLM-L6-v2"
            self._init_models()
        
        try:
            # Clean the transcript text
            cleaned_text = self._clean_transcript(text)
            
            # In fast mode, use bigger chunks with less overlap
            chunk_size = 200 if self._fast_mode else 150
            overlap = 10 if self._fast_mode else 30
            
            # Split the transcript into semantic documents/chunks
            documents = self._split_into_semantic_chunks(
                cleaned_text, 
                min_chunk_size=100,
                max_chunk_size=chunk_size
            )
            
            if not documents:
                logger.warning("No documents created from transcript")
                return self._fallback_keyword_extraction(text)
            
            logger.info(f"Split transcript into {len(documents)} semantic chunks")
            
            # Check if there are enough documents for UMAP
            if len(documents) < 10:  # Increase threshold from 5 to 10 for better reliability
                logger.warning(f"Not enough content for UMAP-based topic detection, only {len(documents)} chunks available. Using fallback method.")
                return self._fallback_keyword_extraction(text)
            
            # Process fewer documents in fast mode if transcript is long
            if self._fast_mode and len(documents) > 20:
                # Take representative samples instead of all documents
                step = max(1, len(documents) // 20)
                sample_docs = documents[::step]
                logger.info(f"Fast mode: sampling {len(sample_docs)} chunks from {len(documents)} total")
                documents = sample_docs
            
            # Additional check to ensure there's enough diversity in the documents
            try:
                # Fit the BERTopic model
                topics, probs = self._topic_model.fit_transform(documents)
                
                # Get unique topics (excluding outlier topic -1)
                unique_topics = sorted(list(set([t for t in topics if t != -1])))
            
            except Exception as e:
                logger.error(f"Error in topic detection: {str(e)}", exc_info=True)
                # Fallback to keyword extraction
                return self._fallback_keyword_extraction(text)
            
            # If we have too few topics, try reducing min_cluster_size and refit
            if len(unique_topics) < min_topics and len(documents) > 10:
                logger.info(f"Found only {len(unique_topics)} topics, adjusting parameters for more topics")
                
                # Adjust HDBSCAN parameters to find more topics
                self._topic_model.hdbscan_model.min_cluster_size = max(2, self._topic_model.hdbscan_model.min_cluster_size - 1)
                self._topic_model.hdbscan_model.min_samples = max(1, self._topic_model.hdbscan_model.min_samples - 1)
                
                # Refit the model
                topics, probs = self._topic_model.fit_transform(documents)
                unique_topics = sorted(list(set([t for t in topics if t != -1])))
            
            # Apply hierarchical topic reduction if we have too many topics
            if len(unique_topics) > max_topics:
                logger.info(f"Found {len(unique_topics)} topics, reducing to {max_topics} with hierarchical clustering")
                self._topic_model.reduce_topics(documents, topics, nr_topics=max_topics)
            
            # Generate topic info
            topic_info = self._topic_model.get_topic_info()
            
            # Create result list
            topics_result = []
            
            # Add outlier topic first if it exists and contains a significant number of documents
            outlier_topic = topic_info[topic_info['Topic'] == -1]
            if not outlier_topic.empty and outlier_topic['Count'].values[0] > len(documents) * 0.1:
                # Generate a general name for outlier topic based on word frequencies
                outlier_words = self._extract_common_words_from_documents(
                    [doc for i, doc in enumerate(documents) if topics[i] == -1]
                )
                
                if outlier_words:
                    topics_result.append({
                        'id': -1,
                        'name': 'Miscellaneous',
                        'keywords': outlier_words[:5],
                        'count': int(outlier_topic['Count'].values[0]),
                        'relevance': 0.5  # Middle relevance for outlier topic
                    })
            
            # Calculate max probability for normalization
            max_prob = max(probs.max(), 0.001)
            
            # For non-outlier topics
            for topic_id in unique_topics:
                # Get top words for this topic
                topic_words = self._topic_model.get_topic(topic_id)
                if not topic_words:
                    continue
                
                # Get just the words (without weights)
                words = [word for word, _ in topic_words]
                
                # Create more natural topic name
                improved_name, description = self._improve_topic_names(words)
                
                # Get count of documents in this topic
                count = len([t for t in topics if t == topic_id])
                
                # Get average probability for this topic (relevance score)
                topic_probs = [probs[i][topic_id] for i, t in enumerate(topics) if t == topic_id]
                avg_prob = sum(topic_probs) / len(topic_probs) if topic_probs else 0
                relevance = min(1.0, avg_prob / max_prob)  # Normalize to 0-1
                
                # Add to results
                topics_result.append({
                    'id': topic_id,
                    'name': improved_name,
                    'keywords': words[:10],  # Include more keywords
                    'count': count,
                    'relevance': round(relevance, 2),
                    'description': description
                })
            
            # Sort by relevance (descending)
            topics_result = sorted(topics_result, key=lambda x: x['relevance'], reverse=True)
            
            # Create topic timeline data (where topics appear in the document)
            topic_timeline = self._create_topic_timeline(documents, topics, probs)
            if topic_timeline:
                for topic in topics_result:
                    if topic['id'] in topic_timeline:
                        topic['timeline'] = topic_timeline[topic['id']]
            
            logger.info(f"Detected {len(topics_result)} topics")
            return topics_result
            
        except Exception as e:
            logger.error(f"Error in topic detection: {str(e)}", exc_info=True)
            # Fallback to keyword extraction
            return self._fallback_keyword_extraction(text)

    def _split_into_semantic_chunks(self, text: str, min_chunk_size: int = 100, max_chunk_size: int = 500) -> List[str]:
        """
        Split text into semantic chunks using sentence boundaries, paragraphs, 
        and speaker transitions as natural breakpoints.
        
        Args:
            text: The transcript text
            min_chunk_size: Minimum chunk size in characters
            max_chunk_size: Maximum chunk size in characters
            
        Returns:
            List of document chunks
        """
        if not text or len(text) < min_chunk_size:
            return [text] if text else []
        
        # Check if NLTK is available for better sentence tokenization
        if NLTK_AVAILABLE:
            try:
                sentences = nltk.sent_tokenize(text)
            except:
                # Fallback to simple regex splitting
                sentences = re.split(r'(?<=[.!?])\s+', text)
        else:
            # Simple regex splitting
            sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Look for speaker transitions (common in transcripts)
        speaker_pattern = re.compile(r'(?:^|\n)(?:Speaker\s+\d+|[A-Z][a-z]+):\s+')
        
        # Find all speaker transitions in the text
        speaker_matches = list(speaker_pattern.finditer(text))
        speaker_positions = [match.start() for match in speaker_matches]
        
        # Combine sentences into chunks considering speaker transitions
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            # If adding this sentence exceeds max length and we have content, 
            # finish current chunk
            if current_length + sentence_len > max_chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # Check if this sentence contains a speaker transition
            for pos in speaker_positions:
                if pos >= 0 and sentence.find(text[pos:pos+20]) >= 0:
                    # Speaker transition found, complete current chunk if not empty
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                        current_chunk = []
                        current_length = 0
                    break
            
            # Add the sentence to current chunk
            current_chunk.append(sentence)
            current_length += sentence_len
            
            # If we've reached min_chunk_size and end with sentence terminator,
            # it's a good breakpoint
            if current_length >= min_chunk_size and sentence[-1] in '.!?':
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
        
        # Add the last chunk if there's content
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _extract_common_words_from_documents(self, documents: List[str]) -> List[str]:
        """
        Extract common words from a set of documents.
        
        Args:
            documents: List of document strings
            
        Returns:
            List of common words
        """
        if not documents:
            return []
            
        # Combine documents
        combined_text = ' '.join(documents)
        
        # Get standard English stopwords
        if NLTK_AVAILABLE:
            try:
                from nltk.corpus import stopwords
                stop_words = set(stopwords.words('english'))
            except:
                stop_words = set()
        else:
            stop_words = set()
        
        # Add podcast stopwords
        stop_words = stop_words.union(self._podcast_stopwords)
        
        # Simple word frequency analysis
        words = re.findall(r'\b[a-zA-Z]{3,}\b', combined_text.lower())
        word_freq = {}
        
        for word in words:
            if word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top words by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:10]]
    
    def _create_topic_timeline(self, documents: List[str], topics: List[int], 
                              probs: np.ndarray) -> Dict[int, List[float]]:
        """
        Create a timeline showing where topics appear in the document.
        
        Args:
            documents: The document chunks
            topics: Topic assignments for each document
            probs: Topic probabilities
            
        Returns:
            Dictionary mapping topic IDs to timeline values
        """
        if not documents or not topics:
            return {}
        
        # Get unique topics excluding outliers (-1)
        unique_topics = sorted(list(set([t for t in topics if t != -1])))
        if not unique_topics:
            return {}
        
        # Create 10 time bins (normalize document positions to 0-1 range)
        num_bins = 10
        timeline = {topic_id: [0.0] * num_bins for topic_id in unique_topics}
        
        # Populate timeline with probability values
        for i, (doc_topic, doc_probs) in enumerate(zip(topics, probs)):
            if doc_topic == -1:
                continue
                
            # Calculate which bin this document belongs to
            bin_idx = min(int(i * num_bins / len(documents)), num_bins - 1)
            
            # Add probability to corresponding bin
            timeline[doc_topic][bin_idx] += doc_probs[doc_topic]
        
        # Normalize each topic's timeline
        for topic_id in timeline:
            max_val = max(timeline[topic_id])
            if max_val > 0:
                timeline[topic_id] = [round(val / max_val, 2) for val in timeline[topic_id]]
        
        return timeline
    
    def _clean_transcript(self, text: str) -> str:
        """
        Clean transcript text by removing speaker labels and timestamps.
        
        Args:
            text: Input transcript text
            
        Returns:
            Cleaned text
        """
        # Remove speaker labels and timestamps like "Speaker 1 [00:01 - 00:05]: "
        cleaned = re.sub(r'Speaker \d+\s+\[\d+:\d+\s+-\s+\d+:\d+\]:\s+', '', text)
        
        # Remove any other timestamp formats like [00:00:00]
        cleaned = re.sub(r'\[\d+:\d+(?::\d+)?\]', '', cleaned)
        
        return cleaned
    
    def _split_into_documents(self, text: str) -> List[str]:
        """
        Split text into documents for topic modeling.
        
        Args:
            text: Input text
            
        Returns:
            List of documents (paragraphs or sentences)
        """
        # Try to split by paragraphs first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if len(paragraphs) >= 5:
            return paragraphs
        
        # If not enough paragraphs, split by sentences
        if NLTK_AVAILABLE:
            try:
                sentences = nltk.sent_tokenize(text)
            except:
                # Fallback to regex if NLTK fails
                sentences = re.split(r'(?<=[.!?])\s+', text)
        else:
            # Use regex directly if NLTK not available
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
        # Group sentences into chunks of 2-3 sentences
        if len(sentences) >= 10:
            chunks = []
            for i in range(0, len(sentences), 2):
                if i + 1 < len(sentences):
                    chunks.append(sentences[i] + " " + sentences[i+1])
                else:
                    chunks.append(sentences[i])
            return chunks
        
        return sentences
    
    def _fallback_keyword_extraction(self, text: str) -> List[Dict]:
        """
        Simple fallback keyword extraction for when BERTopic is not available.
        
        Args:
            text: Input text
            
        Returns:
            List of topic dictionaries
        """
        try:
            # Clean the text
            cleaned_text = self._clean_transcript(text)
            
            # Check if text is too short for meaningful analysis
            if len(cleaned_text.split()) < 20:
                logger.info("Text too short for meaningful topic extraction, using simple word analysis")
                return self._extract_topics_from_short_text(cleaned_text)
            
            # Split into chunks/documents
            documents = self._split_into_documents(cleaned_text)
            
            # If we have very few documents, just use the whole text
            if len(documents) <= 2:
                documents = [cleaned_text]
            
            # Extract common words from all documents
            common_words = self._extract_common_words_from_documents(documents)
            
            # If we couldn't extract any keywords, return a default topic
            if not common_words:
                return [
                    {
                        "name": "General Topic",
                        "keywords": ["general", "content", "audio"],
                        "score": 1.0,
                        "color": "hsl(210, 70%, 50%)"
                    }
                ]
            
            # Group words into topics based on co-occurrence
            topics = []
            
            # Create at least one topic with the most common words
            if common_words:
                top_words = common_words[:min(5, len(common_words))]
                
                # Create a better topic name from the most frequent word
                main_word = top_words[0].title() if top_words else "General"
                
                # Create a more natural topic name
                if main_word.lower() in ["people", "person", "man", "woman", "guy", "girl"]:
                    topic_name = "People Discussion"
                elif main_word.lower() in ["city", "town", "place", "country", "state", "region"]:
                    topic_name = "Location Discussion"
                elif main_word.lower() in ["time", "year", "day", "month", "week"]:
                    topic_name = "Time Discussion"
                else:
                    topic_name = f"{main_word}"
                
                topics.append({
                    "name": topic_name,
                    "keywords": top_words,
                    "score": 0.9,
                    "color": "hsl(210, 70%, 50%)"
                })
            
            # If we have enough words, create a second topic
            if len(common_words) > 5:
                next_words = common_words[5:min(10, len(common_words))]
                
                # Create a better topic name from the most frequent word in this group
                main_word = next_words[0].title() if next_words else "Related"
                
                # Create a more natural topic name
                if main_word.lower() in ["people", "person", "man", "woman", "guy", "girl"]:
                    topic_name = "People Mentioned"
                elif main_word.lower() in ["city", "town", "place", "country", "state", "region"]:
                    topic_name = "Places Mentioned"
                elif main_word.lower() in ["time", "year", "day", "month", "week"]:
                    topic_name = "Time References"
                else:
                    topic_name = f"{main_word}"
                
                topics.append({
                    "name": topic_name,
                    "keywords": next_words,
                    "score": 0.7,
                    "color": "hsl(120, 70%, 50%)"
                })
            
            # If we have even more words, create a third topic
            if len(common_words) > 10:
                last_words = common_words[10:min(15, len(common_words))]
                
                # Create a better topic name from the most frequent word in this group
                main_word = last_words[0].title() if last_words else "Additional"
                
                # Create a more natural topic name
                if main_word.lower() in ["people", "person", "man", "woman", "guy", "girl"]:
                    topic_name = "Individual References"
                elif main_word.lower() in ["city", "town", "place", "country", "state", "region"]:
                    topic_name = "Geographic Context"
                elif main_word.lower() in ["time", "year", "day", "month", "week"]:
                    topic_name = "Temporal Context"
                else:
                    topic_name = f"{main_word}"
                
                topics.append({
                    "name": topic_name,
                    "keywords": last_words,
                    "score": 0.5,
                    "color": "hsl(30, 70%, 50%)"
                })
            
            return topics
            
        except Exception as e:
            logger.error(f"Error in fallback keyword extraction: {str(e)}")
            return self._generate_default_topics(error=str(e))

    def _extract_topics_from_short_text(self, text: str) -> List[Dict]:
        """
        Extract topics from very short text snippets.
        
        Args:
            text: Short text to analyze
            
        Returns:
            List of simple topics
        """
        # Clean and normalize text
        text = text.lower()
        
        # Get standard English stopwords
        if NLTK_AVAILABLE:
            try:
                from nltk.corpus import stopwords
                stop_words = set(stopwords.words('english'))
            except:
                stop_words = set(['a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 'when', 'where', 'how', 'why', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'would', 'should', 'could', 'ought', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'their', 'his', 'her', 'its', 'ours', 'yours', 'theirs'])
        else:
            stop_words = set(['a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 'when', 'where', 'how', 'why', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'would', 'should', 'could', 'ought', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'their', 'his', 'her', 'its', 'ours', 'yours', 'theirs'])
        
        # Add podcast stopwords
        stop_words = stop_words.union(self._podcast_stopwords)
        
        # Extract all words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
        
        # Filter out stopwords and count frequencies
        word_counts = {}
        for word in words:
            if word not in stop_words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # If no meaningful words found, return a generic topic
        if not sorted_words:
            return [{
                "name": "General Content",
                "keywords": ["content", "audio", "speech"],
                "score": 1.0,
                "color": "hsl(210, 70%, 50%)"
            }]
        
        # Extract top words
        top_words = [word for word, _ in sorted_words[:10]]
        
        # Identify potential entities (capitalized words in original text)
        words_with_case = re.findall(r'\b[A-Za-z][a-z]{2,}\b', text)
        potential_entities = [word for word in words_with_case if word[0].isupper()]
        
        topics = []
        
        # If we found potential entities, create a topic for them
        if potential_entities:
            unique_entities = list(set([entity.lower() for entity in potential_entities]))[:5]
            if unique_entities:
                topics.append({
                    "name": "Named Entities",
                    "keywords": unique_entities,
                    "score": 0.9,
                    "color": "hsl(270, 70%, 50%)"
                })
        
        # Create a main topic from the most frequent words
        if top_words:
            main_word = top_words[0].title()
            
            # Categorize the topic based on the main word
            if any(word in ['speak', 'talk', 'say', 'tell', 'said', 'told'] for word in top_words[:3]):
                topic_name = "Speaking"
            elif any(word in ['english', 'language', 'word', 'accent'] for word in top_words[:3]):
                topic_name = "Language"
            elif any(word in ['city', 'town', 'village', 'place', 'area'] for word in top_words[:3]):
                topic_name = "Location"
            elif any(word in ['good', 'great', 'nice', 'best', 'better'] for word in top_words[:3]):
                topic_name = "Quality"
            elif any(word in ['pet', 'dog', 'cat', 'animal'] for word in top_words[:3]):
                topic_name = "Animals"
            else:
                topic_name = f"{main_word}"
            
            topics.append({
                "name": topic_name,
                "keywords": top_words[:5],
                "score": 0.8,
                "color": "hsl(210, 70%, 50%)"
            })
        
        # If we don't have enough topics, add a generic one
        if len(topics) < 2:
            topics.append({
                "name": "General Discussion",
                "keywords": top_words[5:10] if len(top_words) > 5 else ["content", "discussion", "general"],
                "score": 0.6,
                "color": "hsl(120, 70%, 50%)"
            })
        
        return topics

    def extract_detailed_topics(self, text, min_topics=3, max_topics=5, use_best_model=False, fast_mode=False):
        """
        Extract topics with detailed information suitable for visualization.
        
        Args:
            text: Text to analyze
            min_topics: Minimum number of topics to extract
            max_topics: Maximum number of topics to extract
            use_best_model: Whether to use the best (slower) model
            fast_mode: Whether to use the fastest possible processing
            
        Returns:
            List of topics with metadata
        """
        try:
            logger.info(f"Starting detailed topic extraction (min_topics={min_topics}, max_topics={max_topics})")
            
            # Get basic topics first
            topics = self.detect_topics(
                text=text, 
                min_topics=min_topics, 
                max_topics=max_topics, 
                use_best_model=use_best_model,
                fast_mode=fast_mode
            )
            
            if not topics:
                logger.warning("No topics detected in detailed extraction")
                return self._generate_default_topics()
            
            # Assign colors to topics
            topics = self._assign_colors_to_topics(topics)
            
            logger.info(f"Extracted {len(topics)} detailed topics successfully")
            return topics
            
        except Exception as e:
            logger.error(f"Error in detailed topic extraction: {str(e)}", exc_info=True)
            # Return default topics with error information
            return self._generate_default_topics(error=str(e))

    def _generate_default_topics(self, error=None):
        """Generate default topics when topic detection fails"""
        colors = ['#3B82F6', '#EF4444', '#10B981', '#F59E0B', '#6366F1']
        
        topics = [
            {
                'name': 'Content Overview',
                'description': error or 'No specific topics detected',
                'keywords': ['content', 'audio', 'podcast'],
                'relevance': 1.0,
                'color': colors[0]
            }
        ]
        
        # Add a second default topic if we have an error
        if error:
            topics.append({
                'name': 'Audio Analysis',
                'description': 'General audio content analysis',
                'keywords': ['audio', 'recording', 'podcast'],
                'relevance': 0.8,
                'color': colors[1]
            })
        
        return topics

    def _assign_colors_to_topics(self, topics: List[Dict]) -> List[Dict]:
        """Assign colors to topics based on their relevance"""
        colors = ['#3B82F6', '#EF4444', '#10B981', '#F59E0B', '#6366F1']
        
        for topic in topics:
            if 'relevance' in topic:
                topic['color'] = colors[min(int(topic['relevance'] * 4), 4)]
        
        return topics 
