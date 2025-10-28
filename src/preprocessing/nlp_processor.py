"""
NLP preprocessing module with FinBERT integration for sentiment analysis and embeddings.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import re
import logging
from datetime import datetime
import os
import pickle

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Preprocesses text data for NLP analysis."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.min_length = config['nlp']['preprocessing']['min_text_length']
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not isinstance(text, str):
            return ""
        
        # Remove HTML tags if enabled
        if self.config['nlp']['preprocessing']['clean_html']:
            text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs if enabled
        if self.config['nlp']['preprocessing']['remove_urls']:
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Normalize whitespace if enabled
        if self.config['nlp']['preprocessing']['normalize_whitespace']:
            text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove very short texts
        if len(text) < self.min_length:
            return ""
        
        return text
    
    def preprocess_news_batch(self, news_list: List[Dict]) -> List[Dict]:
        """Preprocess a batch of news items."""
        processed_news = []
        
        for news in news_list:
            # Clean title and summary
            title = self.clean_text(news.get('title', ''))
            summary = self.clean_text(news.get('summary', ''))
            
            # Combine title and summary
            combined_text = f"{title}. {summary}".strip()
            combined_text = self.clean_text(combined_text)
            
            if combined_text:
                processed_news.append({
                    **news,
                    'title_clean': title,
                    'summary_clean': summary,
                    'text_combined': combined_text
                })
        
        return processed_news


class FinBERTProcessor:
    """Processes text using FinBERT for sentiment analysis and embeddings."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model_name = config['nlp']['finbert']['model_name']
        self.max_length = config['nlp']['finbert']['max_length']
        self.batch_size = config['nlp']['finbert']['batch_size']
        self.device = torch.device(config['nlp']['finbert']['device'] if torch.cuda.is_available() else 'cpu')
        
        # Load model and tokenizer
        self._load_model()
        
        # Sentiment labels for FinBERT
        self.sentiment_labels = ['negative', 'neutral', 'positive']
        
    def _load_model(self):
        """Load FinBERT model and tokenizer."""
        try:
            logger.info(f"Loading FinBERT model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("FinBERT model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading FinBERT model: {str(e)}")
            raise
    
    def get_embeddings_and_sentiment(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get FinBERT embeddings and sentiment predictions for a batch of texts.
        
        Returns:
            embeddings: (n_texts, 768) - FinBERT embeddings
            sentiment_probs: (n_texts, 3) - Sentiment probabilities [neg, neu, pos]
            sentiment_scores: (n_texts,) - Aggregated sentiment scores
        """
        if not texts:
            return np.array([]), np.array([]), np.array([])
        
        all_embeddings = []
        all_sentiment_probs = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                
                # Get embeddings from the last hidden state (CLS token)
                embeddings = outputs.hidden_states[-1][:, 0, :].cpu().numpy()  # (batch_size, 768)
                
                # Get sentiment probabilities
                logits = outputs.logits
                sentiment_probs = torch.softmax(logits, dim=-1).cpu().numpy()  # (batch_size, 3)
                
                all_embeddings.append(embeddings)
                all_sentiment_probs.append(sentiment_probs)
        
        # Concatenate all batches
        embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])
        sentiment_probs = np.vstack(all_sentiment_probs) if all_sentiment_probs else np.array([])
        
        # Calculate aggregated sentiment scores (positive - negative)
        sentiment_scores = sentiment_probs[:, 2] - sentiment_probs[:, 0]  # pos - neg
        
        return embeddings, sentiment_probs, sentiment_scores
    
    def process_news_batch(self, news_list: List[Dict]) -> List[Dict]:
        """Process a batch of news items with FinBERT."""
        if not news_list:
            return []
        
        # Extract texts
        texts = [news.get('text_combined', '') for news in news_list]
        texts = [text for text in texts if text]  # Remove empty texts
        
        if not texts:
            return news_list
        
        # Get FinBERT outputs
        embeddings, sentiment_probs, sentiment_scores = self.get_embeddings_and_sentiment(texts)
        
        # Add results to news items
        processed_news = []
        embedding_idx = 0
        
        for news in news_list:
            if news.get('text_combined', ''):
                processed_news.append({
                    **news,
                    'finbert_embedding': embeddings[embedding_idx],
                    'sentiment_probs': sentiment_probs[embedding_idx],
                    'sentiment_score': sentiment_scores[embedding_idx],
                    'sentiment_label': self.sentiment_labels[np.argmax(sentiment_probs[embedding_idx])]
                })
                embedding_idx += 1
            else:
                # For news without text, add default values
                processed_news.append({
                    **news,
                    'finbert_embedding': np.zeros(768),
                    'sentiment_probs': np.array([0.33, 0.34, 0.33]),  # Neutral
                    'sentiment_score': 0.0,
                    'sentiment_label': 'neutral'
                })
        
        return processed_news


class NERProcessor:
    """Simple Named Entity Recognition processor using regex patterns."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.entity_types = config['nlp']['ner']['extract_entities']
        
        # Simple regex patterns for financial entities
        self.patterns = {
            'ORG': [
                r'\b[A-Z][a-zA-Z]+ (?:Inc|Corp|Corporation|Company|Co|Ltd|LLC|LP)\b',
                r'\b[A-Z]{2,5}\b',  # Stock symbols
                r'\b(?:Apple|Microsoft|Google|Amazon|Tesla|Meta|Netflix|Nvidia)\b'
            ],
            'PERSON': [
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'  # Simple name pattern
            ],
            'GPE': [  # Geopolitical entities
                r'\b(?:USA|US|United States|China|Europe|Japan|UK|Canada)\b'
            ],
            'MONEY': [
                r'\$\d+(?:\.\d+)?(?:[BM]|billion|million)?\b'
            ]
        }
        
        logger.info("Initialized simple regex-based NER processor")
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text using regex patterns."""
        if not text:
            return {entity_type: [] for entity_type in self.entity_types}
        
        entities = {entity_type: [] for entity_type in self.entity_types}
        
        for entity_type in self.entity_types:
            if entity_type in self.patterns:
                for pattern in self.patterns[entity_type]:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    if matches:
                        entities[entity_type].extend(matches)
        
        # Remove duplicates and clean up
        for entity_type in entities:
            entities[entity_type] = list(set(entities[entity_type]))
        
        return entities
    
    def process_news_batch(self, news_list: List[Dict]) -> List[Dict]:
        """Process a batch of news items for NER."""
        processed_news = []
        
        for news in news_list:
            text = news.get('text_combined', '')
            entities = self.extract_entities(text)
            
            processed_news.append({
                **news,
                'entities': entities
            })
        
        return processed_news


class EventClassifier:
    """Classifies financial events based on text patterns."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.event_patterns = config['nlp']['event_patterns']
        
        # Compile regex patterns
        self.compiled_patterns = {}
        for event_type, keywords in self.event_patterns.items():
            pattern = '|'.join([rf'\b{keyword}\b' for keyword in keywords])
            self.compiled_patterns[event_type] = re.compile(pattern, re.IGNORECASE)
    
    def classify_events(self, text: str) -> Dict[str, bool]:
        """Classify events in text based on patterns."""
        if not text:
            return {event_type: False for event_type in self.event_patterns}
        
        event_flags = {}
        for event_type, pattern in self.compiled_patterns.items():
            event_flags[event_type] = bool(pattern.search(text))
        
        return event_flags
    
    def process_news_batch(self, news_list: List[Dict]) -> List[Dict]:
        """Process a batch of news items for event classification."""
        processed_news = []
        
        for news in news_list:
            text = news.get('text_combined', '')
            events = self.classify_events(text)
            
            processed_news.append({
                **news,
                'events': events
            })
        
        return processed_news


class NewsProcessor:
    """Main news processing pipeline combining all NLP components."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.cache_dir = config['data']['cache_path']
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize processors
        self.text_preprocessor = TextPreprocessor(config)
        self.finbert_processor = FinBERTProcessor(config)
        self.ner_processor = NERProcessor(config)
        self.event_classifier = EventClassifier(config)
    
    def process_news_data(self, news_data: Dict[str, List[Dict]], 
                         use_cache: bool = True) -> Dict[str, List[Dict]]:
        """Process all news data through the NLP pipeline."""
        cache_file = os.path.join(self.cache_dir, 'processed_news.pkl')
        
        if use_cache and os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                logger.info("Loaded processed news data from cache")
                return cached_data
            except Exception as e:
                logger.warning(f"Error loading processed news cache: {str(e)}")
        
        processed_data = {}
        
        for symbol, news_list in news_data.items():
            logger.info(f"Processing news for {symbol} ({len(news_list)} articles)")
            
            if not news_list:
                processed_data[symbol] = []
                continue
            
            # Step 1: Text preprocessing
            news_list = self.text_preprocessor.preprocess_news_batch(news_list)
            
            # Step 2: FinBERT processing (embeddings + sentiment)
            news_list = self.finbert_processor.process_news_batch(news_list)
            
            # Step 3: Named Entity Recognition
            news_list = self.ner_processor.process_news_batch(news_list)
            
            # Step 4: Event classification
            news_list = self.event_classifier.process_news_batch(news_list)
            
            processed_data[symbol] = news_list
        
        # Cache the processed data
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(processed_data, f)
            logger.info("Cached processed news data")
        except Exception as e:
            logger.warning(f"Error caching processed news data: {str(e)}")
        
        return processed_data
    
    def aggregate_daily_sentiment(self, news_list: List[Dict]) -> pd.DataFrame:
        """Aggregate news sentiment by day."""
        if not news_list:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(news_list)
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        df['date'] = df['date'].dt.date
        
        # Group by date and aggregate
        daily_sentiment = df.groupby('date').agg({
            'sentiment_score': ['mean', 'std', 'count'],
            'sentiment_probs': lambda x: np.mean(np.stack(x), axis=0)
        }).reset_index()
        
        # Flatten column names
        daily_sentiment.columns = ['date', 'sentiment_mean', 'sentiment_std', 'news_count', 'sentiment_probs_mean']
        
        # Fill NaN values
        daily_sentiment['sentiment_std'] = daily_sentiment['sentiment_std'].fillna(0)
        
        return daily_sentiment


def main():
    """Test the NLP processing functionality."""
    import yaml
    
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Test with sample news data
    sample_news = [
        {
            'title': 'Apple Reports Strong Q4 Earnings',
            'summary': 'Apple Inc reported better-than-expected quarterly earnings with revenue growth of 8%.',
            'date': datetime.now(),
            'symbol': 'AAPL'
        },
        {
            'title': 'Market Volatility Increases',
            'summary': 'Stock market experiences increased volatility due to economic uncertainty.',
            'date': datetime.now(),
            'symbol': 'general'
        }
    ]
    
    # Test processors
    processor = NewsProcessor(config)
    
    # Test text preprocessing
    cleaned_news = processor.text_preprocessor.preprocess_news_batch(sample_news)
    print(f"Preprocessed {len(cleaned_news)} news items")
    
    # Test FinBERT processing
    finbert_news = processor.finbert_processor.process_news_batch(cleaned_news)
    print(f"Processed {len(finbert_news)} news items with FinBERT")
    
    for news in finbert_news:
        print(f"Title: {news['title']}")
        print(f"Sentiment: {news['sentiment_label']} ({news['sentiment_score']:.3f})")
        print(f"Embedding shape: {news['finbert_embedding'].shape}")
        print("---")


if __name__ == "__main__":
    main()