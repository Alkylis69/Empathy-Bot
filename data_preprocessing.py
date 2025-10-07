"""
Data Collection and Preprocessing Module

This module provides utilities for loading and preprocessing conversational data,
handling missing values, and standardizing text for emotion analysis.
"""

import re
from typing import List, Dict
from datetime import datetime

# Try to import advanced libraries, fall back to basic implementations
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
    
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
        
except ImportError:
    print("NLTK is not available")
    NLTK_AVAILABLE = False

# --MAIN CLASS--
class DataPreprocessor: 
    """
    A comprehensive data preprocessing class for emotional sentiment analysis.
    """
    
    def __init__(self, language: str = 'english'):
        """
        Initialize the preprocessor with language settings.
        
        Args:
            language (str): Language for stop words and processing
        """
        self.language = language
        
        # Initialize components based on availability
        if NLTK_AVAILABLE:
            self.stop_words = set(stopwords.words(language))
            self.lemmatizer = WordNetLemmatizer()
        else:
            # Basic stop words fallback
            self.stop_words = set([
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 
                'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
                'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
                'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
                'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
                'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
                'until', 'while', 'of', 'at', 'by', 'for', 'with', 'through',
                'during', 'before', 'after', 'above', 'below', 'up', 'down',
                'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
                'then', 'once'
            ])
            self.lemmatizer = None
        
        # Emotion labels for annotation
        self.emotion_labels = {
            'joy': 0, 'sadness': 1, 'anger': 2, 'fear': 3, 
            'surprise': 4, 'disgust': 5, 'neutral': 6
        }
        
    def clean_text(self, text: str) -> str:
        """
        Clean and standardize text by removing unwanted characters,
        converting to lowercase, and normalizing.
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned and normalized text
        """
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits, keep only letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stop_words(self, text: str) -> str:
        """
        Remove stop words from text.
        Args:
            text (str): Input text    
        Returns:
            str: Text with stop words removed
        """
        if not text:
            return ""
            
        if NLTK_AVAILABLE:
            words = word_tokenize(text)
            #-----------------------------------------------------------------------------------
            # print("NLTK tokenizer used", words)
        else:
            # Simple word splitting fallback
            words = text.split()
            #-----------------------------------------------------------------------------------
            # print("Basic tokenizer used", words)
            
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        return ' '.join(filtered_words)
    
    def lemmatize_text(self, text: str) -> str:
        """
        Lemmatize text to reduce words to their root form.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Lemmatized text
        """
        if not text:
            return ""
            
        if NLTK_AVAILABLE and self.lemmatizer:
            words = word_tokenize(text)
            lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
            return ' '.join(lemmatized_words)
        else:
            # Simple fallback - just return original text
            return text
    
    def preprocess_text(self, text: str, remove_stops: bool = True, lemmatize: bool = True) -> str:
        """
        Complete text preprocessing pipeline.
        
        Args:
            text (str): Raw input text
            remove_stops (bool): Whether to remove stop words
            lemmatize (bool): Whether to lemmatize
            
        Returns:
            str: Fully preprocessed text
        """
        # Clean text
        processed_text = self.clean_text(text)
        
        # Remove stop words if requested
        if remove_stops:
            processed_text = self.remove_stop_words(processed_text)
            
        # Lemmatize if requested
        if lemmatize:
            processed_text = self.lemmatize_text(processed_text)
            
        return processed_text
    
    def load_conversational_data(self, data_source: str):
        """
        Load conversational data from various sources.
        
        Args:
            data_source (str): Path to data file or data source identifier
            
        Returns:
            DataFrame-like object or list: Loaded conversational data
        """
        try:
            if PANDAS_AVAILABLE:
                if data_source.endswith('.csv'):
                    df = pd.read_csv(data_source)
                elif data_source.endswith('.json'):
                    df = pd.read_json(data_source)
                else:
                    # Create sample data for demonstration
                    df = self._create_sample_data()
                return df
            else:
                # Return as list of dictionaries
                return self._create_sample_data()
                
        except Exception as e:
            print(f"Error loading data: {e}")
            return self._create_sample_data()
    
    def handle_missing_values(self, data, strategy: str = 'drop'):
        """
        Handle missing values in the dataset.
        
        Args:
            data: Input data (DataFrame or list)
            strategy (str): Strategy for handling missing values ('drop', 'fill_empty', 'fill_mean')
            
        Returns:
            Processed data with missing values handled
        """
        if PANDAS_AVAILABLE and hasattr(data, 'dropna'):
            # Pandas DataFrame
            df_copy = data.copy()
            
            if strategy == 'drop':
                df_copy = df_copy.dropna()
            elif strategy == 'fill_empty':
                df_copy = df_copy.fillna('')
            elif strategy == 'fill_mean':
                # Fill numeric columns with mean, text columns with empty string
                for column in df_copy.columns:
                    if df_copy[column].dtype in ['int64', 'float64']:
                        df_copy[column] = df_copy[column].fillna(df_copy[column].mean())
                    else:
                        df_copy[column] = df_copy[column].fillna('')
                        
            return df_copy
        else:
            # Handle list of dictionaries
            if strategy == 'drop':
                return [item for item in data if all(v is not None and v != '' for v in item.values())]
            elif strategy == 'fill_empty':
                return [{k: v if v is not None else '' for k, v in item.items()} for item in data]
            else:
                return data
    
    def create_emotion_annotation(self, text: str, emotion: str) -> Dict:
        """
        Create emotion annotation structure for training data.
        
        Args:
            text (str): Input text
            emotion (str): Emotion label
            
        Returns:
            Dict: Annotation structure with text, emotion, and metadata
        """
        return {
            'text': text,
            'emotion': emotion,
            'emotion_id': self.emotion_labels.get(emotion.lower(), 6),  # Default to neutral
            'processed_text': self.preprocess_text(text),
            'text_length': len(text),
            'word_count': len(text.split()),
            'timestamp': datetime.now().isoformat() if 'datetime' in globals() else str(len(text))
        }
    
    def batch_preprocess(self, texts: List[str], **kwargs) -> List[str]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts (List[str]): List of texts to preprocess
            **kwargs: Additional arguments for preprocess_text
            
        Returns:
            List[str]: List of preprocessed texts
        """
        return [self.preprocess_text(text, **kwargs) for text in texts]
    
    def _create_sample_data(self):
        """
        Create sample conversational data for demonstration.
        
        Returns:
            DataFrame or list: Sample data with conversations and emotions
        """
        sample_conversations = [
            {"text": "I'm so excited about the new job opportunity!", "emotion": "joy"},
            {"text": "I feel really down today, nothing seems to go right.", "emotion": "sadness"},
            {"text": "This traffic is making me furious! I'm going to be late again.", "emotion": "anger"},
            {"text": "I'm worried about the presentation tomorrow.", "emotion": "fear"},
            {"text": "Wow, I didn't expect that to happen!", "emotion": "surprise"},
            {"text": "This food tastes terrible.", "emotion": "disgust"},
            {"text": "The weather is okay today.", "emotion": "neutral"},
            {"text": "I can't believe I won the lottery!", "emotion": "joy"},
            {"text": "My pet passed away yesterday.", "emotion": "sadness"},
            {"text": "How dare they treat me like that!", "emotion": "anger"}
        ]
        
        if PANDAS_AVAILABLE:
            return pd.DataFrame(sample_conversations)
        else:
            return sample_conversations
    

def load_cultural_context() -> Dict:
    """
    Load cultural context information for response adaptation.
    
    Returns:
        Dict: Cultural context mappings and guidelines
    """
    return {
    'western': {
        'communication_style': 'direct',
        'emotional_expression': 'expressive',
        'tone_preference': 'casual',
        'self_expression': 'individual',
        'conflict_response': 'confrontational',
        'feedback_style': 'direct',
        'support_preferences': ['practical_advice', 'emotional_validation'],
        'values': ['autonomy', 'achievement', 'openness']
    },
    'eastern': {
        'communication_style': 'indirect',
        'emotional_expression': 'reserved',
        'tone_preference': 'formal',
        'self_expression': 'collective',
        'conflict_response': 'avoidant',
        'feedback_style': 'indirect',
        'support_preferences': ['group_harmony', 'respectful_distance'],
        'values': ['harmony', 'respect', 'duty']
    },
    'default': {
        'communication_style': 'balanced',
        'emotional_expression': 'adaptive',
        'tone_preference': 'neutral',
        'self_expression': 'flexible',
        'conflict_response': 'contextual',
        'feedback_style': 'constructive',
        'support_preferences': ['empathetic_listening', 'gentle_guidance'],
        'values': ['adaptability', 'understanding', 'balance']
    }
}

# ----------------------------------------- Example usage and testing-----------------------------------------

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    
    # Test text preprocessing
    sample_text = "I just received a promotion at work! I am happy!"
    processed = preprocessor.preprocess_text(sample_text)
    print(f"Original: {sample_text}")
    print(f"Processed: {processed}")
    
    # Test data loading
    data = preprocessor.load_conversational_data("sample_data")
    if PANDAS_AVAILABLE:
        print(f"\nLoaded {len(data)} sample conversations from pandas:")
        print(data.head())
    else:
        print(f"\nLoaded {len(data)} sample conversations")
        for i, item in enumerate(data[:3]):
            print(f"{i+1}. {item}")
    
    # Test emotion annotation
    annotation = preprocessor.create_emotion_annotation(sample_text, "joy")
    print(f"\nEmotion annotation: {annotation}")