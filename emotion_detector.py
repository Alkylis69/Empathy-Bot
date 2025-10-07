"""
Emotion Detection Model

This module implements sentiment analysis and emotion detection using pre-trained
language models with cultural context awareness.
"""

# --IMPORTS--
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers not available, using simple emotion detection")

import numpy as np
from collections import deque
from typing import Dict, List
import logging

try:
    from data_preprocessing import DataPreprocessor, load_cultural_context
except ImportError:
    print("module data_preprocessing not available")
    # Create minimal fallback implementations
    class DataPreprocessor:
        def preprocess_text(self, text, **kwargs):
            if not text:
                return ""
            return text.lower().strip()
    
    def load_cultural_context():
        return {
            'western': {'communication_style': 'direct'},
            'eastern': {'communication_style': 'indirect'},
            'default': {'communication_style': 'balanced'}
        }

# Import simple detector as fallback
try:
    from simple_emotion_detector import SimpleEmotionDetector
except ImportError:
    SimpleEmotionDetector = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --MAIN CLASS--
class EmotionDetector:
    """
    Advanced emotion detection system with cultural context awareness.
    """
    
    def __init__(self, model_name: str = "SamLowe/roberta-base-go_emotions"): # j-hartmann/emotion-english-distilroberta-base   ||  SamLowe/roberta-base-go_emotions
        """
        Initialize the emotion detector with a pre-trained model.
        
        Args:
            model_name (str): HuggingFace model name for emotion detection
        """

        self.model_name = model_name
        self.preprocessor = DataPreprocessor()
        self.cultural_context = load_cultural_context()
        
        # Initialize simple detector as fallback
        if SimpleEmotionDetector:
            self.simple_detector = SimpleEmotionDetector()
        else:
            self.simple_detector = None
        
        # Initialize all extended emotion scores
        self.emotion_scores = {
            "admiration": 0.0, "amusement": 0.0, "anger": 0.0, "annoyance": 0.0, "approval": 0.0, "caring": 0.0, "confusion": 0.0, "curiosity": 0.0,
            "desire": 0.0, "disappointment": 0.0, "disapproval": 0.0, "disgust": 0.0, "embarrassment": 0.0, "excitement": 0.0, "fear": 0.0,
            "gratitude": 0.0, "grief": 0.0, "joy": 0.0, "love": 0.0, "nervousness": 0.0, "optimism": 0.0, "pride": 0.0, "realization": 0.0,
            "relief": 0.0, "remorse": 0.0, "sadness": 0.0, "surprise": 0.0, "neutral": 0.0
        }

        # Initializing intensity modifiers to determine emotional intensity
        self.intensity_modifiers = {
                'high': [
                    'very', 'extremely', 'incredibly', 'absolutely', 'totally', 
                    'completely', 'utterly', 'really really', 'so', '!!!', 
                    'tremendously', 'immensely', 'super', 'insanely', 
                    'ridiculously', 'hugely', 'wildly', 'overwhelmingly',
                    'soooo', 'nooo way', 'yesss', 'omg', 'oh my god', 'wow', 
                    'ugh', 'argh', 'not at all', 'never', 'no way'
                ],
                'medium': [
                    'quite', 'pretty', 'rather', 'fairly', 'somewhat', 'really',
                    'relatively', 'kind of', 'sort of', 'more or less', 'moderately',
                    'reasonably', 'kinda', 'pretty much'
                ],
                'low': [
                    'a bit', 'slightly', 'little', 'maybe', 'perhaps', 'not really',
                    'not sure', 'sorta', 'I guess', 'possibly', 'kinda small', 
                    'barely', 'hardly', 'almost', 'just a touch','hardly ever', 'scarcely'
                ]
            }


        # Initializing Positive and negative emotions
        self.positive_emotions = [
            "admiration", "amusement", "approval", "caring",
            "curiosity", "desire", "excitement", "gratitude",
            "joy", "love", "optimism", "pride",
            "realization", "relief"
        ]
        # Negative emotions
        self.negative_emotions = [
            "anger", "annoyance", "disappointment",
            "disapproval", "disgust", "embarrassment", "fear",
            "grief", "nervousness", "remorse", "sadness"
        ]
        # Neutral / context-dependent emotions
        self.neutral_emotions = [
            "surprise", "neutral", "confusion"
        ]

        # Check if transformers is available and notify whether to use the main model or the fallback model
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available, falling back to simple emotion detection")
            self.emotion_pipeline = None
            return
        
        # Initialize model and tokenizer
        self._load_model()
        
    def _load_model(self):
        """
        Load the pre-trained emotion detection model.
        """

        # Creating pipeline for easier inference
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

            self.emotion_pipeline = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                top_k=None,
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info(f"Successfully loaded emotion detection model: {self.model_name}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self._load_fallback_model()    # Fallback to a basic sentiment analysis
    
    def _load_fallback_model(self):
        """
        Load a fallback sentiment analysis model.
        """

        # Creating pipeline for easier inference
        try:
            self.emotion_pipeline = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base"
            )
            logger.info("Loaded fallback sentiment analysis model")

        except Exception as e:
            logger.error(f"Error loading fallback model: {e}")
            self.emotion_pipeline = None
            # Ensure simple detector is available
            if not hasattr(self, 'simple_detector') or not self.simple_detector:
                if SimpleEmotionDetector:
                    self.simple_detector = SimpleEmotionDetector()
                    logger.info("Fallback to simple emotion detector")
    
    def detect_emotion(self, text: str, cultural_context: str = 'default') -> Dict:
        """
        Detect emotion in text with cultural context consideration. 
        
        Args:
            text (str): Input text to analyze
            cultural_context (str): Cultural context ('western', 'eastern', 'default')
            
        Returns:
            Dict: Emotion detection results with confidence scores
        """

        if not text or not isinstance(text, str):
            return self._create_neutral_result()    # Return neutral result if text is empty or not a string (funciton is defined later in the class)
        
        if not TRANSFORMERS_AVAILABLE or not self.emotion_pipeline:
            if self.simple_detector:
                return self.simple_detector.detect_emotion(text, cultural_context)      # Use simple detector if transformers not available
            else:
                return self._create_neutral_result()
        
        # Preprocess text (set the **kwargs)
        processed_text = self.preprocessor.preprocess_text(text, remove_stops=False, lemmatize = True)
        
        try:
            # Get emotion predictions from the pipeline using the model
            results = self.emotion_pipeline(processed_text)
             
            # Process results based on model type and get the scores of all 6 emotions
            emotion_scores = self._process_model_output(results[0])     # sending results[0] since result is a [list[list{dict}]] such that the called func will receive only [list{dict}]
            
            # Apply cultural context adjustments to the emotion scores
            adjusted_scores = self._apply_cultural_context(emotion_scores, cultural_context)
            
            # Determine primary emotion
            primary_emotion = max(adjusted_scores.keys(), key=lambda k: adjusted_scores[k])

            #return the annotated results in a dictionary
            return {
                'primary_emotion': primary_emotion,
                'confidence': adjusted_scores[primary_emotion],
                'all_emotions': adjusted_scores,
                'cultural_context': cultural_context,
                'original_text': text,
                'processed_text': processed_text,
                'intensity': self._calculate_intensity(adjusted_scores, text)
            }

        except Exception as e:
            logger.error(f"Error in emotion detection: {e}")
            # Fall back to simple detector if available or send neutral results if any kind of error occurs
            if self.simple_detector:
                return self.simple_detector.detect_emotion(text, cultural_context)
            else:
                return self._create_neutral_result()
    
    def _process_model_output(self, results: List[Dict]) -> Dict[str, float]:
        """
        Process raw model output into standardized emotion scores.
        
        Args:
            results (List[Dict]): Raw model output
            
        Returns:
            Dict[str, float]: Standardized emotion scores
        """
         
        # Check the 'results' received from the model is a list object and also it is not null
        if isinstance(results, list) and results:
            result = results

        # since result variable is not null
            # Handle different model outputs:
            for item in result:
                if item['label'] in self.emotion_scores:
                    self.emotion_scores[item['label']] = item['score']
                else: 
                    self.emotion_scores['neutral'] = 0.5

        # Normalize scores to ensure all emotion scores sum to 1.0 (probability distribution)
        total_score = sum(self.emotion_scores.values())
        if total_score > 0:
            self.emotion_scores = {k: v / total_score for k, v in self.emotion_scores.items()}    #normalizing
        else:
            self.emotion_scores['neutral'] = 1.0                                             # if no emotions were detected, sets neutral to 1.0 (100%)

        return self.emotion_scores
    
    def _apply_cultural_context(self, emotion_scores: Dict[str, float], context: str) -> Dict[str, float]:
        """
        Apply cultural context adjustments to emotion scores.
        
        Args:
            emotion_scores (Dict[str, float]): Raw emotion scores
            context (str): Cultural context
            
        Returns:
            Dict[str, float]: Culturally adjusted emotion scores
        """
        try:
            adjusted_scores = emotion_scores.copy()  
            context_info = self.cultural_context.get(context, self.cultural_context.get('default', {}))
            style = context_info.get('emotional_expression', 'adaptive')

            # Reserved cultures → suppress stronger outward emotions
            if style == 'reserved':
                for emotion in adjusted_scores:
                    if emotion not in self.neutral_emotions:  
                        adjusted_scores[emotion] *= 0.8
                    else: 
                        adjusted_scores[emotion] + 0.2 * sum(emotion_scores.values())
            # Expressive cultures → boost all emotions except neutral
            elif style == 'expressive':
                for emotion in adjusted_scores:
                    if emotion not in self.neutral_emotions:
                        adjusted_scores[emotion] *= 1.2

            # Normalize scores to keep distribution valid
            total = sum(adjusted_scores.values())
            if total > 0:
                adjusted_scores = {k: v / total for k, v in adjusted_scores.items()}
            return adjusted_scores
            
        except Exception:
                return emotion_scores

    def _calculate_intensity(self, emotion_scores: Dict[str, float], text: str) -> str:
        """
        Calculate the intensity of the detected emotion.
        
        Args:
            emotion_scores (Dict[str, float]): Emotion scores
            
        Returns:
            str: Intensity level ('low', 'medium', 'high')
        """

        try:
            max_score = max(emotion_scores.values()) if emotion_scores else 0
            text_lower = text.lower() if isinstance(text, str) else ""
            exclam = text.count('!') if text else 0
            caps = sum(1 for c in text if c.isupper()) / max(1, len(text)) if text else 0
            modifiers = self.intensity_modifiers or {'high':[], 'medium':[], 'low':[]}
            high_match = any(mod in text_lower for mod in modifiers.get('high', []))
            med_match = any(mod in text_lower for mod in modifiers.get('medium', []))
            low_match = any(mod in text_lower for mod in modifiers.get('low', []))

            if max_score >= 0.75:
                if exclam > 2 or caps > 0.3 or high_match: return 'high'
                if exclam >= 1 or caps >= 0.1 or med_match: return 'medium'
                if low_match: return 'low'
                return 'high'
            if max_score >= 0.45:
                if exclam >= 3 or caps >= 0.3 or high_match: return 'high'
                if exclam >= 1 or caps >= 0.1 or med_match: return 'medium'
                if low_match: return 'low'
                return 'medium'
            if exclam >= 3 or caps >= 0.3 or high_match: return 'high'
            if exclam >= 1 or caps >= 0.1 or med_match: return 'medium'
            if low_match: return 'low'
            return 'low'
        except Exception:
            return 'low'
            
    def _create_neutral_result(self) -> Dict:
        """
        Create a neutral emotion result for error cases.
        """

        return {
            'primary_emotion': 'neutral',
            'confidence': 0.5,
            'all_emotions': {'neutral': 1.0},
            'cultural_context': 'default',
            'original_text': '',
            'processed_text': '',
            'intensity': 'low'
        }
    
    def batch_detect_emotions(self, texts: List[str], cultural_context: str = 'default') -> List[Dict]:
        """
        Detect emotions for a batch of texts.
        
        Args:
            texts (List[str]): List of texts to analyze
            cultural_context (str): Cultural context for all texts
            
        Returns:
            List[Dict]: List of emotion detection results
        """

        return [self.detect_emotion(text, cultural_context) for text in texts]
    
    def get_emotion_trends(self, emotion_results: List[Dict]) -> Dict:
        """
        Analyze emotion trends from a series of emotion detection results.
        
        Args:
            emotion_results (List[Dict]): List of emotion detection results
            
        Returns:
            Dict: Trend analysis including dominant emotions and patterns
        """

        if not emotion_results:
            return {
                'dominant_emotion': 'neutral', 
                'trend': 'stable', 
                'analysis': 'No data available'
            }

        # Defining a queue with max 10 emotions [ADJUST THE COUNT ACCORDINGLY]
        recent_emotions = deque(maxlen=10)
        # Count primary emotions
        emotion_counts = {}
        intensities = []
        
        for result in emotion_results:
            emotion = result.get('primary_emotion', 'neutral')
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1                # Creating a dynamic dict to count each emotions gathered from the user
            intensities.append(result.get('intensity', 'low'))                          # Creating a series of emotions in a list
        
        # Find dominant emotion
        dominant_emotion = max(emotion_counts.keys(), key=lambda k: emotion_counts[k])  # Allocating the most found emotion
        
        # Analyze intensity trend
        intensity_values = {'low': 1, 'medium': 2, 'high': 3}
        avg_intensity = np.mean([intensity_values.get(i, 1) for i in intensities])
        
        # Determine trend
        for r in emotion_results:
            recent_emotions.append(r.get('primary_emotion','neutral'))        # allocate the primary emotions in the entire chat

        if len(recent_emotions) < 3:
            trend = "short_term: insufficient_data, medium_term: insufficient_data"

        # Short-term = last 3 emotions
        short_term_trend = self._calculate_trend(list(recent_emotions)[-3:])

        # Medium-term = last 7 emotions (or fewer if not enough yet)
        medium_window = list(recent_emotions)[-7:]
        medium_term_trend = self._calculate_trend(medium_window)

        # If both agree → confident trend
        if short_term_trend == medium_term_trend:
            overall = short_term_trend
        else:
            overall = "mixed"
        
        trend = f"short-term: {short_term_trend}, mid-term: {medium_term_trend} and overall: {overall}"

        return {
            'dominant_emotion': dominant_emotion,
            'emotion_distribution': emotion_counts,
            'average_intensity': avg_intensity,
            'trend': trend,
            'total_messages': len(emotion_results),
            'analysis': f"Primary emotion: {dominant_emotion.upper()} with {trend} trend"
        }

    def _calculate_trend(self, emotions):
        """
        Determine trend based on a list of emotions with graceful degradation and error handling.
        """

        try:
            # Fallback: ensure emotions is iterable
            if not emotions or not isinstance(emotions, (list, tuple)):
                return 'insufficient_data'

            # Count positives and negatives safely
            pos_count = sum(1 for e in emotions if isinstance(e, str) and e in self.positive_emotions)
            neg_count = sum(1 for e in emotions if isinstance(e, str) and e in self.negative_emotions)

            try:
                # If all emotions are the same → stable
                if len(set(emotions)) == 1:
                    return 'stable'
                elif pos_count > neg_count:
                    return 'improving'
                elif neg_count > pos_count:
                    return 'declining'
                else:
                    return 'mixed'
            except Exception:
                # Fallback: assume mixed trend if logic fails
                return 'mixed'

        except Exception:
            # Final fallback: if something unexpected occurs
            return 'insufficient_data'

# ----------------------------------------- Example usage and testing-----------------------------------------

if __name__ == "__main__":
    detector = EmotionDetector()
    
    # Test emotion detection
    test_texts = [
    "I'm so proud of myself for finishing this project ahead of time!",
    "Thank you so much for your help, I really appreciate it.",
    "I can't wait for the concert tomorrow, it's going to be amazing!",
    "That was hilarious, I can't stop laughing!",
    "I'm really disappointed in how this turned out.",
    "Ugh, this is so annoying. Why won't it work?",
    "I feel embarrassed about what I said during the meeting.",
    "I can't stop thinking about my loss… it hurts so much.",
    "Oh wow, I didn't expect that to happen!",
    "I understand your point, let's move forward."
]

    print("Testing Emotion Detection:")
    print("-" * 50)
    
    for text in test_texts:
        result = detector.detect_emotion(text)
        print(f"Text: {text}")
        print(f"Emotion: {result['primary_emotion']} (confidence: {result['confidence']:.2f})")
        print(f"Intensity: {result['intensity']}")
        print("-" * 30)
    
    # Test batch processing
    batch_results = detector.batch_detect_emotions(test_texts)
    trends = detector.get_emotion_trends(batch_results)

    print(f"\nEmotion Trends Analysis:")
    print(f"Dominant emotion: {trends['dominant_emotion']}")
    print(f"Trend: {trends['trend']}")
    print(f"Analysis: {trends['analysis']}")