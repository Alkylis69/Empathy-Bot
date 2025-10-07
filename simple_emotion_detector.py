"""
Simplified Emotion Detection (Fallback Implementation)

This is a fallback emotion detector that uses keyword-based analysis
when transformers library is not available.
"""

import re
import json
from typing import Dict, List, Optional
from collections import Counter

class SimpleEmotionDetector:
    """
    Simple keyword-based emotion detector for fallback scenarios.
    """
    
    def __init__(self):
        """Initialize with emotion keywords and patterns."""
        
        # Emotion keyword mappings
        self.emotion_keywords = {
            'joy': [
                'happy', 'excited', 'thrilled', 'delighted', 'elated', 'joyful', 
                'cheerful', 'pleased', 'glad', 'fantastic', 'amazing', 'wonderful',
                'great', 'awesome', 'excellent', 'perfect', 'love', 'celebrating',
                'celebration', 'success', 'achievement', 'won', 'victory', 'proud'
            ],
            'sadness': [
                'sad', 'depressed', 'down', 'blue', 'unhappy', 'miserable', 
                'heartbroken', 'crying', 'tears', 'lonely', 'empty', 'disappointed',
                'grief', 'sorrow', 'hurt', 'pain', 'loss', 'devastated', 'hopeless',
                'despair', 'rejected', 'abandoned', 'betrayed', 'failed'
            ],
            'anger': [
                'angry', 'furious', 'mad', 'rage', 'irritated', 'annoyed', 
                'frustrated', 'outraged', 'livid', 'enraged', 'infuriated',
                'pissed', 'hate', 'disgusted', 'fed up', 'stupid', 'ridiculous',
                'unfair', 'betrayed', 'cheated', 'scammed', 'terrible', 'awful'
            ],
            'fear': [
                'scared', 'afraid', 'frightened', 'terrified', 'worried', 
                'anxious', 'nervous', 'concerned', 'panic', 'stress', 'overwhelmed',
                'helpless', 'vulnerable', 'insecure', 'uncertain', 'doubt',
                'apprehensive', 'tense', 'uneasy', 'paranoid', 'threatened'
            ],
            'surprise': [
                'surprised', 'shocked', 'amazed', 'astonished', 'stunned',
                'unexpected', 'wow', 'unbelievable', 'incredible', 'remarkable',
                'extraordinary', 'sudden', 'abrupt', 'out of nowhere', 'blindsided'
            ],
            'disgust': [
                'disgusting', 'gross', 'revolting', 'repulsive', 'sickening',
                'nauseating', 'appalling', 'horrible', 'repugnant', 'vile',
                'offensive', 'distasteful', 'unpleasant', 'yucky'
            ]
        }
        
        # Intensity modifiers
        self.intensity_modifiers = {
            'high': ['very', 'extremely', 'incredibly', 'absolutely', 'totally', 
                    'completely', 'utterly', 'really really', 'so', '!!!', 
                    'tremendously', 'immensely'],
            'medium': ['quite', 'pretty', 'rather', 'fairly', 'somewhat', 'really'],
            'low': ['a bit', 'slightly', 'little', 'kind of', 'sort of', 'maybe']
        }
        
        # Cultural context patterns
        self.cultural_patterns = {
            'western': {
                'direct_expressions': ['I feel', 'I am', 'This makes me'],
                'emphasis_patterns': ['!', 'CAPS', 'repetition']
            },
            'eastern': {
                'indirect_expressions': ['perhaps', 'it seems', 'I believe'],
                'reserved_patterns': ['humble', 'respectful', 'modest']
            }
        }
    
    def detect_emotion(self, text: str, cultural_context: str = 'default') -> Dict:
        """
        Detect emotion using keyword analysis.
        
        Args:
            text (str): Input text to analyze
            cultural_context (str): Cultural context for analysis
            
        Returns:
            Dict: Emotion detection results
        """
        if not text or not isinstance(text, str):
            return self._create_neutral_result()
        
        # Normalize text
        text_lower = text.lower()
        
        # Count emotion keywords
        emotion_scores = {}
        for emotion, keywords in self.emotion_keywords.items():
            score = 0
            for keyword in keywords:
                # Count occurrences of each keyword
                score += text_lower.count(keyword)
            emotion_scores[emotion] = score
        
        # If no emotions detected, return neutral
        if sum(emotion_scores.values()) == 0:
            return self._create_neutral_result()
        
        # Find primary emotion
        primary_emotion = max(emotion_scores.keys(), key=lambda k: emotion_scores[k])
        
        # Calculate confidence based on keyword matches
        total_matches = sum(emotion_scores.values())
        primary_matches = emotion_scores[primary_emotion]
        confidence = min(0.95, primary_matches / max(1, total_matches) * 0.8 + 0.2)
        
        # Determine intensity
        intensity = self._calculate_intensity(text)
        
        # Apply cultural context adjustments
        adjusted_confidence = self._apply_cultural_context(
            confidence, primary_emotion, text, cultural_context
        )
        
        # Normalize emotion scores for output
        if total_matches > 0:
            normalized_scores = {k: v / total_matches for k, v in emotion_scores.items()}
        else:
            normalized_scores = {'neutral': 1.0}
        
        return {
            'primary_emotion': primary_emotion,
            'confidence': adjusted_confidence,
            'all_emotions': normalized_scores,
            'cultural_context': cultural_context,
            'original_text': text,
            'processed_text': text_lower,
            'intensity': intensity,
            'keyword_matches': primary_matches
        }
    
    def _calculate_intensity(self, text: str) -> str:
        """
        Calculate the intensity of the detected emotion.
        
        Args:
            emotion_scores (Dict[str, float]): Emotion scores
            
        Returns:
            str: Intensity level ('low', 'medium', 'high')
        """

        try:
            text_lower = text.lower() if isinstance(text, str) else ""
            exclam = text.count('!') if text else 0
            caps = sum(1 for c in text if c.isupper()) / max(1, len(text)) if text else 0
            modifiers = self.intensity_modifiers or {'high':[], 'medium':[], 'low':[]}
            high_match = any(mod in text_lower for mod in modifiers.get('high', []))
            med_match = any(mod in text_lower for mod in modifiers.get('medium', []))
            low_match = any(mod in text_lower for mod in modifiers.get('low', []))
            
            if exclam > 2 or caps > 0.3 or high_match: return 'high'
            if exclam >= 1 or caps >= 0.1 or med_match: return 'medium'
            if low_match: return 'low'
            return 'high' 

        except Exception:
            return 'low'
    
    def _apply_cultural_context(self, confidence: float, emotion: str, 
                               text: str, context: str) -> float:
        """Apply cultural context adjustments to confidence."""
        
        if context == 'eastern':
            # Eastern cultures might be more reserved in emotional expression
            if emotion in ['anger', 'joy']:
                confidence *= 0.9  # Slightly reduce confidence for strong emotions
        elif context == 'western':
            # Western cultures might be more direct
            if any(pattern in text.lower() for pattern in ['i feel', 'i am']):
                confidence *= 1.1  # Increase confidence for direct expressions
        
        return min(0.95, confidence)  # Cap at 95%
    
    def _create_neutral_result(self) -> Dict:
        """Create neutral emotion result."""
        return {
            'primary_emotion': 'neutral',
            'confidence': 0.5,
            'all_emotions': {'neutral': 1.0},
            'cultural_context': 'default',
            'original_text': '',
            'processed_text': '',
            'intensity': 'low',
            'keyword_matches': 0
        }
    
    def batch_detect_emotions(self, texts: List[str], 
                             cultural_context: str = 'default') -> List[Dict]:
        """Detect emotions for multiple texts."""
        return [self.detect_emotion(text, cultural_context) for text in texts]
    
    def get_emotion_trends(self, emotion_results: List[Dict]) -> Dict:
        """Analyze emotion trends from results."""
        if not emotion_results:
            return {
                'dominant_emotion': 'neutral',
                'trend': 'stable',
                'analysis': 'No data available'
            }
        
        # Count emotions
        emotions = [result['primary_emotion'] for result in emotion_results]
        emotion_counts = Counter(emotions)
        
        # Find dominant emotion
        dominant_emotion = emotion_counts.most_common(1)[0][0]
        
        # Simple trend analysis
        if len(emotions) >= 3:
            recent = emotions[-3:]
            if all(e in ['joy', 'surprise'] for e in recent):
                trend = 'improving'
            elif all(e in ['sadness', 'anger', 'fear'] for e in recent):
                trend = 'declining'
            else:
                trend = 'mixed'
        else:
            trend = 'stable'
        
        return {
            'dominant_emotion': dominant_emotion,
            'emotion_distribution': dict(emotion_counts),
            'trend': trend,
            'total_messages': len(emotion_results),
            'analysis': f"Primary emotion: {dominant_emotion} with {trend} trend"
        }

# Test the simple detector
if __name__ == "__main__":
    detector = SimpleEmotionDetector()
    
    test_texts = [
        "I'm so happy and excited about this!",
        "I feel really sad and down today.",
        "This is absolutely infuriating!",
        "I'm quite worried about tomorrow.",
        "What a surprising turn of events!",
        "The weather is okay."
    ]
    
    print("Simple Emotion Detector Test:")
    print("-" * 40)
    
    for text in test_texts:
        result = detector.detect_emotion(text)
        print(f"Text: {text}")
        print(f"Emotion: {result['primary_emotion']} "
              f"(confidence: {result['confidence']:.2f}, "
              f"intensity: {result['intensity']})")
        print()