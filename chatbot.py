"""
Emotional Sentiment Analysis Chatbot

Main integration module that connects emotion detection with response generation
to create a complete empathetic chatbot system.
"""

import sys
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

# Import our custom modules
from emotion_detector import EmotionDetector
from response_generator import ResponseGenerator
from data_preprocessing import DataPreprocessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmotionalChatbot:
    """
    Complete emotional sentiment analysis chatbot system.
    Combines emotion detection with culturally aware response generation.
    """
    
    def __init__(self, cultural_context: str = 'default'):
        """
        Initialize the emotional chatbot system.
        
        Args:
            cultural_context (str): Default cultural context for responses
        """
        self.cultural_context = cultural_context
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize components
        logger.info("Initializing Emotional Chatbot components...")
        
        try:
            self.emotion_detector = EmotionDetector()
            self.response_generator = ResponseGenerator()
            self.preprocessor = DataPreprocessor()
            
            # Conversation history and analytics
            self.conversation_history = []
            self.emotion_history = []
            self.user_profile = {
                'cultural_context': cultural_context,
                'session_start': datetime.now().isoformat(),
                'total_messages': 0,
                'dominant_emotions': {},
                'conversation_themes': []
            }
            
            logger.info("✅ Emotional Chatbot initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing chatbot: {e}")
            raise
    
    def process_message(self, user_message: str, cultural_context: Optional[str] = None) -> Dict:
        """
        Process a user message through the complete pipeline.
        
        Args:
            user_message (str): User's input message
            cultural_context (str, optional): Override cultural context for this message
            
        Returns:
            Dict: Complete response with emotion analysis and generated response
        """
        if not user_message or not isinstance(user_message, str):
            return self._create_error_response("Invalid message format")
        
        # Use provided cultural context or default
        context = cultural_context or self.cultural_context
        
        try:
            # Step 1: Detect emotion
            logger.info(f"Processing message: {user_message[:50]}...")
            emotion_result = self.emotion_detector.detect_emotion(user_message, context)
            
            # Step 2: Generate appropriate response
            response_data = self.response_generator.get_contextual_response(
                emotion_result, user_message, context
            )
            
            # Step 3: Update conversation history and analytics
            self._update_conversation_history(user_message, emotion_result, response_data)
            
            # Step 4: Prepare complete response
            complete_response = {
                'bot_response': response_data['response'],
                'emotion_analysis': {
                    'primary_emotion': emotion_result['primary_emotion'],
                    'confidence': emotion_result['confidence'],
                    'intensity': emotion_result['intensity'],
                    'all_emotions': emotion_result['all_emotions']
                },
                'response_metadata': {
                    'response_type': response_data['response_type'],
                    'cultural_context': context,
                    'follow_up_suggestions': response_data['follow_up_suggestions']
                },
                'conversation_metadata': {
                    'message_count': len(self.conversation_history),
                    'session_id': self.session_id,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            logger.info(f"✅ Message processed. Emotion: {emotion_result['primary_emotion']}")
            return complete_response
            
        except Exception as e:
            logger.error(f"❌ Error processing message: {e}")
            return self._create_error_response(f"Processing error: {str(e)}")
    
    def _update_conversation_history(self, user_message: str, emotion_result: Dict, 
                                   response_data: Dict):
        """Update internal conversation tracking."""
        
        # Add to conversation history
        conversation_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_message': user_message,
            'emotion': emotion_result['primary_emotion'],
            'confidence': emotion_result['confidence'],
            'intensity': emotion_result['intensity'],
            'bot_response': response_data['response'],
            'response_type': response_data['response_type']
        }
        
        self.conversation_history.append(conversation_entry)
        self.emotion_history.append(emotion_result)
        
        # Update user profile
        self.user_profile['total_messages'] += 1
        emotion = emotion_result['primary_emotion']
        
        if emotion in self.user_profile['dominant_emotions']:
            self.user_profile['dominant_emotions'][emotion] += 1
        else:
            self.user_profile['dominant_emotions'][emotion] = 1
    
    def _create_error_response(self, error_message: str) -> Dict:
        """Create a standardized error response."""
        return {
            'bot_response': "I apologize, but I'm having trouble processing your message right now. Could you please try again?",
            'emotion_analysis': {
                'primary_emotion': 'neutral',
                'confidence': 0.0,
                'intensity': 'low',
                'all_emotions': {'neutral': 1.0}
            },
            'response_metadata': {
                'response_type': 'error',
                'cultural_context': self.cultural_context,
                'follow_up_suggestions': ['Could you rephrase your message?', 'Is there something specific I can help with?']
            },
            'error': error_message,
            'conversation_metadata': {
                'message_count': len(self.conversation_history),
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def get_emotion_trends(self) -> Dict:
        """
        Get emotional trends analysis for the current conversation.
        
        Returns:
            Dict: Analysis of emotional patterns in the conversation
        """
        if not self.emotion_history:
            return {
                'status': 'No conversation data available',
                'trends': {},
                'recommendations': []
            }
        
        trends = self.emotion_detector.get_emotion_trends(self.emotion_history)
        
        # Add conversation-specific insights
        recent_emotions = [result['primary_emotion'] for result in self.emotion_history[-3:]]
        
        recommendations = []
        if trends['dominant_emotion'] in ['sadness', 'anger', 'grief']:
            recommendations.append("Consider focusing on positive coping strategies")
            recommendations.append("Professional support might be beneficial if these feelings persist")
        elif trends['dominant_emotion'] in ['joy', 'love', 'amusement']:
            recommendations.append("Great to see positive emotions! Keep building on what's working")
        
        return {
            'status': 'Analysis complete',
            'trends': trends,
            'recent_pattern': recent_emotions,
            'recommendations': recommendations,
            'session_summary': {
                'duration_messages': len(self.conversation_history),
                'primary_themes': self._identify_themes(),
                'emotional_range': list(set(recent_emotions))
            }
        }
    
    def _identify_themes(self) -> List[str]:
        """Identify conversation themes from message history."""
        if not self.conversation_history:
            return []
        
        # Simple keyword-based theme identification
        themes = []
        all_messages = ' '.join([entry['user_message'].lower() for entry in self.conversation_history])
        
        theme_keywords = {
            'work': ['work', 'job', 'career', 'colleague', 'boss', 'office', 'meeting',],
            'relationships': ['friend', 'family', 'partner', 'relationship', 'love', 'dating',],
            'health': ['tired', 'sick', 'health', 'doctor', 'medicine', 'pain',],
            'personal_growth': ['learning', 'goal', 'achievement', 'success', 'failure', 'improve',],
            'daily_life': ['day', 'morning', 'evening', 'home', 'routine', 'schedule',]
        }
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in all_messages for keyword in keywords):
                themes.append(theme)
        
        return themes[:3]  # Return top 3 themes
    
    def get_conversation_summary(self) -> Dict:
        """
        Get a comprehensive summary of the conversation session.
        
        Returns:
            Dict: Complete conversation analytics and summary
        """
        if not self.conversation_history:
            return {'status': 'No conversation to summarize'}
        
        # Calculate conversation metrics
        total_messages = len(self.conversation_history)
        emotion_distribution = {}
        response_types = {}
        
        for entry in self.conversation_history:
            emotion = entry['emotion']
            response_type = entry['response_type']
            
            emotion_distribution[emotion] = emotion_distribution.get(emotion, 0) + 1
            response_types[response_type] = response_types.get(response_type, 0) + 1
        
        # Determine overall mood
        if emotion_distribution:
            dominant_emotion = max(emotion_distribution.keys(), key=lambda k: emotion_distribution[k])
        else:
            dominant_emotion = 'neutral'
        
        return {
            'session_summary': {
                'session_id': self.session_id,
                'start_time': self.user_profile['session_start'],
                'end_time': datetime.now().isoformat(),
                'total_messages': total_messages,
                'cultural_context': self.cultural_context
            },
            'emotional_analysis': {
                'dominant_emotion': dominant_emotion,
                'emotion_distribution': emotion_distribution,
                'emotional_range': len(emotion_distribution),
                'trends': self.get_emotion_trends()['trends'] if self.emotion_history else {}
            },
            'conversation_patterns': {
                'response_types_used': response_types,
                'identified_themes': self._identify_themes(),
                'conversation_quality': self._assess_conversation_quality()
            },
            'recommendations': self._generate_session_recommendations()
        }
    
    def _assess_conversation_quality(self) -> Dict:
        """Assess the quality and depth of the conversation."""
        if not self.conversation_history:
            return {'quality': 'unknown', 'depth': 'shallow'}
        
        # Simple quality metrics
        avg_message_length = sum(len(entry['user_message']) for entry in self.conversation_history) / len(self.conversation_history)
        emotion_variety = len(set(entry['emotion'] for entry in self.conversation_history))
        
        quality = 'good' if avg_message_length > 30 and emotion_variety > 4 else 'basic'
        depth = 'deep' if len(self.conversation_history) > 5 and emotion_variety > 6 else 'moderate'
        
        return {
            'quality': quality,
            'depth': depth,
            'engagement_score': min(10, len(self.conversation_history) + emotion_variety),
            'emotional_openness': emotion_variety
        }
    
    def _generate_session_recommendations(self) -> List[str]:
        """Generate recommendations based on the conversation session."""
        recommendations = []
        
        if not self.conversation_history:
            return ["Start a conversation to get personalized recommendations"]
        
        # Analyze emotional patterns
        negative_emotions = sum(1 for entry in self.conversation_history 
                              if entry['emotion'] in ['sadness', 'anger', 'fear'])
        total_messages = len(self.conversation_history)
        
        if negative_emotions / total_messages > 0.6:
            recommendations.append("Consider seeking additional emotional support")
            recommendations.append("Practice mindfulness or stress-reduction techniques")
        elif negative_emotions / total_messages < 0.2:
            recommendations.append("Keep maintaining your positive mindset!")
            recommendations.append("Share your positive energy with others")
        
        # Theme-based recommendations
        themes = self._identify_themes()
        if 'work' in themes:
            recommendations.append("Consider work-life balance strategies")
        if 'relationships' in themes:
            recommendations.append("Focus on healthy communication patterns")
        
        return recommendations[:4]  # Limit to most relevant recommendations

def create_cli_interface():
    """Create a simple command-line interface for the chatbot."""
    print("🤖 Emotional Sentiment Analysis Chatbot")
    print("=" * 50)
    print("I'm here to listen and provide empathetic responses based on your emotions.")
    print("Type 'quit' to exit, 'help' for commands, or just start chatting!")
    print("-" * 50)
    
    # Get cultural context preference
    cultural_contexts = {'1': 'western', '2': 'eastern', '3': 'default'}
    print("\nSelect your cultural context preference:")
    print("1. Western (direct communication)")
    print("2. Eastern (more reserved communication)")
    print("3. Default (balanced approach)")
    
    while True:
        choice = input("Enter choice (1-3): ").strip()
        if choice in cultural_contexts:
            cultural_context = cultural_contexts[choice]
            break
        print("Please enter 1, 2, or 3")
    
    print(f"Cultural context set to: {cultural_context}")
    print("-" * 50)
    
    # Initialize chatbot
    try:
        chatbot = EmotionalChatbot(cultural_context)
        print("✅ Chatbot initialized successfully!\n")
    except Exception as e:
        print(f"❌ Error initializing chatbot: {e}")
        return
    
    # Main conversation loop
    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                # Show session summary before exiting
                summary = chatbot.get_conversation_summary()
                print("\n📊 Conversation Summary:")
                print(f"Messages exchanged: {summary.get('session_summary', {}).get('total_messages', 0)}")
                print(f"Dominant emotion: {summary.get('emotional_analysis', {}).get('dominant_emotion', 'N/A')}")
                print(f"Conversation quality: {summary.get('conversation_patterns', {}).get('conversation_quality', {}).get('quality', 'N/A')}")
                print("\n👋 Thank you for chatting! Take care!")
                break
            
            elif user_input.lower() == 'help':
                print("\n🔧 Available commands:")
                print("- 'trends' - Show emotional trends analysis")
                print("- 'summary' - Show conversation summary")
                print("- 'context <western/eastern/default>' - Change cultural context")
                print("- 'quit' - Exit the chatbot")
                print("- Just type normally to chat!")
                continue
            
            elif user_input.lower() == 'trends':
                trends = chatbot.get_emotion_trends()
                print(f"\n📈 Emotional Trends:")
                print(f"Status: {trends['status']}")
                if 'trends' in trends and trends['trends']:
                    print(f"Dominant emotion: {trends['trends'].get('dominant_emotion', 'N/A')}")
                    print(f"Trend: {trends['trends'].get('trend', 'N/A')}")
                continue
            
            elif user_input.lower() == 'summary':
                summary = chatbot.get_conversation_summary()
                print(f"\n📋 Session Summary:")
                print(json.dumps(summary, indent=2))
                continue
            
            elif user_input.lower().startswith('context '):
                new_context = user_input.split(' ', 1)[1].lower()
                if new_context in ['western', 'eastern', 'default']:
                    chatbot.cultural_context = new_context
                    print(f"✅ Cultural context changed to: {new_context}")
                else:
                    print("❌ Invalid context. Use: western, eastern, or default")
                continue
            
            # Process normal message
            if user_input:
                response = chatbot.process_message(user_input)
                
                # Display response
                print(f"\n🤖 Bot: {response['bot_response']}")
                
                # Show emotion analysis (optional, can be toggled)
                emotion = response['emotion_analysis']
                print(f"   💭 Detected emotion: {emotion['primary_emotion']} (confidence: {emotion['confidence']:.2f})")
                
                # Show follow-up suggestions occasionally
                if len(chatbot.conversation_history) % 3 == 0:  # Every 3rd message
                    suggestions = response['response_metadata']['follow_up_suggestions']
                    if suggestions:
                        print(f"   💡 You might want to explore: {suggestions[0]}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or type 'help' for assistance.")

if __name__ == "__main__":
    create_cli_interface()