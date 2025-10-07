"""
Test Suite for Emotional Sentiment Analysis and Adaptive Response System

This module contains tests for all components of the emotional chatbot system.
"""

import sys
import traceback
from typing import List, Dict, Tuple

# Import our modules
try:
    from data_preprocessing import DataPreprocessor, load_cultural_context
except ImportError as e:
    print(f"Data preprocessing import error: {e}")
    sys.exit(1)

try:
    from emotion_detector import EmotionDetector
except ImportError as e:
    print(f"Emotion detector import error: {e}")
    sys.exit(1)

try:
    from response_generator import ResponseGenerator
except ImportError as e:
    print(f"Response generator import error: {e}")
    sys.exit(1)

try:
    from chatbot import EmotionalChatbot
except ImportError as e:
    print(f"Chatbot import error: {e}")
    sys.exit(1)

class ChatbotTester:
    """Comprehensive testing suite for the emotional chatbot system."""
    
    def __init__(self):
        self.test_results = []
        self.failed_tests = []
        
    def run_test(self, test_name: str, test_function):
        """Run a single test and record results."""
        print(f"🧪 Testing: {test_name}")
        try:
            test_function()
            print(f"✅ PASSED: {test_name}")
            self.test_results.append((test_name, "PASSED", None))
        except Exception as e:
            print(f"❌ FAILED: {test_name} - {str(e)}")
            self.failed_tests.append(test_name)
            self.test_results.append((test_name, "FAILED", str(e)))
        print("-" * 50)
    
    def test_data_preprocessing(self):
        """Test data preprocessing functionality."""
        preprocessor = DataPreprocessor()
        
        # Test text cleaning
        dirty_text = "I'm SO HAPPY!!! 😀😀😀 Check out https://example.com and email@test.com"
        clean_text = preprocessor.clean_text(dirty_text)
        assert len(clean_text) > 0, "Text cleaning failed"
        assert "https" not in clean_text, "URL removal failed"
        assert "@" not in clean_text, "Email removal failed"
        
        # Test preprocessing pipeline
        processed = preprocessor.preprocess_text("I'm feeling really SAD today!!! :(")
        assert len(processed) > 0, "Text preprocessing failed"
        
        # Test sample data creation
        data = preprocessor.load_conversational_data("sample")
        assert len(data) > 0, "Sample data creation failed"
        
        # Handle both DataFrame and list cases
        if hasattr(data, 'columns'):  # pandas DataFrame
            assert 'text' in data.columns, "Missing 'text' column in sample data"
            assert 'emotion' in data.columns, "Missing 'emotion' column in sample data"
        else:  # list of dictionaries
            assert 'text' in data[0], "Missing 'text' key in sample data"
            assert 'emotion' in data[0], "Missing 'emotion' key in sample data"
        
        # Test emotion annotation
        annotation = preprocessor.create_emotion_annotation("Happy text", "joy")
        assert 'text' in annotation, "Missing text in annotation"
        assert 'emotion' in annotation, "Missing emotion in annotation"
        assert annotation['emotion'] == 'joy', "Incorrect emotion in annotation"
        
        print("   ✓ Text cleaning working")
        print("   ✓ Preprocessing pipeline working")  
        print("   ✓ Sample data generation working")
        print("   ✓ Emotion annotation working")
    
    def test_cultural_context(self):
        """Test cultural context loading."""
        context = load_cultural_context()
        
        assert 'western' in context, "Missing western cultural context"
        assert 'eastern' in context, "Missing eastern cultural context"
        assert 'default' in context, "Missing default cultural context"
        
        # Test context structure
        for culture_type in ['western', 'eastern', 'default']:
            culture_data = context[culture_type]
            assert 'communication_style' in culture_data, f"Missing communication_style in {culture_type}"
            assert 'emotional_expression' in culture_data, f"Missing emotional_expression in {culture_type}"
        
        print("   ✓ Cultural contexts loaded")
        print("   ✓ Context structure validated")
    
    def test_emotion_detection(self):
        """Test emotion detection functionality."""
        detector = EmotionDetector()
        
        # Test basic emotion detection
        test_texts = [
            ("I'm so happy today!", "joy"),
            ("I feel really sad", "sadness"),
            ("This makes me angry!", "anger"),
            ("I'm scared about tomorrow", "fear"),
            ("What a surprise!", "surprise")
        ]
        
        for text, expected_category in test_texts:
            result = detector.detect_emotion(text)
            
            # Validate result structure
            assert 'primary_emotion' in result, "Missing primary_emotion in result"
            assert 'confidence' in result, "Missing confidence in result"
            assert 'all_emotions' in result, "Missing all_emotions in result"
            assert 'intensity' in result, "Missing intensity in result"
            assert 'cultural_context' in result, "Missing cultural_context in result"
            
            # Check confidence is valid
            assert 0 <= result['confidence'] <= 1, "Invalid confidence score"
            
            # Check intensity is valid
            assert result['intensity'] in ['low', 'medium', 'high'], "Invalid intensity level"
            
            print(f"   ✓ '{text}' -> {result['primary_emotion']} (confidence: {result['confidence']:.2f})")
        
        # Test batch processing
        texts = [item[0] for item in test_texts]
        batch_results = detector.batch_detect_emotions(texts)
        assert len(batch_results) == len(texts), "Batch processing length mismatch"
        
        # Test emotion trends
        trends = detector.get_emotion_trends(batch_results)
        assert 'dominant_emotion' in trends, "Missing dominant_emotion in trends"
        assert 'trend' in trends, "Missing trend analysis"
        
        print("   ✓ Batch emotion detection working")
        print("   ✓ Emotion trends analysis working")
    
    def test_response_generation(self):
        """Test response generation functionality."""
        generator = ResponseGenerator()
        
        # Test response generation for different emotions
        test_scenarios = [
            {
                'emotion_result': {'primary_emotion': 'joy', 'confidence': 0.9, 'intensity': 'high'},
                'user_text': "I got the job!",
                'cultural_context': 'western'
            },
            {
                'emotion_result': {'primary_emotion': 'sadness', 'confidence': 0.8, 'intensity': 'medium'},
                'user_text': "I'm feeling down",
                'cultural_context': 'eastern'
            },
            {
                'emotion_result': {'primary_emotion': 'anger', 'confidence': 0.7, 'intensity': 'high'},
                'user_text': "This is so frustrating!",
                'cultural_context': 'default'
            }
        ]
        
        for scenario in test_scenarios:
            response = generator.generate_response(
                scenario['emotion_result'],
                scenario['user_text'],
                scenario['cultural_context']
            )
            
            # Validate response structure
            assert 'response' in response, "Missing response text"
            assert 'emotion_addressed' in response, "Missing emotion_addressed"
            assert 'cultural_context' in response, "Missing cultural_context"
            assert 'response_type' in response, "Missing response_type"
            assert 'follow_up_suggestions' in response, "Missing follow_up_suggestions"
            
            # Check response is not empty
            assert len(response['response']) > 0, "Empty response generated"
            
            # Check follow-up suggestions
            assert isinstance(response['follow_up_suggestions'], list), "Follow-up suggestions not a list"
            assert len(response['follow_up_suggestions']) > 0, "No follow-up suggestions provided"
            
            print(f"   ✓ {scenario['emotion_result']['primary_emotion']} -> '{response['response'][:50]}...'")
        
        # Test contextual response (requires conversation memory)
        emotion_result = {'primary_emotion': 'neutral', 'confidence': 0.5, 'intensity': 'low'}
        contextual_response = generator.get_contextual_response(emotion_result, "Hello", 'western')
        assert 'response' in contextual_response, "Missing contextual response"
        
        print("   ✓ Contextual response generation working")
    
    def test_chatbot_integration(self):
        """Test complete chatbot integration."""
        chatbot = EmotionalChatbot('western')
        
        # Test message processing
        test_messages = [
            "I'm so excited about my vacation!",
            "I've been feeling really down lately",
            "This traffic is making me furious",
            "I'm nervous about the presentation tomorrow",
            "What a beautiful day it is!"
        ]
        
        for message in test_messages:
            response = chatbot.process_message(message)
            
            # Validate complete response structure
            assert 'bot_response' in response, "Missing bot_response"
            assert 'emotion_analysis' in response, "Missing emotion_analysis"
            assert 'response_metadata' in response, "Missing response_metadata"
            assert 'conversation_metadata' in response, "Missing conversation_metadata"
            
            # Check emotion analysis structure
            emotion_analysis = response['emotion_analysis']
            assert 'primary_emotion' in emotion_analysis, "Missing primary_emotion in analysis"
            assert 'confidence' in emotion_analysis, "Missing confidence in analysis"
            assert 'intensity' in emotion_analysis, "Missing intensity in analysis"
            
            # Check response is meaningful
            assert len(response['bot_response']) > 10, "Response too short"
            
            print(f"   ✓ Message processed: {emotion_analysis['primary_emotion']}")
        
        # Test conversation analytics
        trends = chatbot.get_emotion_trends()
        assert 'status' in trends, "Missing status in trends"
        
        summary = chatbot.get_conversation_summary()
        assert 'session_summary' in summary, "Missing session_summary"
        assert 'emotional_analysis' in summary, "Missing emotional_analysis in summary"
        
        print("   ✓ Conversation analytics working")
        print(f"   ✓ Processed {len(test_messages)} messages successfully")
    
    def test_error_handling(self):
        """Test error handling and edge cases."""
        # Test with empty/invalid inputs
        preprocessor = DataPreprocessor()
        
        # Empty text handling
        empty_result = preprocessor.preprocess_text("")
        assert empty_result == "", "Empty text not handled correctly"
        
        # None input handling  
        none_result = preprocessor.preprocess_text(None)
        assert none_result == "", "None input not handled correctly"
        
        # Test emotion detector with edge cases
        detector = EmotionDetector()
        
        # Empty text
        empty_emotion = detector.detect_emotion("")
        assert empty_emotion['primary_emotion'] == 'neutral', "Empty text should return neutral"
        
        # Very long text (should not crash)
        long_text = "I am happy " * 1000
        long_result = detector.detect_emotion(long_text)
        assert 'primary_emotion' in long_result, "Long text processing failed"
        
        # Test chatbot with invalid input
        chatbot = EmotionalChatbot()
        
        # Empty message
        empty_response = chatbot.process_message("")
        assert 'error' in empty_response or 'bot_response' in empty_response, "Empty message not handled"
        
        # None message
        none_response = chatbot.process_message(None)
        assert 'error' in none_response or 'bot_response' in none_response, "None message not handled"
        
        print("   ✓ Empty input handling working")
        print("   ✓ Invalid input handling working")
        print("   ✓ Long text processing working")
    
    def run_all_tests(self):
        """Run all tests and provide summary."""
        print("🚀 Starting Emotional Chatbot Test Suite")
        print("=" * 60)
        
        # Define all tests
        tests = [
            ("Data Preprocessing", self.test_data_preprocessing),
            ("Cultural Context Loading", self.test_cultural_context),
            ("Emotion Detection", self.test_emotion_detection),
            ("Response Generation", self.test_response_generation),
            ("Chatbot Integration", self.test_chatbot_integration),
            ("Error Handling", self.test_error_handling)
        ]
        
        # Run each test
        for test_name, test_function in tests:
            self.run_test(test_name, test_function)
        
        # Print summary
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        passed = len([r for r in self.test_results if r[1] == "PASSED"])
        failed = len([r for r in self.test_results if r[1] == "FAILED"])
        
        print(f"Total Tests: {len(self.test_results)}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"Success Rate: {(passed/len(self.test_results)*100):.1f}%")
        
        if self.failed_tests:
            print(f"\n❌ Failed Tests:")
            for test_name in self.failed_tests:
                print(f"   - {test_name}")
        else:
            print(f"\n🎉 All tests passed! The emotional chatbot system is working correctly.")
        
        return len(self.failed_tests) == 0

def run_interactive_demo():
    """Run an interactive demo of the chatbot system."""
    print("\n🎯 Interactive Demo Mode")
    print("=" * 40)
    print("This will demonstrate the chatbot with some sample interactions.")
    
    try:
        chatbot = EmotionalChatbot('western')
        
        demo_conversations = [
            "I just got promoted at work! I'm so excited!",
            "I've been feeling really overwhelmed lately with all the responsibilities.",
            "My presentation went terribly wrong and I'm so embarrassed.",
            "I'm worried about my upcoming exam next week.",
            "What a beautiful sunset today!"
        ]
        
        print("\n💬 Demo Conversations:")
        print("-" * 30)
        
        for i, message in enumerate(demo_conversations, 1):
            print(f"\n{i}. User: {message}")
            response = chatbot.process_message(message)
            print(f"   Bot: {response['bot_response']}")
            print(f"   Emotion detected: {response['emotion_analysis']['primary_emotion']} "
                  f"(confidence: {response['emotion_analysis']['confidence']:.2f})")
        
        # Show conversation summary
        print(f"\n📋 Conversation Summary:")
        summary = chatbot.get_conversation_summary()
        print(f"Messages: {summary['session_summary']['total_messages']}")
        print(f"Dominant emotion: {summary['emotional_analysis']['dominant_emotion']}")
        print(f"Themes identified: {', '.join(summary['conversation_patterns']['identified_themes'])}")
        
        print(f"\n🎉 Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # Run the test suite
    tester = ChatbotTester()
    
    print("Choose an option:")
    print("1. Run full test suite")
    print("2. Run interactive demo")
    print("3. Run both")
    
    try:
        choice = input("Enter choice (1-3): ").strip()
        
        if choice in ['1', '3']:
            success = tester.run_all_tests()
            if not success:
                sys.exit(1)
        
        if choice in ['2', '3']:
            run_interactive_demo()
            
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
    except Exception as e:
        print(f"❌ Test runner error: {e}")
        traceback.print_exc()
        sys.exit(1)