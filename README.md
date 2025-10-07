# Emotional Sentiment Analysis and Adaptive Response System

A comprehensive chatbot system that detects emotional states in user messages and responds with culturally appropriate, empathetic responses.

## Features

- **Advanced Emotion Detection**: Uses machine learning models (when available) or keyword-based analysis to identify 7 different emotions (joy, sadness, anger, fear, surprise, disgust, neutral)
- **Cultural Context Awareness**: Adapts responses based on cultural communication styles (Western, Eastern, Default)
- **Empathetic Response Generation**: Generates contextually appropriate and supportive responses
- **Conversation Analytics**: Tracks emotional trends and provides insights
- **Modular Architecture**: Clean, extensible codebase with proper error handling

## Quick Start

### Installation

```bash
# Install basic dependencies (optional for enhanced features)
pip install -r requirements.txt
```

Note: The system works with or without external ML libraries. It gracefully falls back to keyword-based analysis when advanced libraries aren't available.

### Basic Usage

```python
from chatbot import EmotionalChatbot

# Initialize the chatbot
chatbot = EmotionalChatbot(cultural_context='western')

# Process a message
response = chatbot.process_message("I'm feeling really excited about this!")

print(f"Bot: {response['bot_response']}")
print(f"Detected emotion: {response['emotion_analysis']['primary_emotion']}")
```

### CLI Interface

```bash
python chatbot.py
```

This launches an interactive command-line interface where you can chat with the bot.

### Testing

```bash
python test_chatbot.py
```

Choose option 1 for full test suite, option 2 for interactive demo, or option 3 for both.

## Architecture

### Core Components

1. **Data Preprocessing Module** (`data_preprocessing.py`)
   - Text cleaning and standardization
   - Stop word removal and lemmatization
   - Missing value handling
   - Sample data generation

2. **Emotion Detection** (`emotion_detector.py`, `simple_emotion_detector.py`)
   - Primary: Uses transformer models for accurate emotion classification
   - Fallback: Keyword-based emotion detection
   - Cultural context adjustments
   - Batch processing capabilities

3. **Response Generation** (`response_generator.py`)
   - Template-based response generation
   - Cultural context adaptation
   - Conversation memory and continuity
   - Follow-up suggestions

4. **Main Chatbot Integration** (`chatbot.py`)
   - Complete conversation management
   - Emotional trend analysis
   - Session analytics and summaries
   - CLI interface

### Emotion Categories

- **Joy**: Happy, excited, pleased, delighted
- **Sadness**: Down, depressed, disappointed, lonely
- **Anger**: Frustrated, annoyed, furious, irritated
- **Fear**: Worried, anxious, scared, concerned
- **Surprise**: Unexpected, astonished, amazed
- **Disgust**: Revolted, appalled, repulsed
- **Neutral**: Balanced, calm, matter-of-fact

### Cultural Contexts

- **Western**: Direct communication style, moderate emotional expression
- **Eastern**: Indirect communication style, reserved emotional expression
- **Default**: Balanced approach, adaptive to context

## Example Interactions

### Joyful User
```
User: "I just got promoted at work!"
Bot: "What fantastic news! You must be thrilled. Your positive energy is inspiring."
Emotion: joy (confidence: 0.92)
```

### Sad User
```
User: "I've been feeling really down lately."
Bot: "I can hear the pain in your words, and I'm here with you. It's okay to feel sad - these emotions are part of being human."
Emotion: sadness (confidence: 0.88)
```

### Angry User
```
User: "This traffic is so frustrating!"
Bot: "I can feel the frustration and anger in your words. It's completely normal to feel angry about this."
Emotion: anger (confidence: 0.85)
```

## Advanced Features

### Conversation Analytics
- Emotional trend analysis
- Dominant emotion identification
- Conversation quality assessment
- Personalized recommendations

### Extensibility
- Easy to add new emotions
- Customizable response templates
- Pluggable emotion detection models
- Cultural context expansion

## Dependencies

### Required (Built-in)
- Python 3.7+
- Standard library modules (re, json, datetime, logging)

### Optional (Enhanced Features)
- `transformers>=4.30.0` - Advanced emotion detection
- `torch>=2.0.0` - Neural network support
- `pandas>=1.3.0` - Data handling
- `numpy>=1.21.0` - Numerical computations
- `nltk>=3.8` - Natural language processing
- `scikit-learn>=1.2.0` - Machine learning utilities
- `gradio>=3.35.0` - Web interface (future enhancement)

## Testing

The system includes comprehensive tests:

- **Data Preprocessing Tests**: Text cleaning, data loading, annotation
- **Emotion Detection Tests**: Accuracy, batch processing, trends
- **Response Generation Tests**: Template selection, cultural adaptation
- **Integration Tests**: End-to-end conversation flow
- **Error Handling Tests**: Edge cases, invalid inputs

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Future Enhancements

- Web interface using Gradio
- Voice input/output support
- Multi-language support
- Advanced cultural context detection
- Integration with external APIs
- Conversation history persistence
- Real-time emotion visualization

## Support

For issues, questions, or contributions, please refer to the project repository or contact the development team.