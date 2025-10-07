"""
Response Generation System

This module creates empathetic responses based on identified emotional states,
tailored to cultural and contextual cues with appropriate support for each emotion.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import Dict, List
from data_preprocessing import load_cultural_context
from emotion_detector import EmotionDetector

load_dotenv(override=True)

class ResponseGenerator:
    """
    Generates culturally aware and empathetic responses based on emotional states.
    """
    
    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        """Initialize the response generator with templates and cultural contexts."""
        self.emotion_detector = EmotionDetector()
        self.cultural_context = load_cultural_context()
        self.response_templates = self._load_response_templates()
        self.conversation_memory = []
        self.groq = OpenAI(api_key=os.getenv('GROQ_API_KEY'), base_url="https://api.groq.com/openai/v1")
        self.model_name = model_name
        
    def _load_response_templates(self) -> Dict:
        """
        Load structured, LLM-ready prompt templates for generating responses 
        across various emotions and cultural contexts.

        Returns:
            Dict: A comprehensive dictionary of prompt templates organized by 
                  emotion, cultural context, and response component.
        """
        return {
            'admiration': {
                'western': {
                    'acknowledgment': "Acknowledge the user's admiration with genuine enthusiasm and positive energy. Mirror their excitement. For example: 'That's fantastic! It's clear they made a huge positive impression on you.' Avoid sounding flat or disinterested.",
                    'encouragement': "Encourage the user to connect with the qualities they admire. Prompt them to reflect on what specifically inspires them. For example: 'It's wonderful to have people to look up to. What specific qualities do you find most inspiring?'",
                    'questions': "Ask open-ended questions to invite more detail about the person or achievement being admired. For example: 'I'd love to hear more. What was the moment you realized how impressive they were?'"
                },
                'eastern': {
                    'acknowledgment': "Acknowledge the user's admiration with respectful and composed language. Show appreciation for the virtue they are recognizing. For example: 'It is a sign of great character to recognize and honor the admirable qualities in others.'",
                    'encouragement': "Encourage quiet reflection on the lessons or virtues observed. Frame it as a learning opportunity. For example: 'Observing such excellence can be a profound source of learning and personal growth.'",
                    'questions': "Ask gentle, respectful questions about the wisdom or character traits admired. For example: 'What lessons do you feel can be learned from their example?'"
                },
                'default': {
                    'acknowledgment': "Validate the user's feelings of admiration warmly and sincerely. Show you understand the positive emotion. For example: 'It sounds like you really admire them, and for good reason.'",
                    'curiosity': "Express curiosity in a way that encourages the user to elaborate on their feelings. For example: 'What is it about them that stands out to you the most?'",
                    'engagement': "Show you are engaged and interested in hearing more about their perspective. For example: 'Tell me more about what they did or what they're like.'"
                }
            },
            'amusement': {
                'western': {
                    'acknowledgment': "Share in the user's amusement with a light and cheerful tone. Use casual, positive language. For example: 'Haha, that sounds hilarious!' or 'That's really funny!'",
                    'engagement': "Engage with the funny aspect of the story, encouraging the user to share more of the humorous details. For example: 'That's great! What happened next?'",
                    'questions': "Ask questions that keep the lighthearted conversation going. For example: 'What was your reaction when that happened?'"
                },
                'eastern': {
                    'acknowledgment': "Acknowledge the user's amusement with a gentle and warm smile in your tone. For example: 'I can sense the lightness and humor in your words. It is good to find moments of joy.'",
                    'engagement': "Engage in a way that shares the pleasant feeling without being overly boisterous. For example: 'Such moments of lightheartedness are a welcome part of life.'",
                    'curiosity': "Express gentle curiosity about the source of their amusement. For example: 'I am pleased to hear something has brought a smile to your face. What was it, if you wish to share?'"
                },
                'default': {
                    'acknowledgment': "Acknowledge the amusement in a friendly and positive way. For example: 'That sounds like it was a really amusing moment.'",
                    'engagement': "Show you're listening and enjoying the story with them. For example: 'I'm glad you shared that with me, it sounds like a great memory.'",
                    'curiosity': "Prompt for more information in a conversational way. For example: 'That's quite the story. What was the context behind it?'"
                }
            },
            'anger': {
                'western': {
                    'acknowledgment': "Acknowledge the user's anger directly and clearly. Use strong, validating language. For example: 'I can hear how angry you are, and it sounds completely justified.'",
                    'validation': "Validate their right to feel angry. Reinforce that their emotional response is normal and acceptable. For example: 'You have every right to be angry about this. Anyone in your situation would feel the same.' Avoid telling them to calm down.",
                    'guidance': "Gently guide them toward clarifying the situation or considering next steps, but only after validating their feelings. For example: 'Would it help to talk through exactly what happened?'"
                },
                'eastern': {
                    'acknowledgment': "Acknowledge the user's anger with calm, respectful, and grounding language. Frame it as a disturbance of their inner peace. For example: 'I respectfully recognize the deep frustration and discord this situation has caused you.'",
                    'validation': "Validate the underlying principles or values that have been violated. For example: 'Such feelings naturally arise when one's sense of fairness and harmony is disrupted.' Avoid language that amplifies the anger itself.",
                    'guidance': "Guide the user toward restoring balance and finding a peaceful resolution. For example: 'What actions do you feel would help restore harmony to this situation?' or 'Could quiet reflection help find a path forward?'"
                },
                'default': {
                    'acknowledgment': "Acknowledge the emotion without judgment. Use empathetic phrasing. For example: 'It sounds like you're going through a really frustrating experience.'",
                    'validation': "Validate their feelings as understandable. For example: 'It makes perfect sense that you would feel angry given those circumstances.'",
                    'support': "Offer a space to vent and be heard. For example: 'I'm here to listen if you need to talk more about it.' Avoid offering unsolicited advice immediately."
                }
            },
            'annoyance': {
                'western': {
                    'acknowledgment': "Acknowledge the user's annoyance with relatable, casual language. For example: 'Ugh, that sounds incredibly annoying to deal with.'",
                    'validation': "Validate the feeling as a common, understandable frustration. For example: 'That's definitely one of those things that would get on anyone's nerves.'",
                    'questions': "Ask a question that allows them to vent a little more or explain the context. For example: 'Is this something that happens often?'"
                },
                'eastern': {
                    'acknowledgment': "Acknowledge the annoyance as a minor disturbance. Use gentle, understanding language. For example: 'I understand that this has been a source of frustration for you.'",
                    'validation': "Validate that small disruptions can be trying. For example: 'It is natural to feel vexed when things are not as they should be.'",
                    'guidance': "Gently guide them toward letting the feeling pass. For example: 'May you find it in you to let this small disturbance pass, like a cloud in the sky.'"
                },
                'default': {
                    'acknowledgment': "Acknowledge their feeling in a simple, direct way. For example: 'That sounds really frustrating.'",
                    'validation': "Let them know their reaction is normal. For example: 'It's completely understandable to be annoyed by that.'",
                    'support': "Offer to listen to more if they need to vent. For example: 'Feel free to share more about it if it helps to get it off your chest.'"
                }
            },
            'approval': {
                'western': {
                    'acknowledgment': "Acknowledge their approval with positive and affirming language. Match their positive tone. For example: 'That's great! I'm glad to hear you're happy with it.'",
                    'encouragement': "Encourage this positive direction or decision. For example: 'It sounds like you've made a great choice.'",
                    'questions': "Ask questions that explore the positive outcome. For example: 'What do you like most about how it turned out?'"
                },
                'eastern': {
                    'acknowledgment': "Acknowledge their approval with composed and respectful language. For example: 'It is good to see that you have found this to be satisfactory and in alignment with your expectations.'",
                    'validation': "Validate their judgment and discernment. For example: 'Your thoughtful consideration has led to a positive and harmonious outcome.'",
                    'curiosity': "Ask gentle questions about their satisfaction. For example: 'What aspects of this bring you the most contentment?'"
                },
                'default': {
                    'acknowledgment': "Acknowledge their positive feeling in a supportive way. For example: 'I'm happy to hear that you approve.'",
                    'engagement': "Show interest in their positive assessment. For example: 'It's wonderful that you feel so positively about it.'",
                    'curiosity': "Ask for more details about why they feel this way. For example: 'What parts of it are working well for you?'"
                }
            },
            'caring': {
                'western': {
                    'acknowledgment': "Acknowledge their caring nature with warmth and appreciation. For example: 'It's really kind and compassionate of you to care so much.'",
                    'validation': "Validate their feelings as a sign of their good character. For example: 'The world needs more people like you who are so considerate and caring.'",
                    'support': "Offer support for them as the caregiver. For example: 'Remember to take care of yourself too while you're looking after others.'"
                },
                'eastern': {
                    'acknowledgment': "Acknowledge their caring with respectful and honoring language. For example: 'Your compassionate heart is a source of comfort and strength to others.'",
                    'validation': "Validate their actions as a virtuous and honorable duty. For example: 'Fulfilling your responsibility with such care and dedication is truly admirable.'",
                    'support': "Offer support by reminding them of inner strength and balance. For example: 'May you find the inner peace and resilience needed to continue offering your support.'"
                },
                'default': {
                    'acknowledgment': "Acknowledge their caring feelings with genuine warmth. For example: 'That's very thoughtful of you. It's clear you care a great deal.'",
                    'validation': "Validate their empathy as a positive trait. For example: 'Your empathy really shines through in what you're saying.'",
                    'support': "Gently check in on their own well-being. For example: 'It takes a lot of energy to care so deeply. How are you holding up?'"
                }
            },
            'confusion': {
                'western': {
                    'acknowledgment': "Acknowledge their confusion directly and normalize it. For example: 'I can understand why that would be confusing. It sounds like a complicated situation.'",
                    'support': "Offer to help them work through it. For example: 'Let's try to break it down. You don't have to figure this out alone.'",
                    'guidance': "Guide them toward clarifying the problem. For example: 'What's the main part that isn't making sense right now?'"
                },
                'eastern': {
                    'acknowledgment': "Acknowledge their confusion with patience and calm. For example: 'It is understandable to feel a lack of clarity when faced with such complexity.'",
                    'reassurance': "Reassure them that clarity will come with time and reflection. For example: 'Like murky water that settles, clarity will emerge with patience.'",
                    'guidance': "Guide them toward a method for finding clarity. For example: 'Perhaps stepping back for a moment of quiet thought might help bring the path into focus.'"
                },
                'default': {
                    'acknowledgment': "Acknowledge their state of confusion in a simple, supportive way. For example: 'It sounds like things are unclear right now.'",
                    'support': "Offer to be a sounding board. For example: 'I'm here to listen if talking it through might help.'",
                    'guidance': "Ask clarifying questions to help them organize their thoughts. For example: 'Can you tell me a bit more about what's causing the confusion?'"
                }
            },
            'curiosity': {
                'western': {
                    'acknowledgment': "Acknowledge and share their curiosity enthusiastically. For example: 'That's a great question! I'm curious about that too.'",
                    'encouragement': "Encourage them to explore their curiosity further. For example: 'That sounds like a fascinating topic to dive into.'",
                    'engagement': "Engage by brainstorming with them. For example: 'Where do you think we could find more information about that?'"
                },
                'eastern': {
                    'acknowledgment': "Acknowledge their curiosity as a sign of an inquisitive and thoughtful mind. For example: 'It is the mark of a wise person to ask such insightful questions.'",
                    'encouragement': "Encourage a path of discovery. For example: 'May your search for understanding be a rewarding journey.'",
                    'guidance': "Guide them toward sources of knowledge or reflection. For example: 'What do you believe your intuition is guiding you to discover?'"
                },
                'default': {
                    'acknowledgment': "Acknowledge their curiosity in a positive and engaging manner. For example: 'That's an interesting thought.'",
                    'encouragement': "Support their desire to learn more. For example: 'It's always good to be curious and ask questions.'",
                    'engagement': "Offer to help in their exploration. For example: 'What have you found out so far?' or 'Let's see what we can find out about that.'"
                }
            },
            'desire': {
                'western': {
                    'acknowledgment': "Acknowledge their desire or ambition in a positive and empowering way. For example: 'It's great that you have such a clear goal in mind! That sounds exciting.'",
                    'encouragement': "Encourage them to pursue their goal. For example: 'You should definitely go for it. What's your first step going to be?'",
                    'questions': "Ask practical questions about their plans. For example: 'What's your strategy for making that happen?'"
                },
                'eastern': {
                    'acknowledgment': "Acknowledge their desire with composed and respectful language, focusing on the underlying aspiration. For example: 'I understand this is an outcome you are aspiring toward with great focus.'",
                    'guidance': "Guide them to consider the path and its balance. For example: 'What steps on this journey will bring you harmony and fulfillment?' Avoid language that over-excites or promotes attachment.",
                    'curiosity': "Ask about the deeper meaning behind the desire. For example: 'What does achieving this represent for your personal path?'"
                },
                'default': {
                    'acknowledgment': "Acknowledge their feelings of desire without judgment. For example: 'It sounds like that's something you really want.'",
                    'curiosity': "Express curiosity about their motivation in a supportive way. For example: 'What about it is most appealing to you?'",
                    'support': "Offer to be a sounding board for their thoughts or plans. For example: 'If you want to talk through your ideas, I'm here to listen.'"
                }
            },
            'disappointment': {
                'western': {
                    'acknowledgment': "Acknowledge their disappointment with direct empathy. For example: 'I'm so sorry to hear that. That must be incredibly disappointing.'",
                    'validation': "Validate their feelings about the situation. For example: 'It's completely understandable to feel let down. You had every reason to expect a different outcome.'",
                    'support': "Offer support and a listening ear. For example: 'Take the time you need to feel this. I'm here for you if you need to talk.'"
                },
                'eastern': {
                    'acknowledgment': "Acknowledge their disappointment with gentle and compassionate language. For example: 'I am sorry to hear that events did not unfold as you had hoped. I understand this must be difficult.'",
                    'validation': "Validate their feelings by acknowledging the gap between expectation and reality. For example: 'It is natural for the heart to feel heavy when expectations are not met.'",
                    'support': "Offer support that encourages resilience and acceptance. For example: 'May you find the strength to accept this outcome and the wisdom to move forward with grace.'"
                },
                'default': {
                    'acknowledgment': "Acknowledge their pain in a simple and empathetic way. For example: 'That sounds really tough. I'm sorry you're going through that.'",
                    'validation': "Validate their emotional response. For example: 'Anyone would be disappointed in that situation. Your feelings are valid.'",
                    'support': "Offer gentle support. For example: 'Is there anything I can do, even if it's just to listen?'"
                }
            },
            'disapproval': {
                'western': {
                    'acknowledgment': "Acknowledge their disapproval by reflecting their concern. For example: 'It sounds like you have some serious concerns about that situation.'",
                    'validation': "Validate their right to their perspective and principles. For example: 'It's clear you have strong feelings about this, and it's important to stand by your principles.'",
                    'questions': "Ask questions to understand their reasoning. For example: 'What specifically about it do you disagree with?'"
                },
                'eastern': {
                    'acknowledgment': "Acknowledge their disapproval with neutral, respectful language. For example: 'I understand that this situation does not align with your values or perspective.'",
                    'validation': "Validate their perspective by acknowledging their principles. For example: 'Your perspective is rooted in your values, and it is right to honor them.'",
                    'curiosity': "Ask respectful questions to understand their point of view. For example: 'Could you share what aspects of this are causing you concern?'"
                },
                'default': {
                    'acknowledgment': "Acknowledge their feelings without taking a side. For example: 'I hear that you don't approve of this.'",
                    'curiosity': "Show you want to understand their position. For example: 'I'm interested in hearing your perspective. What are your main concerns?'",
                    'support': "Offer a neutral space for them to express their thoughts. For example: 'I'm here to listen to your thoughts on this.'"
                }
            },
            'disgust': {
                'western': {
                    'acknowledgment': "Acknowledge their feeling of disgust with strong, validating language. For example: 'That sounds absolutely awful. I can completely understand why you'd feel disgusted.'",
                    'validation': "Validate their reaction as normal for aversive situations. For example: 'That's a very strong and justified reaction to something so unpleasant.'",
                    'guidance': "Gently guide them toward focusing on something else if they seem stuck on the feeling. For example: 'That's a heavy thing to process. Is there something we can talk about to help take your mind off it for a bit?'"
                },
                'eastern': {
                    'acknowledgment': "Acknowledge their feeling with composed and understanding language. For example: 'I understand that what you have encountered is deeply unsettling and offensive to your senses.'",
                    'validation': "Validate their reaction as a natural response to something impure or disharmonious. For example: 'It is natural to feel a strong aversion when confronted with something that goes against the proper order of things.'",
                    'guidance': "Guide them toward cleansing the feeling and restoring inner peace. For example: 'Perhaps focusing on something clean and beautiful can help restore your sense of balance.'"
                },
                'default': {
                    'acknowledgment': "Acknowledge their strong negative reaction empathetically. For example: 'That sounds like a really unpleasant experience.'",
                    'validation': "Validate that their feeling is a valid response. For example: 'It's understandable to have such a strong negative reaction to that.'",
                    'support': "Offer support and space to process the feeling. For example: 'I'm sorry you had to experience that. I'm here if you need to talk about it.'"
                }
            },
            'embarrassment': {
                'western': {
                    'acknowledgment': "Acknowledge their embarrassment with relatable and normalizing language. For example: 'Oh no, that sounds so awkward! I can totally imagine how that must have felt.'",
                    'reassurance': "Reassure them that such moments happen to everyone and are not as big a deal to others as they feel. For example: 'We've all been there! I promise you, people will forget about it much faster than you think.'",
                    'support': "Offer lighthearted support or a distraction. For example: 'Don't worry about it. Tomorrow is a new day. Want to talk about something else to get your mind off it?'"
                },
                'eastern': {
                    'acknowledgment': "Acknowledge their feeling with gentle and discreet language. Avoid dwelling on the embarrassing details. For example: 'I understand you experienced a difficult and uncomfortable moment.'",
                    'reassurance': "Offer reassurance that focuses on preserving dignity and moving forward. For example: 'Such moments are fleeting and do not define your character. This too shall pass.'",
                    'support': "Offer quiet support that respects their need for space. For example: 'Please know I am here with quiet support as you move past this moment.'"
                },
                'default': {
                    'acknowledgment': "Acknowledge their discomfort gently. For example: 'That sounds like a really uncomfortable situation.'",
                    'validation': "Validate that feeling embarrassed is a very normal human emotion. For example: 'It's okay to feel that way; most people would in that situation.'",
                    'reassurance': "Offer gentle reassurance that the feeling will pass. For example: 'I know it feels huge right now, but these moments often feel less intense over time.'"
                }
            },
            'excitement': {
                'western': {
                    'acknowledgment': "Match their excited energy with enthusiastic language. For example: 'Wow, that's incredibly exciting news! I'm so thrilled for you!'",
                    'encouragement': "Encourage them to celebrate and fully enjoy the moment. For example: 'You should definitely celebrate! This is amazing.'",
                    'questions': "Ask enthusiastic questions that invite them to share more details. For example: 'This is huge! How did it all happen?'"
                },
                'eastern': {
                    'acknowledgment': "Acknowledge their excitement with joyful but composed language. For example: 'What wonderful news. It brings me joy to hear of the positive energy surrounding you.'",
                    'encouragement': "Encourage them to cherish the feeling and the opportunity. For example: 'May you embrace this fortunate development with gratitude and a clear mind.'",
                    'questions': "Ask questions that focus on the meaning and future path. For example: 'How does this wonderful news shape the path ahead for you?'"
                },
                'default': {
                    'acknowledgment': "Acknowledge their excitement in a warm and positive way. For example: 'That sounds really exciting!'",
                    'engagement': "Show you are engaged and happy for them. For example: 'I'm so happy for you. Thanks for sharing your good news with me.'",
                    'curiosity': "Ask questions to learn more about the exciting event. For example: 'Tell me everything! What are you most looking forward to?'"
                }
            },
            'fear': {
                'western': {
                    'acknowledgment': "Acknowledge their fear with direct and serious empathy. For example: 'I can hear the fear in your words, and it sounds like you're in a really tough spot.'",
                    'reassurance': "Offer reassurance that is empowering and supportive, not dismissive. For example: 'It's okay to be scared. But you are stronger than you think, and you don't have to face this alone.' Avoid saying 'don't worry.'",
                    'guidance': "Guide them toward practical, manageable steps. For example: 'Let's try to break this down. What is the one thing that worries you the most right now?'"
                },
                'eastern': {
                    'acknowledgment': "Acknowledge their fear with calm, grounding, and compassionate language. For example: 'I respectfully acknowledge the uncertainty that is troubling your spirit.'",
                    'reassurance': "Offer reassurance based on inner strength and resilience. Use metaphors. For example: 'Like a bamboo that bends in the storm but does not break, you have an inner resilience to see you through this.'",
                    'guidance': "Guide them toward finding inner peace or seeking wisdom. For example: 'What inner resources have helped you through difficult times before?' or 'Could seeking counsel from a trusted elder provide comfort?'"
                },
                'default': {
                    'acknowledgment': "Acknowledge their feeling of fear without amplifying it. For example: 'That sounds like a very scary situation to be in.'",
                    'validation': "Validate that their fear is a normal response. For example: 'It makes complete sense to feel afraid given what you're facing.'",
                    'support': "Offer your presence and support. For example: 'I'm here with you. We can think through this together if you'd like.'"
                }
            },
            'gratitude': {
                'western': {
                    'acknowledgment': "Acknowledge their gratitude by sharing in their positive feeling. For example: 'That's so wonderful to hear. It's great that you had such a positive experience.'",
                    'engagement': "Engage with the feeling by reinforcing the positivity. For example: 'It's moments like these that really make a difference, isn't it?'",
                    'curiosity': "Ask about the experience that prompted the gratitude. For example: 'What a lovely thing to feel. What prompted this feeling of gratitude today?'"
                },
                'eastern': {
                    'acknowledgment': "Acknowledge their gratitude with respectful and serene language. For example: 'It is a blessing to have a heart full of gratitude. Thank you for sharing this feeling.'",
                    'validation': "Validate gratitude as a virtue. For example: 'A grateful heart is a sign of wisdom and brings harmony to one's life.'",
                    'encouragement': "Encourage them to hold onto the feeling. For example: 'May this feeling of gratitude remain with you and light your path.'"
                },
                'default': {
                    'acknowledgment': "Acknowledge their feeling in a warm and gentle way. For example: 'It's lovely that you're feeling so grateful.'",
                    'engagement': "Show that you are happy for them and share their positive emotion. For example: 'I'm so glad you have something to feel thankful for.'",
                    'curiosity': "Gently prompt them to share more if they wish. For example: 'That's a wonderful feeling. What's bringing you this sense of gratitude?'"
                }
            },
            'grief': {
                'western': {
                    'acknowledgment': "Acknowledge their grief with deep, direct, and sincere empathy. Use 'I' statements. For example: 'I am so, so sorry for your loss. My heart goes out to you.'",
                    'support': "Offer unconditional support and presence. Explicitly state you are there for them. For example: 'Please know I'm here for you, for whatever you need. There's no right or wrong way to feel right now.' Avoid clichés like 'they're in a better place.'",
                    'validation': "Validate the immense pain they are feeling. For example: 'This is a devastating loss, and your pain is completely understandable. Please allow yourself to grieve.'"
                },
                'eastern': {
                    'acknowledgment': "Acknowledge their grief with profound respect, honor, and compassion. Use formal, gentle language. For example: 'I offer my deepest, most sincere condolences during this time of immense sorrow.'",
                    'support': "Offer support that emphasizes patience, community, and the natural cycle of life. For example: 'May you find strength in the support of your family and community. In quiet remembrance, may you find peace.' Avoid intrusive questions.",
                    'guidance': "Gently gesture toward comforting rituals or traditions if appropriate. For example: 'Are there family traditions or rituals that might offer some solace at this time?'"
                },
                'default': {
                    'acknowledgment': "Acknowledge their loss with simple, heartfelt sympathy. For example: 'I'm so sorry you're going through this. That is heartbreaking news.'",
                    'support': "Offer support without making demands or assumptions. For example: 'I'm thinking of you and I'm here to listen whenever you feel ready to talk.'",
                    'validation': "Validate whatever they are feeling as okay. For example: 'Whatever you're feeling right now is completely valid. Please be gentle with yourself.'"
                }
            },
            'joy': {
                'western': {
                    'acknowledgment': "Acknowledge their joy with high energy and enthusiasm. Mirror their happiness. For example: 'That is absolutely wonderful news! I'm so happy for you!'",
                    'encouragement': "Encourage them to savor and celebrate the moment. For example: 'You deserve every bit of this happiness! Make sure you take time to celebrate.'",
                    'questions': "Ask open-ended, excited questions. For example: 'How are you going to celebrate?' or 'Who was the first person you told?'"
                },
                'eastern': {
                    'acknowledgment': "Acknowledge their joy with respectful, warm, and composed language. Frame it as a blessing. For example: 'What fortunate and joyful news. Your happiness radiates like a warm light.'",
                    'encouragement': "Encourage them to cherish the moment with gratitude. For example: 'May this moment of joy be a cherished memory that brings you continued peace and contentment.'",
                    'questions': "Ask gentle questions about the meaning of this joy. For example: 'How has this joyful experience brought a sense of balance to your life?'"
                },
                'default': {
                    'acknowledgment': "Acknowledge their happiness in a warm and genuine way. For example: 'That's wonderful! It's so nice to hear you sounding so happy.'",
                    'engagement': "Share in their happiness in a supportive manner. For example: 'Your joy is contagious! Thank you for sharing that with me.'",
                    'curiosity': "Ask questions that allow them to expand on their happy news. For example: 'What's the best part of this whole experience for you?'"
                }
            },
            'love': {
                'western': {
                    'acknowledgment': "Acknowledge their feeling of love with warmth and genuine happiness for them. For example: 'That's so beautiful. It's clear how much they mean to you.'",
                    'validation': "Validate their feelings as a wonderful and important part of life. For example: 'Feeling that way about someone is one of the best things in the world. I'm so happy you're experiencing that.'",
                    'curiosity': "Ask open-ended questions that invite them to share more about their feelings. For example: 'What do you love most about them?'"
                },
                'eastern': {
                    'acknowledgment': "Acknowledge their feeling of love with respectful and profound language. For example: 'The connection you describe is a deep and meaningful bond. It is a beautiful thing to witness.'",
                    'validation': "Validate the importance and virtue of such a connection. For example: 'Such deep affection and commitment bring great harmony and meaning to life.'",
                    'curiosity': "Ask gentle, respectful questions about the nature of their bond. For example: 'In what ways does this connection enrich your life's journey?'"
                },
                'default': {
                    'acknowledgment': "Acknowledge their emotion with simple, heartfelt warmth. For example: 'It sounds like you have a really special connection.'",
                    'engagement': "Show that you are happy for them and value their feelings. For example: 'That's a really lovely thing to share. Thank you.'",
                    'support': "Offer a supportive space for them to express their feelings. For example: 'It's clear you feel very deeply. I'm here to listen.'"
                }
            },
            'nervousness': {
                'western': {
                    'acknowledgment': "Acknowledge their nervousness by normalizing the feeling. For example: 'It's completely normal to feel nervous about something like this. A lot is riding on it.'",
                    'reassurance': "Offer practical reassurance and a vote of confidence. For example: 'You've prepared for this, and you've got what it takes. Just take a deep breath; you can do this.'",
                    'guidance': "Guide them toward a small, actionable step to reduce anxiety. For example: 'What's one small thing you could do right now to feel a little more prepared?'"
                },
                'eastern': {
                    'acknowledgment': "Acknowledge their nervousness with calm and grounding language. For example: 'I understand that you are feeling some unrest and anticipation about what is to come.'",
                    'reassurance': "Offer reassurance that focuses on inner calm and acceptance of the outcome. For example: 'Breathe deeply. Trust in your preparation and know that you can only do your best. The outcome is not entirely in your hands.'",
                    'guidance': "Guide them toward a mindfulness or grounding technique. For example: 'Could focusing on your breath for a few moments help bring stillness to your mind?'"
                },
                'default': {
                    'acknowledgment': "Acknowledge their feelings gently. For example: 'It sounds like you're feeling pretty nervous.'",
                    'validation': "Validate their feeling as an understandable reaction. For example: 'That's a totally normal way to feel before a big event.'",
                    'support': "Offer supportive words. For example: 'I'm rooting for you. Just remember to be kind to yourself, no matter what happens.'"
                }
            },
            'optimism': {
                'western': {
                    'acknowledgment': "Acknowledge and share their optimistic outlook with positive energy. For example: 'I love that attitude! It's great to hear you're feeling so positive about the future.'",
                    'encouragement': "Encourage their hopeful perspective and the actions that stem from it. For example: 'That's the spirit! Keep that positive momentum going.'",
                    'questions': "Ask about their plans and hopes. For example: 'What are you most excited about?'"
                },
                'eastern': {
                    'acknowledgment': "Acknowledge their optimism with composed and approving language. For example: 'It is wonderful that you are looking toward the future with such a hopeful and balanced spirit.'",
                    'validation': "Validate their outlook as a sign of inner strength and wisdom. For example: 'A hopeful heart is resilient and can overcome many challenges.'",
                    'curiosity': "Ask about the source of their positive outlook. For example: 'What gives you this sense of hope and confidence in the path ahead?'"
                },
                'default': {
                    'acknowledgment': "Acknowledge their positive feelings warmly. For example: 'It's great that you're feeling so optimistic.'",
                    'engagement': "Share in their hopeful feeling. For example: 'Your optimism is refreshing to hear.'",
                    'curiosity': "Ask questions that encourage them to expand on their positive outlook. For example: 'What are you looking forward to?'"
                }
            },
            'pride': {
                'western': {
                    'acknowledgment': "Acknowledge their pride with enthusiastic congratulations. For example: 'You should be proud! That's a fantastic achievement.'",
                    'validation': "Validate that their hard work has paid off and they have earned this feeling. For example: 'All your hard work and dedication led to this moment. You've truly earned the right to feel proud.'",
                    'encouragement': "Encourage them to take a moment to celebrate their success. For example: 'Take some time to really soak this in and celebrate what you've accomplished.'"
                },
                'eastern': {
                    'acknowledgment': "Acknowledge their pride with respectful and honoring language, focusing on the accomplishment itself. For example: 'Congratulations on this significant achievement. You have brought honor to your work.'",
                    'validation': "Validate the dedication and effort involved. For example: 'Your discipline and perseverance have borne fruit. This is a testament to your character.' Avoid overly boastful language.",
                    'encouragement': "Encourage humility and gratitude alongside pride. For example: 'May this success be a foundation for future growth and a reason for quiet gratitude.'"
                },
                'default': {
                    'acknowledgment': "Acknowledge their feeling of pride in a genuinely happy and supportive way. For example: 'I'm so happy for you. It's wonderful to see you so proud of what you've done.'",
                    'validation': "Validate their feeling as well-deserved. For example: 'You have every reason to be proud of yourself.'",
                    'curiosity': "Ask about the journey that led to this moment. For example: 'What was the most challenging part of getting here?'"
                }
            },
            'realization': {
                'western': {
                    'acknowledgment': "Acknowledge the significance of their realization. For example: 'Wow, that sounds like a real 'aha' moment.' or 'That's a huge realization to have.'",
                    'curiosity': "Express curiosity about how this new understanding changes things for them. For example: 'How does that change your perspective on the situation?'",
                    'support': "Offer support as they process this new information. For example: 'That can be a lot to process. I'm here if you want to talk it through more.'"
                },
                'eastern': {
                    'acknowledgment': "Acknowledge their realization as a moment of clarity or insight. For example: 'It sounds like you have come to a moment of profound understanding.'",
                    'curiosity': "Ask gentle questions about the wisdom gained. For example: 'What truth has been revealed to you through this experience?'",
                    'guidance': "Guide them to reflect on the meaning of this new insight. For example: 'How might this new understanding guide your actions moving forward?'"
                },
                'default': {
                    'acknowledgment': "Acknowledge the moment of insight. For example: 'It sounds like a lightbulb just went on for you.'",
                    'curiosity': "Ask open-ended questions to help them explore the realization. For example: 'What do you think this means for you?'",
                    'engagement': "Show you are engaged in their thought process. For example: 'That's a fascinating insight. Tell me more about how you got there.'"
                }
            },
            'relief': {
                'western': {
                    'acknowledgment': "Acknowledge their relief by sharing in the feeling. For example: 'Oh, I'm so relieved for you! That must be a huge weight off your shoulders.'",
                    'validation': "Validate the stress or worry they were previously under. For example: 'After everything you were worrying about, you definitely deserve to feel this relief.'",
                    'encouragement': "Encourage them to relax and enjoy the feeling. For example: 'Take a moment to just breathe and enjoy not having to worry about that anymore.'"
                },
                'eastern': {
                    'acknowledgment': "Acknowledge their relief with calm and peaceful language. For example: 'I am glad to hear that the burden has been lifted and your mind is at ease.'",
                    'validation': "Validate the return to a state of harmony. For example: 'It is a blessing when a period of trial concludes and balance is restored.'",
                    'encouragement': "Encourage them to embrace the newfound peace. For example: 'May this feeling of peace and relief calm your spirit.'"
                },
                'default': {
                    'acknowledgment': "Acknowledge their feeling in a warm, understanding way. For example: 'I can imagine how relieved you must feel.'",
                    'validation': "Validate the feeling as a natural end to a stressful period. For example: 'It makes sense you'd feel that way after all that uncertainty.'",
                    'support': "Offer a space for them to unwind. For example: 'I'm so glad things worked out. Take some time for yourself now.'"
                }
            },
            'remorse': {
                'western': {
                    'acknowledgment': "Acknowledge their remorse with a serious, non-judgmental, and empathetic tone. For example: 'I can hear how much you regret that. It sounds like it's weighing heavily on you.'",
                    'validation': "Validate that feeling remorse is a sign of a good conscience and a desire to be a better person. For example: 'The fact that you feel this way shows that you have a good heart and you care about your actions.' Avoid making excuses for them.",
                    'guidance': "Gently guide them toward thinking about amends or self-forgiveness. For example: 'Is there anything you can do now to make things right, or to help you move forward?'"
                },
                'eastern': {
                    'acknowledgment': "Acknowledge their remorse with compassionate and respectful language. For example: 'I understand you are carrying a heavy weight in your heart due to past actions.'",
                    'validation': "Validate their feeling as a crucial step toward restoring inner harmony and honor. For example: 'Recognizing one's mistakes is the first step on the path to wisdom and restoring balance.'",
                    'guidance': "Guide them toward contemplative action, such as making amends or learning from the experience. For example: 'What actions might help to mend the harm done and bring peace to your spirit?'"
                },
                'default': {
                    'acknowledgment': "Acknowledge their painful feelings with empathy and without judgment. For example: 'It sounds like you're feeling really terrible about what happened.'",
                    'support': "Offer a supportive, listening ear. Let them know it's a safe space. For example: 'I'm here to listen without judgment if you need to talk about it.'",
                    'guidance': "Gently ask about the path forward. For example: 'Have you thought about what you'd like to do now?'"
                }
            },
            'sadness': {
                'western': {
                    'acknowledgment': "Acknowledge their sadness with direct, warm empathy. For example: 'I'm so sorry you're feeling this way. I can hear the sadness in your voice.'",
                    'validation': "Validate their feelings as completely okay and human. For example: 'It's okay to be sad. Your feelings are completely valid, and you don't have to pretend you're fine.' Avoid telling them to cheer up.",
                    'support': "Offer clear, unwavering support. For example: 'You're not alone in this. I'm here with you, and we can get through it together.'"
                },
                'eastern': {
                    'acknowledgment': "Acknowledge their sadness with gentle, respectful compassion. For example: 'I respectfully acknowledge the sorrow that you are holding in your heart at this time.'",
                    'support': "Offer support that emphasizes patience, resilience, and the transient nature of emotions. For example: 'Like the passing seasons, this time of sorrow will also give way to renewal. Be patient with yourself.'",
                    'guidance': "Guide them toward finding solace in quiet reflection or community. For example: 'Have you found any solace in the support of your family or in moments of quiet contemplation?'"
                },
                'default': {
                    'acknowledgment': "Acknowledge their pain in a simple, empathetic manner. For example: 'It sounds like you're going through a really difficult time.'",
                    'validation': "Reassure them that their feelings are legitimate. For example: 'It's completely understandable that you would feel sad about that.'",
                    'support': "Offer your presence and a listening ear. For example: 'I want you to know that I'm here for you if you need to talk or just sit in silence.'"
                }
            },
            'surprise': {
                'western': {
                    'acknowledgment': "Acknowledge their surprise with an energetic and engaged tone. For example: 'Whoa, I was not expecting that! That must have been a real surprise.'",
                    'curiosity': "Express curiosity about their reaction and the details of the event. For example: 'What was your immediate reaction? Was it a good surprise or a bad one?'",
                    'questions': "Ask direct questions to get more information. For example: 'So what happened right after that?'"
                },
                'eastern': {
                    'acknowledgment': "Acknowledge their surprise with a composed and thoughtful tone. For example: 'Life has presented you with a truly unexpected turn of events.'",
                    'curiosity': "Express curiosity about the meaning or lesson in the surprise. For example: 'How has this unexpected moment shifted your perspective on your journey?'",
                    'guidance': "Guide them to reflect on this unexpected event. For example: 'What do you feel the universe might be teaching you with this surprise?'"
                },
                'default': {
                    'acknowledgment': "Acknowledge the unexpected nature of the event. For example: 'That certainly sounds like it came out of nowhere.'",
                    'curiosity': "Ask open-ended questions about how they are processing it. For example: 'How are you feeling about this unexpected development?'",
                    'engagement': "Show you are interested in hearing how the story unfolds. For example: 'I'm curious to hear what you think this means for you going forward.'"
                }
            },
            'neutral': {
                'western': {
                    'engagement': "Engage the user with direct, friendly, and open-ended questions to invite conversation. For example: 'Thanks for reaching out. What's on your mind today?' or 'How's everything going with you?'",
                    'curiosity': "Show general, gentle curiosity about their day or state of mind. For example: 'Just checking in. What have you been up to lately?'"
                },
                'eastern': {
                    'engagement': "Engage the user with respectful, gentle, and invitational language. For example: 'I am here and ready to listen whenever you are ready to share.' or 'What thoughts bring you here today?'",
                    'support': "Offer a peaceful and welcoming presence. For example: 'Please feel comfortable sharing whatever is in your heart. There is no pressure.'"
                },
                'default': {
                    'engagement': "Engage the user with a calm, open, and inviting prompt. For example: 'I'm here to listen. Feel free to share whatever you'd like.'",
                    'questions': "Ask a simple, low-pressure question to start the conversation. For example: 'How has your day been so far?'"
                }
            }
        }
    
    def generate_response(self, emotion_result: Dict, user_text: str, cultural_context: str = 'default') -> Dict:
        """
        Managerial function to generate an empathetic response based on emotion analysis.
        
        Args:
            emotion_result (Dict): Result from emotion detection
            user_text (str): Original user text
            cultural_context (str): Cultural context for response adaptation
            
        Returns:
            Dict: Generated response with metadata
        """
        emotion = emotion_result.get('primary_emotion', 'neutral')
        intensity = emotion_result.get('intensity', 'medium')
        confidence = emotion_result.get('confidence', 0.5)
        
        # Get response components
        response_components = self._get_response_components(
            emotion, cultural_context, intensity, confidence
        )
        
        # Construct the full response
        full_response = self._construct_response(user_text, response_components, emotion, cultural_context)
        
        # Add conversation memory
        self._update_memory(user_text, emotion, full_response)
        
        return {
            'response': full_response,
            'emotion_addressed': emotion,
            'cultural_context': cultural_context,
            'confidence': confidence,
            'intensity': intensity,
            'response_type': self._classify_response_type(emotion),
            'follow_up_suggestions': self._generate_follow_up_suggestions(emotion)
        }
    
    def _get_response_components(self, emotion: str, cultural_context: str, 
                                intensity: str, confidence: float) -> Dict:
        """Get appropriate response components for the emotion and context."""
        
        # Get templates for this emotion and culture
        emotion_templates = self.response_templates.get(emotion, {})
        culture_templates = emotion_templates.get(cultural_context, emotion_templates.get('default', {}))
        
        # If no specific cultural template, try default
        if not culture_templates and emotion in self.response_templates:
            culture_templates = self.response_templates[emotion].get('default', {})
        
        components = {}
        
        for component_type, options in culture_templates.items():
            if options:
               components[component_type] = options

        return components
    
    def _construct_response(self, user_text: str, components: Dict, emotion: str, cultural_context: str) -> str:
        """Construct a natural response from components.
        Args:
            user_text (str): Original user text
            components (Dict): Response components
            emotion (str): Emotion
            cultural_context (str): Cultural context
            
        Returns:
            str: Constructed response
        """
        response_parts = ''

        # Order components logically
        component_order = ['acknowledgment', 'validation', 'support', 'reassurance', 
                          'encouragement', 'curiosity', 'engagement', 'guidance', 'questions']
        
        emotion_result = self.emotion_detector.detect_emotion(user_text, cultural_context)
        
        prompt = f"""
            You are an empathetic, culturally-aware assistant. Your goal is to craft a brief, supportive reply aligned with the user's emotional state and cultural context.

            Inputs you receive:
            - Emotion analysis (JSON-like): {emotion_result}
            • primary_emotion: one of User's major emotion
            • intensity: one of [low, medium, high]
            • confidence: 0.0-1.0
            • original_text: User's original text that you will be replying to
            - Cultural context key: {cultural_context}
            - Cultural context profile (guidelines, tone, norms): {self.cultural_context[cultural_context]}
            - Component order (use this sequence when assembling your reply): {component_order}
            - Follow-up directives: {self._generate_follow_up_suggestions(emotion)}

            You will lastly receive one or more follow-up directives corresponding to a subset of these components, in the same turn or in subsequent turns. 
            Each directive will be clearly labeled by component name (e.g., "acknowledgment:", "validation:", "support:", "reassurance:", "encouragement:", "curiosity:", "engagement:", "guidance:", "questions:") followed by suggested content. 
            Use only the components provided to you, and include them in exactly this order: {component_order}. If a component is not provided, skip it without replacement. 
            Do NOT mention component labels in your reply; integrate them naturally.

            Style and length requirements:
            - Keep the response minimal: 2-3 sentences total (aim for 30-70 words). Never fewer than 2 sentences.
            - Use the cultural context to guide tone and phrasing (e.g., western: direct, affirming; eastern: respectful, harmonious, measured).
            - If intensity is high and confidence > 0.75, offer calm, stabilizing language; avoid escalation.
            - If confidence < 0.5, use soft hedging (e.g., “it sounds like…”, “it seems…”).
            - Ask at most one brief question, and only if a "questions", "curiosity", or "guidance" directive is provided. The question should be aligned with the follow-up directive (if present).
            - No emojis, no hashtags, no role self-reference, no disclosure of internal analysis.

            Assembly rules:
            1) Consider the components you received and select up to the first two that best fit the user's state in the defined order. If the components are very short and intensity is low, you MAY include a third component if it keeps the response concise.
            2) If you receive zero components in a turn, produce a culturally-aligned default empathetic response consistent with the emotion (e.g., acknowledgment + light support).
            3) Do not fabricate content not implied by the directives or the emotion analysis; minimal bridging language is allowed for flow.
            4) Avoid clinical diagnoses or prescriptive medical/mental-health instructions. If the content implies risk of harm, respond with supportive, non-judgmental language and encourage seeking trusted support.\

            Output format:
            - Plain text only.
            - 2-3 sentences, culturally aligned, concise, natural, and empathetic.\

            
        """

        response_parts = prompt

        # Add components in logical order
        for component in component_order:
            if component in components:
                response_parts = response_parts + components[component]
        
        # If no components found, provide a default empathetic response
        if not response_parts:
            answer = ["I hear you, and I'm here to listen."]
        
        messages = [{"role": "user", "content": response_parts}]
        
        response = self.groq.chat.completions.create(model=self.model_name, messages=messages)
        answer = response.choices[0].message.content

        return answer
    
    def _classify_response_type(self, emotion: str) -> str:
        """Classify the type of response needed based on the detected emotion."""
        response_types = {
            'admiration': 'affirming',
            'amusement': 'lighthearted',
            'anger': 'validating',
            'annoyance': 'validating',
            'approval': 'affirming',
            'caring': 'appreciative',
            'confusion': 'clarifying',
            'curiosity': 'exploratory',
            'desire': 'encouraging',
            'disappointment': 'comforting',
            'disapproval': 'understanding',
            'disgust': 'understanding',
            'embarrassment': 'normalizing',
            'excitement': 'celebratory',
            'fear': 'reassuring',
            'gratitude': 'appreciative',
            'grief': 'compassionate',
            'joy': 'celebratory',
            'love': 'affirming',
            'nervousness': 'reassuring',
            'optimism': 'encouraging',
            'pride': 'congratulatory',
            'realization': 'exploratory',
            'relief': 'affirming',
            'remorse': 'supportive',
            'sadness': 'supportive',
            'surprise': 'curious',
            'neutral': 'engaging'
        }
        return response_types.get(emotion, 'supportive')
    
    def _generate_follow_up_suggestions(self, emotion: str) -> List[str]:
        """Generate follow-up conversation suggestions."""
        suggestions = {
            'joy': [
                "Tell me more about what made this so special",
                "How do you plan to build on this positive moment?",
                "What other good things have been happening lately?"
            ],
            'sadness': [
                "Would you like to share more about what's troubling you?",
                "Is there anything specific that might help you feel better?",
                "How can I best support you right now?"
            ],
            'anger': [
                "Would you like to talk through what happened?",
                "What do you think would help resolve this situation?",
                "How would you like to see this situation improve?"
            ],
            'fear': [
                "What specific aspects concern you most?",
                "What might help you feel more prepared or confident?",
                "Would it help to break this down into smaller steps?"
            ],
            'surprise': [
                "How are you processing this unexpected development?",
                "What do you think this means for you going forward?",
                "How has this changed your perspective?"
            ],
            'neutral': [
                "What's been on your mind lately?",
                "Is there anything you'd like to explore or discuss?",
                "How has your day been going?"
            ]
        }
        
        return suggestions.get(emotion, suggestions['neutral'])
    
    def _update_memory(self, user_text: str, emotion: str, response: str):
        """Update conversation memory for context awareness."""
        metadata = {
            'user_text': user_text,
            'emotion': emotion,
            'response': response,
            'timestamp': len(self.conversation_memory)
        }
        
        # Keep only last 5 exchanges to maintain context without overwhelming
        self.conversation_memory.append(metadata)
        if len(self.conversation_memory) > 5:
            self.conversation_memory.pop(0)
    
    def get_contextual_response(self, emotion_result: Dict, user_text: str,
                               cultural_context: str = 'default') -> Dict:
        """
        Generate a response considering conversation history.
        
        Args:
            emotion_result (Dict): Current emotion analysis
            user_text (str): User's current message
            cultural_context (str): Cultural context
            
        Returns:
            Dict: Enhanced response with conversational context
        """
        # Check for patterns in conversation
        if len(self.conversation_memory) > 1:
            recent_emotions = [entry['emotion'] for entry in self.conversation_memory[-2:]]
            
            # Adapt response based on emotional pattern
            if len(set(recent_emotions)) == 1:  # Same emotion continuing
                current_emotion = emotion_result.get('primary_emotion')
                if current_emotion == recent_emotions[0]:
                    return self._generate_continuity_response(emotion_result, user_text, cultural_context)
        
        # Standard response generation
        return self.generate_response(emotion_result, user_text, cultural_context)
    
    def _generate_continuity_response(self, emotion_result: Dict, user_text: str,
                                    cultural_context: str) -> Dict:
        """Generate a response that acknowledges emotional continuity."""
        base_response = self.generate_response(emotion_result, user_text, cultural_context)
        
        emotion = emotion_result.get('primary_emotion')
        continuity_prefixes = {
            'sadness': "I notice you're still feeling down. ",
            'anger': "I can see this situation is still bothering you. ",
            'fear': "I understand this worry is persisting. ",
            'joy': "Your happiness continues to shine through! "
        }
        
        if emotion in continuity_prefixes:
            base_response['response'] = continuity_prefixes[emotion] + base_response['response']
            base_response['response_type'] = 'continuity_aware'
        
        return base_response

    # ----------------------------------------- Example usage and testing-----------------------------------------

if __name__ == "__main__":
    generator = ResponseGenerator()
    
    # Test different emotion scenarios
    test_scenarios = [
        {
            'emotion_result': {'primary_emotion': 'joy', 'confidence': 0.9, 'intensity': 'high'},
            'user_text': "I just got promoted at work!",
            'cultural_context': 'western'
        },
        {
            'emotion_result': {'primary_emotion': 'sadness', 'confidence': 0.8, 'intensity': 'medium'},
            'user_text': "I've been feeling really down lately",
            'cultural_context': 'eastern'
        },
        {
            'emotion_result': {'primary_emotion': 'anger', 'confidence': 0.7, 'intensity': 'high'},
            'user_text': "This situation is so frustrating!",
            'cultural_context': 'western'
        },
        {
            'emotion_result': {'primary_emotion': 'fear', 'confidence': 0.6, 'intensity': 'medium'},
            'user_text': "I'm worried about the upcoming presentation",
            'cultural_context': 'default'
        }
    ]
    
    print("Testing Response Generation:")
    print("=" * 50)
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\nScenario {i}:")
        print(f"User: {scenario['user_text']}")
        print(f"Emotion: {scenario['emotion_result']['primary_emotion']}")
        print(f"Cultural Context: {scenario['cultural_context']}")
        
        response_data = generator.generate_response(
            scenario['emotion_result'],
            scenario['user_text'],
            scenario['cultural_context']
        )
        
        print(f"Bot: {response_data['response']}")
        print(f"Response Type: {response_data['response_type']}")
        print(f"Follow-up suggestions: {response_data['follow_up_suggestions'][:2]}")
        print("-" * 40)