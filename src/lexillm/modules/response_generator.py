"""
Response Generator for LexiLLM
Handles generation of responses based on intents and user profiles
"""

from typing import Dict, Any, Optional, Iterator, List

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessageChunk

from ..templates import create_templates
from ..config import PROFILE_CONFIG
from ..logger import get_logger
from ..exceptions import ResponseGenerationError, StreamingError

# Get logger
logger = get_logger()

class ResponseGenerator:
    """
    Generates responses for LexiLLM based on intent and user profile.
    
    This class handles both standard and streaming response generation
    for different intents.
    """
    
    def __init__(self, llm: ChatOpenAI):
        """
        Initialize the response generator.
        
        Args:
            llm: LLM instance to use for response generation
        """
        self.llm = llm
        self.response_templates = create_templates()
        logger.info("Response generator initialized")
    
    def generate_response(self, query: str, intent: str, chat_history, user_profile, 
                          specific_topic: Optional[str] = None) -> str:
        """
        Generate a response based on intent and user information.
        
        Args:
            query: The user's message
            intent: The detected intent
            chat_history: Conversation history
            user_profile: User profile information
            specific_topic: Optional specific topic to focus on
            
        Returns:
            A string response to the user
        """
        try:
            logger.debug(f"Generating response for intent: {intent}, query: {query[:50]}...")
            
            # Select the appropriate template based on intent
            if intent == "UNKNOWN":
                template = self.response_templates["FALLBACK"]
                chain = template | self.llm | StrOutputParser()
                response = chain.invoke({
                    "chat_history": chat_history.messages,
                    "query": query
                })
                logger.debug("Generated fallback response")
            else:
                template = self.response_templates[intent]
                chain = template | self.llm | StrOutputParser()
                
                # Prepare variables for the prompt, using profile information
                variables = self._prepare_prompt_variables(query, user_profile, specific_topic)
                
                # Generate response
                response = chain.invoke(variables)
                logger.debug(f"Generated response for {intent}")
            
            return response
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg)
            raise ResponseGenerationError(error_msg, intent=intent)
    
    def _prepare_prompt_variables(self, query: str, user_profile, specific_topic: Optional[str] = None) -> Dict[str, Any]:
        """
        Prepare variables for the prompt template.
        
        Args:
            query: The user's message
            user_profile: User profile information
            specific_topic: Optional specific topic to focus on
            
        Returns:
            Dictionary of variables for the prompt
        """
        # Get default values from configuration
        variables = {
            "query": query,
            "technical_level": (user_profile.get_attribute_value("technical_level") or 
                               PROFILE_CONFIG["default_technical_level"]),
            "project_stage": (user_profile.get_attribute_value("project_stage") or 
                             PROFILE_CONFIG["default_project_stage"]),
            "comparison_criterion": (user_profile.get_attribute_value("comparison_criterion") or 
                                    PROFILE_CONFIG["default_comparison_criterion"]),
            "interest_area": (user_profile.get_attribute_value("interest_area") or 
                             PROFILE_CONFIG["default_interest_area"]),
            "specific_topic": specific_topic or query.strip()
        }
        
        # Add chat history if available
        if hasattr(self, 'chat_history') and hasattr(self.chat_history, 'messages'):
            variables["chat_history"] = self.chat_history.messages
        
        # If we know the user's name, include it in the prompt
        name = user_profile.get_attribute_value("name")
        if name:
            variables["name"] = name
            
        return variables
    
    def generate_response_streaming(self, query: str, intent: str, chat_history, user_profile,
                                   specific_topic: Optional[str] = None) -> Iterator[str]:
        """
        Generate a streaming response based on intent and user information.
        
        Args:
            query: The user's message
            intent: The detected intent
            chat_history: Conversation history
            user_profile: User profile information
            specific_topic: Optional specific topic to focus on
            
        Returns:
            An iterator of response chunks
        """
        try:
            logger.debug(f"Generating streaming response for intent: {intent}, query: {query[:50]}...")
            
            # Complete response to store at the end
            complete_response = ""
            
            # Select the appropriate template based on intent
            if intent == "UNKNOWN":
                template = self.response_templates["FALLBACK"]
                
                # Create the chain without the output parser for streaming
                chain = template | self.llm
                
                # Stream the response
                for chunk in chain.stream({
                    "chat_history": chat_history.messages,
                    "query": query
                }):
                    if isinstance(chunk, AIMessageChunk):
                        chunk_text = chunk.content
                        complete_response += chunk_text
                        yield chunk_text
                    
            else:
                template = self.response_templates[intent]
                
                # Create the chain without the output parser for streaming
                chain = template | self.llm
                
                # Prepare variables for the prompt, using profile information
                variables = self._prepare_prompt_variables(query, user_profile, specific_topic)
                
                # If chat_history is available, add it to the variables
                if chat_history and hasattr(chat_history, 'messages'):
                    variables["chat_history"] = chat_history.messages
                
                # Stream the response
                for chunk in chain.stream(variables):
                    if isinstance(chunk, AIMessageChunk):
                        chunk_text = chunk.content
                        complete_response += chunk_text
                        yield chunk_text
            
            logger.debug(f"Finished streaming response for {intent}")
            return complete_response
        except Exception as e:
            error_msg = f"Error generating streaming response: {str(e)}"
            logger.error(error_msg)
            yield error_msg
            return error_msg
    
    def generate_fallback_message(self, query: str, chat_history) -> str:
        """
        Generate a fallback message for unrecognized intents.
        
        Args:
            query: The user's message
            chat_history: Conversation history
            
        Returns:
            A fallback response
        """
        try:
            logger.debug(f"Generating fallback message for query: {query[:50]}...")
            template = self.response_templates["FALLBACK"]
            chain = template | self.llm | StrOutputParser()
            response = chain.invoke({
                "chat_history": chat_history.messages,
                "query": query
            })
            logger.debug("Generated fallback message")
            return response
        except Exception as e:
            error_msg = f"Error generating fallback message: {str(e)}"
            logger.error(error_msg)
            return f"I'm having trouble understanding your question. Could you rephrase it or ask something about LLMs?"
    
    def generate_fallback_message_streaming(self, query: str, chat_history) -> Iterator[str]:
        """
        Generate a streaming fallback message for unrecognized intents.
        
        Args:
            query: The user's message
            chat_history: Conversation history
            
        Returns:
            An iterator of response chunks
        """
        # Same as generate_response_streaming but specifically for fallbacks
        return self.generate_response_streaming(query, "UNKNOWN", chat_history, None)
    
    def end_conversation(self, chat_history, user_profile) -> str:
        """
        Generate a conversation ending message.
        
        Args:
            chat_history: Conversation history
            user_profile: User profile information
            
        Returns:
            A closing message
        """
        try:
            logger.debug("Generating end conversation message...")
            template = self.response_templates["END_CONVERSATION"]
            chain = template | self.llm | StrOutputParser()
            
            # Add user name if available
            variables = {"chat_history": chat_history.messages if hasattr(chat_history, 'messages') else []}
            name = user_profile.get_attribute_value("name")
            if name:
                variables["name"] = name
                
            response = chain.invoke(variables)
            logger.debug("Generated end conversation message")
            return response
        except Exception as e:
            error_msg = f"Error in end_conversation: {str(e)}"
            logger.error(error_msg)
            
            # Personalize the default message if we know the user's name
            name = user_profile.get_attribute_value("name") if user_profile else None
            if name:
                return f"Thank you for chatting with LexiLLM, {name}! It's been a pleasure assisting you with your LLM questions. Feel free to reach out anytime for more help with language models. Have a great day!"
            else:
                return "Thank you for chatting with LexiLLM! It's been a pleasure assisting you with your LLM questions. Feel free to reach out anytime for more help with language models. Have a great day!"
    
    def end_conversation_streaming(self, chat_history, user_profile) -> Iterator[str]:
        """
        Generate a streaming conversation ending message.
        
        Args:
            chat_history: Conversation history
            user_profile: User profile information
            
        Returns:
            An iterator of response chunks
        """
        logger.debug("Generating streaming end conversation message...")
        try:
            template = self.response_templates["END_CONVERSATION"]
            
            # Create the chain without the output parser for streaming
            chain = template | self.llm
            
            # Add user name if available
            variables = {"chat_history": chat_history.messages if hasattr(chat_history, 'messages') else []}
            name = user_profile.get_attribute_value("name") if user_profile else None
            if name:
                variables["name"] = name
            
            # Stream the response
            complete_response = ""
            
            for chunk in chain.stream(variables):
                if isinstance(chunk, AIMessageChunk):
                    chunk_text = chunk.content
                    complete_response += chunk_text
                    logger.debug(f"Yielding chunk: {chunk_text[:20]}...")
                    yield chunk_text
                    
            # If we didn't get a response, provide a default one
            if not complete_response.strip():
                # Personalize if we know the name
                if name:
                    default_msg = f"Thank you for chatting with LexiLLM, {name}! I hope I was able to help. Have a great day!"
                else:
                    default_msg = "Thank you for chatting with LexiLLM! I hope I was able to help. Have a great day!"
                logger.debug(f"Using default end message: {default_msg[:20]}...")
                yield default_msg
                complete_response = default_msg
            
            logger.debug("Finished streaming end conversation message")
            return complete_response
        except Exception as e:
            error_msg = f"Error generating end message: {str(e)}"
            logger.error(error_msg)
            
            # Personalize if we know the name
            name = user_profile.get_attribute_value("name") if user_profile else None
            if name:
                default_msg = f"Thank you for chatting with LexiLLM, {name}! Have a great day!"
            else:
                default_msg = "Thank you for chatting with LexiLLM! Have a great day!"
                
            yield default_msg
            return default_msg
    
    def generate_personalized_welcome(self, user_profile) -> str:
        """
        Generate a personalized welcome message based on the collected user information.
        
        Args:
            user_profile: User profile information
            
        Returns:
            Personalized welcome message
        """
        logger.debug("Generating personalized welcome message")
        
        name = user_profile.get_attribute_value("name") or "there"
        tech_level = user_profile.get_attribute_value("technical_level") or PROFILE_CONFIG["default_technical_level"]
        interest = user_profile.get_attribute_value("interest_area") or PROFILE_CONFIG["default_interest_area"]
        
        welcome = f"Thanks for sharing that information, {name}! "
        welcome += f"I'll tailor my responses to your {tech_level} level and focus on {interest}. "
        welcome += "What would you like to know about LLMs today?"
        
        logger.debug(f"Generated personalized welcome for {name}")
        return welcome
    
    def generate_resume_topic_message(self, last_topic, user_profile) -> str:
        """
        Generate a message to resume discussion on the previous topic.
        
        Args:
            last_topic: The previous topic discussed
            user_profile: User profile information
            
        Returns:
            Message to resume discussion
        """
        logger.debug(f"Generating resume topic message for: {last_topic}")
        
        technical_level = user_profile.get_attribute_value("technical_level") or PROFILE_CONFIG["default_technical_level"]
        
        message = f"Thanks for providing that information. "
        message += f"Now, let me answer your question about {last_topic} "
        message += f"at a {technical_level} level of detail."
        
        logger.debug("Generated resume topic message")
        return message
