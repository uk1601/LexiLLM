"""
Intent Manager for LexiLLM
Handle intent classification and related operations
"""

import time
import re
from typing import Dict, Any, Optional, Tuple, List
import traceback

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from ..schemas import IntentClassifier, DomainRelevance
from ..templates import create_intent_prompt, create_domain_relevance_prompt
from ..config import INTENT_CONFIG, CONVERSATION_CONFIG
from ..logger import get_logger
from ..exceptions import IntentClassificationError, TimeoutError

# Get logger
logger = get_logger()

class IntentManager:
    """
    Manages intent classification for LexiLLM.
    
    This class handles all aspects of determining the user's intent,
    including classification and follow-up detection.
    """
    
    def __init__(self, classifier_llm: ChatOpenAI):
        """
        Initialize the intent manager.
        
        Args:
            classifier_llm: LLM instance to use for classification
        """
        self.classifier_llm = classifier_llm
        self.intent_prompt = create_intent_prompt()
        self.domain_relevance_prompt = create_domain_relevance_prompt()
        self.setup_chains()
        
        # Record the last few intents for context-aware classification
        self.recent_intents = []
        self.max_recent_intents = 5
        
        logger.info("Enhanced intent manager initialized with semantic classification")
    
    def setup_chains(self):
        """Set up the classification chains."""
        try:
            # Intent classification using structured output
            self.intent_chain = self.intent_prompt | self.classifier_llm.with_structured_output(
                IntentClassifier,
                method="function_calling"
            )
            logger.debug("Intent classification chain created")
            
            # Domain relevance classification
            self.domain_relevance_chain = self.domain_relevance_prompt | self.classifier_llm.with_structured_output(
                DomainRelevance,
                method="function_calling"
            )
            logger.debug("Domain relevance classification chain created")
        except Exception as e:
            logger.error(f"Failed to set up classification chains: {str(e)}")
            raise IntentClassificationError(f"Failed to set up classification chains: {str(e)}")
    
    def check_domain_relevance(self, query: str) -> DomainRelevance:
        """
        Check if a query is related to LLMs using semantic understanding.
        
        Args:
            query: The user's message
            
        Returns:
            DomainRelevance object with relevance assessment
        """
        try:
            logger.debug(f"Checking domain relevance for: {query[:50]}...")
            # Use the domain relevance chain to determine if the query is LLM-related
            result = self.domain_relevance_chain.invoke({"query": query})
            
            # Log the result for debugging
            logger.info(f"Domain relevance: {result.is_relevant} with confidence {result.confidence}")
            if result.related_topics:
                logger.debug(f"Related topics: {result.related_topics}")
            
            return result
        except Exception as e:
            # If an error occurs, use a more conservative approach
            logger.error(f"Error in domain relevance check: {str(e)}")
            
            # Default to assuming it might be related with low confidence
            fallback = DomainRelevance(
                is_relevant=True,  # Assume it might be relevant
                confidence=0.5,    # But with low confidence
                related_topics=[],
                reasoning="Fallback due to error in domain relevance check"
            )
            return fallback
    
    def classify_intent(self, query: str, chat_history=None) -> Dict[str, Any]:
        """
        Determine the user's intent from their input using semantic understanding.
        
        Args:
            query: The user's message
            chat_history: Optional conversation history for context
            
        Returns:
            Dictionary with intent, confidence, topics, and related_topics
        """
        try:
            logger.debug(f"Classifying intent for query: {query[:50]}...")
            
            # First, check if the query is related to LLMs
            domain_relevance = self.check_domain_relevance(query)
            
            # If the query isn't LLM-related with sufficient confidence, return UNKNOWN
            if not domain_relevance.is_relevant and domain_relevance.confidence >= 0.6:
                logger.info(f"Query detected as non-LLM related: {query[:50]}...")
                return {
                    "intent": "UNKNOWN", 
                    "confidence": domain_relevance.confidence,
                    "related_topics": domain_relevance.related_topics,
                    "reasoning": domain_relevance.reasoning
                }
            
            # Proceed with intent classification for LLM-related queries
            # or queries with low confidence in non-relevance
            result = self.intent_chain.invoke({"query": query})
            
            # Store this intent in recent_intents for context-aware classification
            if result.intent != "UNKNOWN":
                self.recent_intents.append(result.intent)
                # Keep only the last max_recent_intents
                if len(self.recent_intents) > self.max_recent_intents:
                    self.recent_intents.pop(0)
            
            logger.info(f"Classified intent as {result.intent} with confidence {result.confidence}")
            
            # Return a comprehensive result
            return {
                "intent": result.intent, 
                "confidence": result.confidence,
                "reasoning": result.reasoning if hasattr(result, 'reasoning') else "",
                "topics": result.topics if hasattr(result, 'topics') else [],
                "related_topics": domain_relevance.related_topics
            }
        except Exception as e:
            logger.error(f"Intent classification error: {str(e)}")
            # Provide a more informative fallback with traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "intent": "UNKNOWN", 
                "confidence": 0.0, 
                "related_topics": [],
                "reasoning": f"Error in classification: {str(e)}"
            }
    
    def is_followup_question(self, message: str, chat_history) -> bool:
        """
        Determine if a message is a follow-up question to the previous topic.
        
        Args:
            message: The user's message
            chat_history: The conversation history
            
        Returns:
            Boolean indicating if this is a follow-up question
        """
        # If we don't have much context yet, assume it's not a follow-up
        if not chat_history or len(chat_history.messages) < 4:
            logger.debug(f"Message is not a follow-up (not enough context): {message}")
            return False
            
        try:
            # Get context from chat history
            recent_messages = chat_history.messages[-4:] if len(chat_history.messages) >= 4 else chat_history.messages
            context = "\n".join([f"{msg.type}: {msg.content[:100]}..." for msg in recent_messages])
            
            # Create a simple prompt for follow-up detection that uses the LLM's understanding
            prompt = f"""Given this conversation context:
            {context}
            
            Is the user's last message: "{message}"
            A follow-up to the previous conversation? Consider the meaning and context, not just keywords.
            
            Respond with YES or NO followed by a brief explanation.
            """
            
            # Use a timeout to avoid slowing down the conversation
            start_time = time.time()
            max_time = CONVERSATION_CONFIG.get("followup_detection_timeout", 2.0)
            
            if time.time() - start_time < max_time:
                response = self.classifier_llm.invoke(prompt).content.lower()
                is_followup = "yes" in response
                logger.debug(f"LLM determined message is{'' if is_followup else ' not'} a follow-up: {message}")
                
                # Extract reasoning if available
                reasoning = response.split("yes" if is_followup else "no", 1)[-1].strip()
                if reasoning:
                    logger.debug(f"Follow-up reasoning: {reasoning}")
                    
                return is_followup
            else:
                # Fall back to simple heuristics if timeout
                # Check for pronouns which often indicate follow-ups
                has_followup_pronouns = any(pronoun in message.lower() for pronoun in 
                                           ["it", "this", "that", "these", "those", "they", "them"])
                
                # Very short messages are often follow-ups
                short_message_threshold = CONVERSATION_CONFIG.get("short_message_threshold", 5)
                is_short = len(message.lower().strip().split()) < short_message_threshold
                
                # Questions are often follow-ups in an ongoing conversation
                is_question = message.strip().endswith("?")
                
                return has_followup_pronouns or is_short or is_question
                
        except Exception as e:
            logger.error(f"Error in follow-up detection: {str(e)}")
            
            # Fall back to simple heuristics if error
            # Short messages or questions are likely follow-ups
            is_short = len(message.lower().strip().split()) < CONVERSATION_CONFIG.get("short_message_threshold", 5)
            is_question = message.strip().endswith("?")
            return is_short or is_question

    def get_related_topics(self, topics: List[str]) -> List[str]:
        """
        Get a list of related LLM topics based on identified topics.
        
        Args:
            topics: List of identified topics
            
        Returns:
            List of related topics for suggestions
        """
        if not topics:
            # If no topics provided, return general LLM topics
            return [
                "LLM fundamentals", 
                "LLM architecture",
                "Prompt engineering",
                "Model fine-tuning",
                "LLM applications"
            ]
            
        try:
            # Create a prompt to generate related topics
            prompt = f"""Given these LLM-related topics: {', '.join(topics)}
            
            Generate 3-5 specific, closely related topics that the user might be interested in learning about.
            Focus on practical, specific topics rather than broad categories.
            
            Format your response as a comma-separated list of topics.
            """
            
            response = self.classifier_llm.invoke(prompt).content
            
            # Parse the comma-separated list
            related_topics = [topic.strip() for topic in response.split(",") if topic.strip()]
            
            # Ensure we limit the number of suggestions
            return related_topics[:5]
        except Exception as e:
            logger.error(f"Error generating related topics: {str(e)}")
            
            # Fallback to a simple approach
            topic_clusters = {
                "transformer": ["transformer architecture", "attention mechanisms", "encoder-decoder"],
                "embedding": ["vector embeddings", "word embeddings", "semantic search"],
                "vector": ["vector databases", "retrieval augmented generation", "semantic search"],
                "token": ["tokenization", "context windows", "token limits"],
                "prompt": ["prompt engineering", "few-shot prompting", "instruction tuning"],
                "fine-tuning": ["parameter-efficient fine-tuning", "PEFT", "LoRA adapter"],
                "llm": ["LLM architectures", "LLM applications", "LLM evaluation"],
                "gpt": ["GPT architecture", "instruction-tuned models", "OpenAI API"],
            }
            
            # Find relevant clusters for the provided topics
            related = []
            for topic in topics:
                topic_lower = topic.lower()
                for cluster_key, suggestions in topic_clusters.items():
                    if cluster_key in topic_lower:
                        related.extend(suggestions)
            
            # If no matches, return general suggestions
            if not related:
                related = ["prompt engineering", "model fine-tuning", "retrieval augmented generation"]
                
            # Return unique topics, up to 5
            return list(set(related))[:5]
