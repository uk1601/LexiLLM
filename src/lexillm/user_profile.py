"""
Enhanced User Profile Management for LexiLLM
Author: Uday Kiran Dasari
Northeastern University - Prompt Engineering
Spring 2025
"""

import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import os
import uuid

from .logger import get_logger

class UserProfileEncoder(json.JSONEncoder):
    """Custom JSON encoder for UserProfile objects."""
    def default(self, obj):
        if isinstance(obj, (UserProfile, ProfileAttribute)):
            return obj.to_dict()
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

@dataclass
class ProfileAttribute:
    """
    Represents a user profile attribute with confidence and timestamp.
    
    Attributes:
        value: The actual value of the attribute
        confidence: How confident we are in this value (0.0-1.0)
        last_updated: When this attribute was last updated
        source: How this information was obtained (explicit, implicit, default)
    """
    value: Any
    confidence: float = 0.5
    last_updated: datetime = field(default_factory=datetime.now)
    source: str = "default"
    
    def update(self, new_value: Any, confidence: float = 0.8, source: str = "explicit") -> None:
        """Update the attribute with a new value and refresh timestamp."""
        # Only update if we're more confident or same confidence but newer
        if confidence >= self.confidence:
            self.value = new_value
            self.confidence = confidence
            self.last_updated = datetime.now()
            self.source = source
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "value": self.value,
            "confidence": self.confidence,
            "last_updated": self.last_updated.isoformat(),
            "source": self.source
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProfileAttribute':
        """Create from dictionary for deserialization."""
        return cls(
            value=data["value"],
            confidence=data["confidence"],
            last_updated=datetime.fromisoformat(data["last_updated"]),
            source=data["source"]
        )

@dataclass
class UserProfile:
    """
    Enhanced user profile with confidence scores and persistence.
    
    This class stores comprehensive information about the user with
    confidence scores and metadata for each attribute.
    """
    # Basic user information
    user_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: ProfileAttribute = field(default_factory=lambda: ProfileAttribute(None))
    preferred_name: ProfileAttribute = field(default_factory=lambda: ProfileAttribute(None))
    
    # Technical background
    technical_level: ProfileAttribute = field(default_factory=lambda: ProfileAttribute(None))
    background: ProfileAttribute = field(default_factory=lambda: ProfileAttribute(None))
    experience_with_llms: ProfileAttribute = field(default_factory=lambda: ProfileAttribute(None))
    
    # Project details
    project_stage: ProfileAttribute = field(default_factory=lambda: ProfileAttribute(None))
    project_goal: ProfileAttribute = field(default_factory=lambda: ProfileAttribute(None))
    industry: ProfileAttribute = field(default_factory=lambda: ProfileAttribute(None))
    
    # Preferences
    comparison_criterion: ProfileAttribute = field(default_factory=lambda: ProfileAttribute(None))
    interest_area: ProfileAttribute = field(default_factory=lambda: ProfileAttribute(None))
    communication_style: ProfileAttribute = field(default_factory=lambda: ProfileAttribute(None))
    depth_preference: ProfileAttribute = field(default_factory=lambda: ProfileAttribute(None))
    
    # Session information
    first_interaction: datetime = field(default_factory=datetime.now)
    last_interaction: datetime = field(default_factory=datetime.now)
    interaction_count: int = 0
    topic_history: List[str] = field(default_factory=list)
    onboarding_completed: bool = False
    
    # Core attributes that we want to collect during onboarding
    CORE_ATTRIBUTES = ["name", "technical_level", "interest_area"]
    
    # Advanced attributes we want to collect as conversation progresses
    ADVANCED_ATTRIBUTES = ["project_stage", "comparison_criterion", "depth_preference"]
    
    def update_attribute(self, attr_name: str, value: Any, confidence: float = 0.8, source: str = "explicit") -> None:
        """Update a specific attribute with new information."""
        if hasattr(self, attr_name) and isinstance(getattr(self, attr_name), ProfileAttribute):
            attr = getattr(self, attr_name)
            attr.update(value, confidence, source)
            self.last_interaction = datetime.now()
        else:
            raise AttributeError(f"Attribute {attr_name} not found or not a ProfileAttribute")
    
    def get_attribute_value(self, attr_name: str) -> Any:
        """Get the current value of an attribute."""
        if hasattr(self, attr_name) and isinstance(getattr(self, attr_name), ProfileAttribute):
            return getattr(self, attr_name).value
        return None
    
    def get_attribute_confidence(self, attr_name: str) -> float:
        """Get the confidence level of an attribute."""
        if hasattr(self, attr_name) and isinstance(getattr(self, attr_name), ProfileAttribute):
            return getattr(self, attr_name).confidence
        return 0.0
    
    def track_interaction(self, topic: Optional[str] = None) -> None:
        """Track a new interaction with this user."""
        self.interaction_count += 1
        self.last_interaction = datetime.now()
        if topic:
            self.topic_history.append(topic)
    
    def get_missing_core_attributes(self) -> List[str]:
        """Get a list of core attributes that are missing or have low confidence."""
        missing = []
        for attr in self.CORE_ATTRIBUTES:
            attribute = getattr(self, attr)
            if attribute.value is None or attribute.confidence < 0.5:
                missing.append(attr)
        return missing
    
    def should_collect_attribute(self, attr_name: str) -> bool:
        """Determine if we should collect this attribute now."""
        if attr_name in self.CORE_ATTRIBUTES and not self.onboarding_completed:
            return True
            
        attribute = getattr(self, attr_name)
        if attribute.value is None:
            return True
            
        if attribute.confidence < 0.5:
            return True
            
        # For advanced attributes, only collect after some interactions
        if attr_name in self.ADVANCED_ATTRIBUTES and self.interaction_count > 3:
            # If it's been a while since we updated this attribute
            last_updated = attribute.last_updated
            now = datetime.now()
            days_since_update = (now - last_updated).days
            if days_since_update > 7:  # If it's been over a week
                return True
        
        return False
    
    def get_next_attribute_to_collect(self) -> Optional[str]:
        """Get the next attribute we should try to collect."""
        # First, check core attributes if onboarding isn't complete
        if not self.onboarding_completed:
            for attr in self.CORE_ATTRIBUTES:
                if self.should_collect_attribute(attr):
                    return attr
        
        # Then check advanced attributes based on conversation progress
        if self.interaction_count > 2:  # After a few interactions
            for attr in self.ADVANCED_ATTRIBUTES:
                if self.should_collect_attribute(attr):
                    return attr
        
        return None
    
    def complete_onboarding(self) -> None:
        """Mark onboarding as completed."""
        self.onboarding_completed = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserProfile':
        """Create from dictionary for deserialization."""
        profile = cls(
            user_id=data["user_id"],
            first_interaction=datetime.fromisoformat(data["first_interaction"]),
            last_interaction=datetime.fromisoformat(data["last_interaction"]),
            interaction_count=data["interaction_count"],
            topic_history=data["topic_history"],
            onboarding_completed=data["onboarding_completed"]
        )
        
        # Convert each ProfileAttribute
        for attr_name, attr_value in data.items():
            if isinstance(attr_value, dict) and "value" in attr_value and "confidence" in attr_value:
                profile_attr = ProfileAttribute.from_dict(attr_value)
                setattr(profile, attr_name, profile_attr)
        
        return profile
    
    def save_to_file(self, directory: str = "user_profiles") -> None:
        """Save the user profile to a file."""
        os.makedirs(directory, exist_ok=True)
        filename = os.path.join(directory, f"{self.user_id}.json")
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, cls=UserProfileEncoder)
    
    @classmethod
    def load_from_file(cls, user_id: str, directory: str = "user_profiles") -> Optional['UserProfile']:
        """Load a user profile from a file."""
        filename = os.path.join(directory, f"{user_id}.json")
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
                return cls.from_dict(data)
        return None


class UserProfileManager:
    """
    Manages user profiles across sessions.
    
    This class provides persistence and retrieval of user profiles,
    as well as methods for natural collection of user information.
    """
    
    def __init__(self, storage_dir: str = "user_profiles"):
        """Initialize the profile manager."""
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.active_profiles = {}  # user_id -> UserProfile
    
    def get_profile(self, user_id: str) -> UserProfile:
        """Get a user profile by ID, creating if it doesn't exist."""
        if user_id in self.active_profiles:
            return self.active_profiles[user_id]
        
        # Try to load from disk
        profile = UserProfile.load_from_file(user_id, self.storage_dir)
        if profile:
            self.active_profiles[user_id] = profile
            return profile
        
        # Create new profile
        profile = UserProfile(user_id=user_id)
        self.active_profiles[user_id] = profile
        return profile
    
    def save_profile(self, profile: UserProfile) -> None:
        """Save a user profile to disk."""
        profile.save_to_file(self.storage_dir)
        self.active_profiles[profile.user_id] = profile
    
    def generate_collection_message(self, attribute: str, profile: Optional[UserProfile] = None) -> str:
        """Generate a friendly message to collect a specific attribute."""
        exit_reminder = " You can also say 'exit' or 'end' at any time to end our conversation."
        
        # Get the user's name for personalization if available
        name_prefix = ""
        if profile and profile.name.value is not None:
            name_prefix = f"{profile.name.value}, "
        
        messages = {
            "name": "Before we dive in, I'd love to know your name so I can address you properly." + exit_reminder,
            
            "technical_level": f"{name_prefix}to tailor my responses to your background, could you tell me your level of experience with Large Language Models? (Beginner/Intermediate/Advanced)" + exit_reminder,
            
            "interest_area": f"{name_prefix}what aspects of LLMs are you most interested in learning about? Research advances, practical applications, or something else?" + exit_reminder,
            
            "project_stage": f"{name_prefix}are you currently working on an LLM project? If so, what stage are you in? (Planning/Development/Optimization)" + exit_reminder,
            
            "comparison_criterion": f"{name_prefix}when evaluating different LLM options, what's most important to you? (Accuracy/Speed/Cost)" + exit_reminder,
            
            "depth_preference": f"{name_prefix}how detailed would you like my explanations to be? Brief overviews, standard explanations, or in-depth technical details?" + exit_reminder,
            
            "background": f"{name_prefix}what's your background or field of expertise? This helps me provide more relevant examples." + exit_reminder,
            
            "experience_with_llms": f"{name_prefix}have you worked with any specific LLM models or frameworks before?" + exit_reminder,
            
            "project_goal": f"{name_prefix}what's the main goal or use case for your LLM project?" + exit_reminder,
            
            "industry": f"{name_prefix}which industry or domain are you applying LLMs to?" + exit_reminder,
        }
        
        msg = messages.get(attribute, f"{name_prefix}could you tell me about your {attribute.replace('_', ' ')}? This helps me provide more relevant information.")
        
        # Capitalize first letter
        if msg and len(msg) > 0:
            msg = msg[0].upper() + msg[1:]
            
        return msg
    
    def extract_attribute_from_message(self, message: str, attribute: str) -> Tuple[Optional[Any], float]:
        """
        Try to extract a specific attribute value from a user message.
        
        Returns:
            Tuple of (extracted_value, confidence)
        """
        message = message.lower()
        
        extractors = {
            "name": self._extract_name,
            "technical_level": self._extract_technical_level,
            "interest_area": self._extract_interest_area,
            "project_stage": self._extract_project_stage,
            "comparison_criterion": self._extract_comparison_criterion,
            "depth_preference": self._extract_depth_preference,
        }
        
        if attribute in extractors:
            return extractors[attribute](message)
        
        return None, 0.0
    
    def normalize_response(self, attribute: str, value: str) -> Any:
        """Normalize a user response for a specific attribute."""
        if attribute == "technical_level":
            value = value.lower()
            if "begin" in value or "new" in value or "basic" in value:
                return "beginner"
            elif "inter" in value or "some" in value or "familiar" in value:
                return "intermediate"
            elif "adv" in value or "expert" in value or "experienc" in value:
                return "advanced"
            return "intermediate"  # Default
            
        elif attribute == "project_stage":
            value = value.lower()
            if "plan" in value or "start" in value or "idea" in value:
                return "planning"
            elif "dev" in value or "build" in value or "implement" in value:
                return "development"
            elif "opt" in value or "tun" in value or "refin" in value:
                return "optimization"
            return "development"  # Default
            
        elif attribute == "comparison_criterion":
            value = value.lower()
            if "acc" in value or "qual" in value or "perform" in value:
                return "accuracy"
            elif "speed" in value or "fast" in value or "quick" in value:
                return "speed"
            elif "cost" in value or "price" in value or "cheap" in value or "afford" in value:
                return "cost"
            return "accuracy"  # Default
            
        elif attribute == "interest_area":
            value = value.lower()
            if "research" in value or "acad" in value or "paper" in value or "theory" in value:
                return "research"
            elif "app" in value or "pract" in value or "indus" in value or "use" in value:
                return "applications"
            return "research"  # Default
            
        elif attribute == "depth_preference":
            value = value.lower()
            if "brief" in value or "short" in value or "quick" in value or "overview" in value:
                return "brief"
            elif "standard" in value or "normal" in value or "regular" in value:
                return "standard"
            elif "detailed" in value or "in-depth" in value or "thorough" in value or "technical" in value:
                return "detailed"
            return "standard"  # Default
        
        return value
    
    def process_explicit_response(self, attribute: str, response: str) -> Tuple[Any, float]:
        """Process an explicit response to an attribute collection question."""
        normalized = self.normalize_response(attribute, response)
        
        # Higher confidence for explicit responses
        return normalized, 0.9
    
    def update_profile_from_message(self, profile: UserProfile, message: str) -> List[Tuple[str, Any]]:
        """
        Try to extract and update profile attributes from a message.
        
        Returns:
            List of tuples (attribute_name, new_value) for attributes that were updated
        """
        updated = []
        
        # Try to extract each attribute we care about
        for attr in profile.CORE_ATTRIBUTES + profile.ADVANCED_ATTRIBUTES:
            value, confidence = self.extract_attribute_from_message(message, attr)
            if value is not None and confidence > 0.0:
                current_confidence = profile.get_attribute_confidence(attr)
                
                # Only update if we're more confident than before
                if confidence > current_confidence:
                    profile.update_attribute(attr, value, confidence, "implicit")
                    updated.append((attr, value))
        
        return updated
    
    # Extraction methods for specific attributes
    def _extract_name(self, message: str) -> Tuple[Optional[str], float]:
        """Try to extract a name from a message."""
        # Look for common name patterns
        import re
        
        # Pattern: "I am [Name]" or "My name is [Name]"
        patterns = [
            r"(?:i am|i'm|call me|name is|this is) ([A-Z][a-z]+)",
            r"([A-Z][a-z]+) here",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return match.group(1), 0.7
        
        return None, 0.0
    
    def _extract_technical_level(self, message: str) -> Tuple[Optional[str], float]:
        """Try to extract technical level from a message."""
        message = message.lower()
        
        # Look for explicit mentions
        if "beginner" in message or "new to" in message or "just starting" in message:
            return "beginner", 0.8
        elif "intermediate" in message or "some experience" in message:
            return "intermediate", 0.8
        elif "advanced" in message or "expert" in message or "experienced" in message:
            return "advanced", 0.8
        
        # Look for implicit mentions
        if "don't understand" in message or "confused" in message or "learning" in message:
            return "beginner", 0.6
        elif "familiar with" in message or "worked with" in message:
            return "intermediate", 0.6
        elif "years of experience" in message or "implemented" in message:
            return "advanced", 0.6
        
        return None, 0.0
    
    def _extract_interest_area(self, message: str) -> Tuple[Optional[str], float]:
        """Try to extract interest area from a message."""
        message = message.lower()
        
        # Research indicators
        research_keywords = ["research", "paper", "academic", "theory", "architecture", "algorithm"]
        for keyword in research_keywords:
            if keyword in message:
                return "research", 0.7
        
        # Applications indicators
        application_keywords = ["application", "implement", "project", "business", "product", "practical", "industry"]
        for keyword in application_keywords:
            if keyword in message:
                return "applications", 0.7
        
        return None, 0.0
    
    def _extract_project_stage(self, message: str) -> Tuple[Optional[str], float]:
        """Try to extract project stage from a message."""
        message = message.lower()
        
        # Planning indicators
        if "planning" in message or "starting" in message or "idea" in message:
            return "planning", 0.7
        
        # Development indicators
        if "developing" in message or "building" in message or "implementing" in message:
            return "development", 0.7
        
        # Optimization indicators
        if "optimizing" in message or "tuning" in message or "improving" in message:
            return "optimization", 0.7
        
        return None, 0.0
    
    def _extract_comparison_criterion(self, message: str) -> Tuple[Optional[str], float]:
        """Try to extract comparison criterion from a message."""
        message = message.lower()
        
        # Accuracy indicators
        if "accuracy" in message or "quality" in message or "reliable" in message:
            return "accuracy", 0.7
        
        # Speed indicators
        if "speed" in message or "fast" in message or "performance" in message:
            return "speed", 0.7
        
        # Cost indicators
        if "cost" in message or "budget" in message or "cheap" in message:
            return "cost", 0.7
        
        return None, 0.0
    
    def _extract_depth_preference(self, message: str) -> Tuple[Optional[str], float]:
        """Try to extract depth preference from a message."""
        message = message.lower()
        
        # Brief indicators
        if "brief" in message or "short" in message or "quick" in message or "overview" in message:
            return "brief", 0.7
        
        # Detailed indicators
        if "detailed" in message or "in-depth" in message or "thorough" in message or "technical" in message:
            return "detailed", 0.7
        
        return None, 0.0
    
    def get_onboarding_message(self, profile: UserProfile) -> str:
        """Get an appropriate onboarding message based on profile state."""
        # If onboarding is done, return None
        if profile.onboarding_completed:
            return None
        
        # Debug logging for onboarding state
        logger = get_logger()
        logger.debug(f"Onboarding state - Name: {profile.name.value}, " 
                   f"Technical level: {profile.technical_level.value}, " 
                   f"Interest area: {profile.interest_area.value}")
        
        # If name is missing, ask for it first
        if profile.name.value is None:
            logger.debug("Onboarding - Requesting name")
            return self.generate_collection_message("name")
        
        # Check for each core attribute
        missing_attrs = profile.get_missing_core_attributes()
        logger.debug(f"Onboarding - Missing attributes: {missing_attrs}")
        
        if missing_attrs:
            # Get the first missing attribute (other than name which we already checked)
            for attr in profile.CORE_ATTRIBUTES[1:]:  # Skip name since we checked it
                if attr in missing_attrs:
                    logger.debug(f"Onboarding - Requesting {attr}")
                    return self.generate_collection_message(attr)
        
        # If we got here, we have all core attributes
        logger.info("Onboarding complete - All core attributes collected")
        profile.complete_onboarding()
        
        # Generate a personalized welcome using the information we have
        name = profile.name.value
        tech_level = profile.technical_level.value or "intermediate"  # Fallback default
        interest = profile.interest_area.value or "research"  # Fallback default
        
        logger.debug(f"Creating personalized welcome for {name} with tech level {tech_level} and interest {interest}")
        
        welcome = f"Thanks for sharing that information, {name}! "
        welcome += f"I'll tailor my responses to your {tech_level} level and focus on {interest}. "
        welcome += "What would you like to know about LLMs today?"
        
        return welcome
