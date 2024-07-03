from dataclasses import dataclass
from typing import List, Any, Optional


@dataclass
class UserState:
    token_count: int = 0
    begun_tutoring: bool = False
    topic_chosen: bool = False
    tutor_conversation: List[Any] = None
    articles: List[Any] = None
    questions: Optional[Any] = None
    content: Optional[Any] = None
    topic: Optional[Any] = None
    user_input_buffer: str = ""

    def __post_init__(self):
        # Initialize lists if they are not provided
        if self.tutor_conversation is None:
            self.tutor_conversation = []
        if self.articles is None:
            self.articles = []