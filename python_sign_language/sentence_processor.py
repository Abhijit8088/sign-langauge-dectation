from typing import List
import re

class SentenceProcessor:
    def __init__(self):
        self.current_signs: List[str] = []
        self.sentence_buffer: List[str] = []
        self.last_sign = None
        
    def update_sentence(self, new_sign: str) -> str:
        """
        Updates the current sentence with a new detected sign.
        Implements basic grammar rules and sentence structure.
        """
        if new_sign == self.last_sign:
            return " ".join(self.sentence_buffer)
            
        self.last_sign = new_sign
        self.current_signs.append(new_sign)
        
        # Process signs into a grammatically correct sentence
        if self._is_sentence_complete():
            sentence = self._process_signs_to_sentence()
            self.sentence_buffer.append(sentence)
            self.current_signs = []
            
            # Keep only last 3 sentences in buffer
            if len(self.sentence_buffer) > 3:
                self.sentence_buffer.pop(0)
                
        return " ".join(self.sentence_buffer)
        
    def _is_sentence_complete(self) -> bool:
        """
        Determines if the current signs form a complete sentence.
        """
        # Implement your sentence completion logic here
        # This is a simple example - you should expand based on your needs
        return len(self.current_signs) >= 3 or "." in self.current_signs
        
    def _process_signs_to_sentence(self) -> str:
        """
        Converts a sequence of signs into a grammatically correct sentence.
        """
        # Join signs into basic sentence
        sentence = " ".join(self.current_signs)
        
        # Capitalize first letter
        sentence = sentence.capitalize()
        
        # Add period if missing
        if not sentence.endswith((".", "!", "?")):
            sentence += "."
            
        return sentence