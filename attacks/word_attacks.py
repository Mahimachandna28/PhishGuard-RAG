import random
from typing import List

SYNONYM_DICT = {
    'urgent': ['immediate', 'critical', 'pressing', 'time-sensitive'],
    'verify': ['confirm', 'validate', 'check', 'authenticate'],
    'account': ['profile', 'access', 'membership', 'portal'],
    'security': ['protection', 'safety', 'guard', 'defense'],
    'suspended': ['disabled', 'frozen', 'blocked', 'locked'],
    'password': ['passcode', 'credentials', 'login details', 'secret key'],
    'click': ['press', 'tap', 'follow', 'visit'],
    'bank': ['financial institution', 'credit union', 'banking portal']
}

class WordLevelAttacks:
    """
    Word-level synonym substitution attack simulator (TextFooler-style).
    """

    @classmethod
    def substitute_synonyms(cls, text: str, swap_prob: float = 0.3) -> str:
        words = text.split()
        new_words = []
        for word in words:
            word_lower = word.lower().strip('.,!?')
            if word_lower in SYNONYM_DICT and random.random() < swap_prob:
                syn = random.choice(SYNONYM_DICT[word_lower])
                new_words.append(syn)
            else:
                new_words.append(word)
        return " ".join(new_words)

    @classmethod
    def attack(cls, texts: List[str], swap_prob: float = 0.3) -> List[str]:
        return [cls.substitute_synonyms(t, swap_prob=swap_prob) for t in texts]
