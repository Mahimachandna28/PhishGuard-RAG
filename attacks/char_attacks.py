import random
import re
from typing import List

# Character substitution map for homoglyphs
HOMOGLYPH_INJECT_MAP = {
    'a': 'а', 'e': 'е', 'o': 'о', 'p': 'р', 'c': 'с', 'y': 'у', 'x': 'х',
    'A': 'А', 'B': 'В', 'E': 'Е', 'K': 'К', 'M': 'М', 'H': 'Н', 'O': 'О'
}

ZERO_WIDTH_CHARS = ['\u200B', '\u200C', '\u200D', '\uFEFF']

class CharLevelAttacks:
    """
    Character-level adversarial attack generators (Section 10).
    Generates homoglyph injection, zero-width insertion, typos, and URL obfuscation.
    """

    @staticmethod
    def inject_homoglyphs(text: str, swap_prob: float = 0.3) -> str:
        res = []
        for char in text:
            if char in HOMOGLYPH_INJECT_MAP and random.random() < swap_prob:
                res.append(HOMOGLYPH_INJECT_MAP[char])
            else:
                res.append(char)
        return "".join(res)

    @staticmethod
    def inject_zero_width(text: str, insert_prob: float = 0.2) -> str:
        res = []
        for char in text:
            res.append(char)
            if random.random() < insert_prob:
                res.append(random.choice(ZERO_WIDTH_CHARS))
        return "".join(res)

    @staticmethod
    def inject_typos(text: str, typo_prob: float = 0.05) -> str:
        res = []
        for char in text:
            if char.isalpha() and random.random() < typo_prob:
                # Swaps or duplicates character
                res.append(char * 2 if random.random() < 0.5 else char.swapcase())
            else:
                res.append(char)
        return "".join(res)

    @staticmethod
    def obfuscate_urls(text: str) -> str:
        # Obfuscates URLs using hex encoding or subdomains
        def replace_url(match):
            url = match.group(0)
            return url.replace('http://', 'http://%77%77%77.').replace('.', '%2e')
        return re.sub(r'https?://[^\s<>"]+', replace_url, text)

    @classmethod
    def attack(cls, texts: List[str], attack_type: str = "combined") -> List[str]:
        attacked = []
        for t in texts:
            if attack_type == "homoglyph":
                attacked.append(cls.inject_homoglyphs(t))
            elif attack_type == "zero_width":
                attacked.append(cls.inject_zero_width(t))
            elif attack_type == "url_obfuscation":
                attacked.append(cls.obfuscate_urls(t))
            elif attack_type == "combined":
                t1 = cls.inject_homoglyphs(t, swap_prob=0.2)
                t2 = cls.inject_zero_width(t1, insert_prob=0.15)
                attacked.append(cls.obfuscate_urls(t2))
            else:
                attacked.append(t)
        return attacked
