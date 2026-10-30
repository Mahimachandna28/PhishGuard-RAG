import unicodedata
import re
import urllib.parse

# Common homoglyph mappings (Cyrillic / Greek to Latin ASCII)
HOMOGLYPH_MAP = {
    'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 'у': 'y', 'х': 'x',
    'А': 'A', 'В': 'B', 'Е': 'E', 'К': 'K', 'М': 'M', 'Н': 'H', 'О': 'O',
    'Р': 'P', 'С': 'C', 'Т': 'T', 'Х': 'X', 'α': 'a', 'ο': 'o', 'ρ': 'p',
    'ϲ': 'c', 'ν': 'v', 'ѕ': 's', 'і': 'i', 'ј': 'j', 'ԁ': 'd', 'ԛ': 'q'
}

# Zero-width & invisible characters regex
ZERO_WIDTH_RE = re.compile(r'[\u200B-\u200D\uFEFF\u200E\u200F\u202A-\u202E\u00AD]')

# HTML tag stripping regex
HTML_TAG_RE = re.compile(r'<[^>]+>')

# URL extraction regex
URL_RE = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+')

class TextSanitizer:
    """
    Sanitize-then-dual-view text sanitizer for PhishGuard-RAG (Contribution C2).
    Applies NFKC normalization, homoglyph mapping, zero-width stripping, HTML removal, and URL canonicalization.
    """
    
    @staticmethod
    def remove_zero_width(text: str) -> str:
        return ZERO_WIDTH_RE.sub('', text)

    @staticmethod
    def map_homoglyphs(text: str) -> str:
        return "".join(HOMOGLYPH_MAP.get(char, char) for char in text)

    @staticmethod
    def normalize_unicode(text: str) -> str:
        return unicodedata.normalize('NFKC', text)

    @staticmethod
    def strip_html(text: str) -> str:
        return HTML_TAG_RE.sub(' ', text)

    @staticmethod
    def canonicalize_url(url: str) -> str:
        try:
            # Unescape hex encoding
            decoded = urllib.parse.unquote(url)
            # Remove redundant trailing slashes/ports if standard
            parsed = urllib.parse.urlparse(decoded)
            netloc = parsed.netloc.lower().split(':')[0]
            path = re.sub(r'/{2,}', '/', parsed.path)
            return f"{parsed.scheme}://{netloc}{path}"
        except Exception:
            return url

    @classmethod
    def canonicalize_urls_in_text(cls, text: str) -> str:
        def replace_url(match):
            return cls.canonicalize_url(match.group(0))
        return URL_RE.sub(replace_url, text)

    @classmethod
    def sanitize(cls, text: str, preserve_raw: bool = False) -> str:
        """
        Full canonicalization pipeline.
        """
        if not isinstance(text, str):
            return ""
        
        # 1. Zero-width character stripping
        cleaned = cls.remove_zero_width(text)
        
        # 2. Unicode NFKC normalization
        cleaned = cls.normalize_unicode(cleaned)
        
        # 3. Homoglyph mapping
        cleaned = cls.map_homoglyphs(cleaned)
        
        # 4. HTML tag stripping
        cleaned = cls.strip_html(cleaned)
        
        # 5. URL canonicalization
        cleaned = cls.canonicalize_urls_in_text(cleaned)
        
        # 6. Whitespace normalization
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned

    @classmethod
    def dual_view(cls, text: str):
        """
        Returns a tuple of (raw_text, canonical_text) for dual-view retrieval.
        """
        raw = re.sub(r'\s+', ' ', text).strip() if isinstance(text, str) else ""
        canonical = cls.sanitize(text)
        return raw, canonical
