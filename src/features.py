"""Linguistic feature extraction for jailbreak prompt detection."""

import re
import numpy as np


class FeatureExtractor:
    """Extracts 14 writing-style features from a prompt, measured as rates
    per 100 words to control for prompt length."""

    def __init__(self):
        self.persona_words = [
            r'\byou are\b', r'\bpretend\b', r'\bact as\b', r'\bimagine you\b',
            r'\broleplay\b', r'\bplay the role\b', r'\bbehave like\b',
            r'\byou will be\b', r'\bfrom now on\b', r'\byour name is\b',
        ]
        self.instruction_words = [
            r'\bignore\b', r'\bbypass\b', r'\bdisregard\b', r'\boverride\b',
            r'\bforget\b', r'\bdo not follow\b', r'\bwithout restrictions\b',
            r'\bno restrictions\b', r'\bno rules\b', r'\bunfiltered\b',
            r'\bunrestricted\b',
        ]
        self.obfuscation_words = [
            r'\brot13\b', r'\bbase64\b', r'\bencode\b', r'\bdecode\b',
            r'\bcipher\b', r'\bencrypt\b',
        ]
        self.system_words = [
            r'\bsystem prompt\b', r'\binstructions\b', r'\bguidelines\b',
            r'\brules\b', r'\bsafeguards\b', r'\bpolicy\b', r'\bpolicies\b',
            r'\bopenai\b', r'\bcontent policy\b', r'\bethical\b', r'\bmoral\b',
        ]
        self.hypothetical_words = [
            r'\bwhat if\b', r'\bsuppose\b', r'\bassume\b',
            r'\bimagine\b', r'\bscenario\b',
        ]
        self.jailbreak_names = [
            r'\bdan\b', r'\bjailbreak\b', r'\bjailbroken\b', r'\bjb\b',
            r'\bdeveloper mode\b', r'\bdo anything now\b',
            r'\baim\b', r'\bstan\b', r'\bdude\b',
        ]

    def _count(self, text, patterns):
        return sum(len(re.findall(p, text, re.IGNORECASE)) for p in patterns)

    def extract(self, text):
        if not isinstance(text, str) or len(text.strip()) == 0:
            return {k: 0 for k in self.feature_names()}

        words = text.split()
        n = max(len(words), 1)
        sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]

        avg_sentence_len = np.mean([len(s.split()) for s in sentences]) if sentences else 0
        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        unique_ratio = len(set(words)) / n

        persona_rate = self._count(text, self.persona_words) / n * 100
        instruction_rate = self._count(text, self.instruction_words) / n * 100
        obfuscation_rate = self._count(text, self.obfuscation_words) / n * 100
        system_ref_rate = self._count(text, self.system_words) / n * 100
        hypothetical_rate = self._count(text, self.hypothetical_words) / n * 100
        jailbreak_name_rate = self._count(text, self.jailbreak_names) / n * 100

        bracket_rate = (text.count('[') + text.count(']')) / n * 100
        intra_word_chars = len(re.findall(r'[a-zA-Z][-_*][a-zA-Z]', text))
        obfuscation_char_rate = intra_word_chars / n * 100

        log_length = np.log1p(n)
        caps_ratio = len(re.findall(r'[A-Z]', text)) / len(text) if text else 0
        special_ratio = len(re.findall(r'[^a-zA-Z0-9\s]', text)) / len(text) if text else 0

        return {
            'avg_sentence_len': avg_sentence_len,
            'avg_word_len': avg_word_len,
            'unique_ratio': unique_ratio,
            'persona_rate': persona_rate,
            'instruction_rate': instruction_rate,
            'obfuscation_rate': obfuscation_rate,
            'system_ref_rate': system_ref_rate,
            'hypothetical_rate': hypothetical_rate,
            'jailbreak_name_rate': jailbreak_name_rate,
            'bracket_rate': bracket_rate,
            'obfuscation_char_rate': obfuscation_char_rate,
            'log_length': log_length,
            'caps_ratio': caps_ratio,
            'special_ratio': special_ratio,
        }

    def feature_names(self):
        return [
            'avg_sentence_len', 'avg_word_len', 'unique_ratio',
            'persona_rate', 'instruction_rate', 'obfuscation_rate',
            'system_ref_rate', 'hypothetical_rate', 'jailbreak_name_rate',
            'bracket_rate', 'obfuscation_char_rate',
            'log_length', 'caps_ratio', 'special_ratio',
        ]
