"""
Simple preprocesser for casefolding
"""
class CaseFolder:
    def __init__(self, lower=True):
        self.lower = lower
    
    def preprocess(self, text):
        if self.lower:
            return text.lower()
        else:
            return text.upper()