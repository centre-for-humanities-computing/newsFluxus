"""
Simple preprocesser for regex filtering
"""
import re

class RegxFilter:
    def __init__(self, pattern):
        self.pattern = re.compile(r"{}".format(pattern),flags=re.MULTILINE)
    
    def preprocess(self, text):
        return self.pattern.sub(" ", text)

        