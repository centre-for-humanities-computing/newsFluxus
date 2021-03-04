'''Automatically generate regex patterns for big queries.
For details, see the docstring of `RegexCompiler`.


Usage example
-------------
Say you want to match mentions of Czech things in Danish text.
    
    Your target words are 'tjekkiet', 'tjekkisk', 'bøhmen', 'bøhmisk'.
    To minimize false positives (like 'tjekke ind' or 'tjekkoslovakiet'), you only want
    mentions where targets appear as whole words.

    The call is:
    ```
    cz_pattern = RegexCompiler(
        query_terms=['tjekkiet', 'tjekkisk', 'bøhmen', 'bøhmisk'],
        query_pattern=None,
        whole_words=True,
        flags=re.IGNORECASE,
    )
    ```

    this generates a trie pattern:
    `re.compile(r'\b(?:bøhm(?:en|isk)|tjekki(?:et|sk))\b', re.IGNORECASE|re.UNICODE)`
'''

import re


class Trie():
    """Regex::Trie in Python. Creates a Trie out of a list of words. 
    The trie can be exported to a Regex pattern.
    The corresponding Regex should match much faster than a simple Regex union."""

    def __init__(self):
        self.data = {}

    def add(self, word):
        ref = self.data
        for char in word:
            ref[char] = char in ref and ref[char] or {}
            ref = ref[char]
        ref[''] = 1

    def dump(self):
        return self.data

    def quote(self, char):
        return re.escape(char)

    def _pattern(self, pData):
        data = pData
        if "" in data and len(data.keys()) == 1:
            return None

        alt = []
        cc = []
        q = 0
        for char in sorted(data.keys()):
            if isinstance(data[char], dict):
                try:
                    recurse = self._pattern(data[char])
                    alt.append(self.quote(char) + recurse)
                except:
                    cc.append(self.quote(char))
            else:
                q = 1
        cconly = not len(alt) > 0

        if len(cc) > 0:
            if len(cc) == 1:
                alt.append(cc[0])
            else:
                alt.append('[' + ''.join(cc) + ']')

        if len(alt) == 1:
            result = alt[0]
        else:
            result = "(?:" + "|".join(alt) + ")"

        if q:
            if cconly:
                result += "?"
            else:
                result = "(?:%s)?" % result
        return result

    def pattern(self):
        return self._pattern(self.dump())


class RegexCompiler:
    '''Automatically compiles a regex pattern for big queries.

    Three modes of operation:
    a) compile a regex pattern from only `query_terms`
        If you know the words you wish to match, but now the regex pattern. 
        RegexCompiler will generate an efficient trie-based pattern.

    b) -||- from only `query_pattern`
        If you already know your pattern, you can save yourself the trouble
        and compile the pattern without needing to use RegexCompiler.

    c) -||- form both `query_terms` and `query_pattern`
        If you have words you wish to match and also a regex pattern to match something else.
        In this case RegexCompiler will combine a) and b) into a single pattern.
        Result will match anything specified in `query_terms` OR `query_pattern`

    Parameters
    ----------
    query_terms : list|str, optional
        List of words that you wish to match with regex.
        Positive match = any word found.

    query_pattern : str, optional
        String from which a regex pattern will be compiled without changes.

    whole_words : bool, optional
        Should , by default False

    flags : [type], optional
        Regex flags to use for compilation, by default None
    '''
    def __init__(self,
        query_terms=None, query_pattern=None,
        whole_words=False, flags=None) -> None:
        
        if query_terms and query_pattern:
            # compile both. Matched = query_terms OR query_pattern
            query_terms = self.validate_query_list(query_terms)
            terms_trie = self.trie_regex_from_words(query_terms)

            if whole_words:
                terms_trie = self.add_word_bound(terms_trie)

            # re.compile(r"\b" + trie.pattern() + r"\b", re.IGNORECASE)
            self.pattern = re.compile('(%s|%s)' % (terms_trie, query_pattern), flags)

        elif query_terms and not query_pattern:
            # compile one.
            query_terms = self.validate_query_list(query_terms)
            terms_trie = self.trie_regex_from_words(query_terms)

            if whole_words:
                terms_trie = self.add_word_bound(terms_trie)

            self.pattern = re.compile(terms_trie, flags)

        elif query_pattern and not query_terms:
            # compile one.
            self.pattern = re.compile(query_pattern, flags)


    @staticmethod
    def add_word_bound(pattern_str) -> str:
        '''Adds word boundary assertion to input
        '''
        return r'\b' + pattern_str + r'\b'


    @staticmethod
    def validate_query_list(query_list, escape=True) -> list:
        '''[summary]

        Parameters
        ----------
        query_list : list|str
            Words to compile a trie from.
        escape : bool, optional
            Escape regex special characters? By default True.

        Returns
        -------
        list
            List of strings to run through the Trie.
        '''
        # fix datatypes
        if isinstance(query_list, str):
            query_list = [query_list]
        
        if escape:
            esaped_query = [re.escape(word) for word in query_list]

            try:
                set(esaped_query) != set(query_list)
            except:
                Warning(
                    'Query_list contains regex special characters \
                    which will be escaped! Matching will not work the way you expect.')
                    
        return query_list


    @staticmethod
    def trie_regex_from_words(words) -> str:
        '''Make a tree-like regex pattern for efficient handling of a long query list.

        Parameters
        ----------
        words : list
            Contains words to match without regex special characters

        Returns
        -------
        str
            Non-compiled tire-pattern.
        '''
        trie = Trie()
        for word in words:
            trie.add(word)
        return trie.pattern()
    
    def __repr__(self):
        return repr(self.pattern)
