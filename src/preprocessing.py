import re
import unicodedata
import emoji
import contractions
import pandas as pd

class SocialMediaPreprocessor:
    def __init__(self, lowercase=False):
        self.lowercase = lowercase

        # slang dictionary
        self.slang_dict = {
            "kys": "kill yourself",
            "wtf": "what the fuck",
            "idk": "i do not know",
            "imo": "in my opinion",
            "lmao": "laughing my ass off",
            "lol": "laughing out loud",
            "omg": "oh my god",
            "stfu": "shut the fuck up",
            "smh": "shaking my head",
            "btw": "by the way",
            "ur": "your",
            "u": "you"
        }

        # common profanity obfuscations
        self.profanity_patterns = {
            r"f[\W_]*u[\W_]*c[\W_]*k": "fuck",
            r"s[\W_]*h[\W_]*i[\W_]*t": "shit",
            r"b[\W_]*i[\W_]*t[\W_]*c[\W_]*h": "bitch",
            r"a[\W_]*s[\W_]*s": "ass",
            r"d[\W_]*i[\W_]*c[\W_]*k": "dick",
            r"f[\W_]*a[\W_]*g": "fag"
        }

        # leetspeak substitutions
        self.leet_map = {
            "0": "o",
            "1": "i",
            "3": "e",
            "4": "a",
            "5": "s",
            "7": "t",
            "@": "a",
            "$": "s"
        }

        self.url_pattern = re.compile(r"https?://\S+|www\.\S+")
        self.mention_pattern = re.compile(r"@\w+")
        self.hashtag_pattern = re.compile(r"#(\w+)")
        self.repeat_pattern = re.compile(r"(.)\1{2,}")
        self.whitespace_pattern = re.compile(r"\s+")
        self.leet_token_pattern = re.compile(r"[a-zA-Z]*[0-9@$]+[a-zA-Z]+")

    def normalize_unicode(self, text):
        return unicodedata.normalize("NFKC", text)

    def replace_urls(self, text):
        return self.url_pattern.sub("", text)

    def replace_mentions(self, text):
        return self.mention_pattern.sub("", text)

    def normalize_hashtags(self, text):
        return self.hashtag_pattern.sub(r"\1", text)

    def convert_emojis(self, text):
        return emoji.demojize(text, delimiters=(" ", " "))

    def expand_contractions(self, text):
        return contractions.fix(text)

    def normalize_repeated_chars(self, text):
        return self.repeat_pattern.sub(r"\1\1", text)

    def normalize_slang(self, text):
        words = text.split()
        normalized = []
        for word in words:
            key = word.lower()
            if key in self.slang_dict:
                normalized.append(self.slang_dict[key])
            else:
                normalized.append(word)
        return " ".join(normalized)

    def normalize_leetspeak(self, text):

        tokens = text.split()
        normalized_tokens = []

        for token in tokens:

            # skip pure numbers
            if token.isdigit():
                normalized_tokens.append(token)
                continue

            # only normalize suspicious tokens
            if self.leet_token_pattern.search(token):

                new_token = token
                for k, v in self.leet_map.items():
                    new_token = new_token.replace(k, v)

                normalized_tokens.append(new_token)

            else:
                normalized_tokens.append(token)

        return " ".join(normalized_tokens)

    def normalize_profanity(self, text):
        for pattern, replacement in self.profanity_patterns.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def normalize_whitespace(self, text):
        return self.whitespace_pattern.sub(" ", text).strip()

    def preprocess(self, text):

        text = self.normalize_unicode(text)
        text = self.replace_urls(text)          # now remove
        text = self.replace_mentions(text)      # now remove
        text = self.normalize_hashtags(text)
        text = self.convert_emojis(text)

        text = self.expand_contractions(text)
        # text = self.normalize_slang(text)

        # text = self.normalize_leetspeak(text)
        # text = self.normalize_profanity(text)

        text = self.normalize_repeated_chars(text)

        if self.lowercase:
            text = text.lower()

        text = self.normalize_whitespace(text)

        return text
    
    
def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning to the 'text' column, drop empty rows and duplicates."""
    clean_tweet = SocialMediaPreprocessor(lowercase=False).preprocess
    df = df.copy()
    df["text"] = df["text"].apply(clean_tweet)
    # df = df[df["text"].str.strip().astype(bool)].reset_index(drop=True)
    # df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return df