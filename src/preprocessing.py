import re
import string

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langdetect import detect, LangDetectException
import contractions
import pandas as pd

# Ensure required NLTK data is available
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

_stop_words = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()


def strip_emoji(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", text)


def strip_all_entities(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r"\r|\n", " ", text.lower()) # Converts all characters to lowercase. Replaces newline (\n) and carriage return (\r) with spaces.
    text = re.sub(r"(?:\@|https?\://)\S+", "", text) # remove mentions and URLs
    text = re.sub(r"[^\x00-\x7f]", "", text)    # remove non-ASCII characters
    table = str.maketrans("", "", string.punctuation)   # Deletes all punctuation characters: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    text = text.translate(table)
    text = " ".join(word for word in text.split() if word not in _stop_words) # remove stopwords
    return text


def clean_hashtags(tweet: str) -> str:
    if not isinstance(tweet, str):
        tweet = str(tweet)
    new_tweet = re.sub(r"(\s+#[\w-]+)+\s*$", "", tweet).strip()
    new_tweet = re.sub(r"#([\w-]+)", r"\1", new_tweet).strip()
    return new_tweet


def filter_chars(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    return " ".join(
        "" if ("$" in word) or ("&" in word) else word for word in text.split()
    )


def remove_mult_spaces(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    return re.sub(r"\s\s+", " ", text)


def filter_non_english(text: str) -> str:
    try:
        lang = detect(text)
    except LangDetectException:
        lang = "unknown"
    return text if lang == "en" else ""


def expand_contractions_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    return contractions.fix(text)


def remove_numbers(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    return re.sub(r"\d+", "", text)


def lemmatize(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    words = word_tokenize(text)
    lemmatized_words = [_lemmatizer.lemmatize(word) for word in words]
    return " ".join(lemmatized_words)


def remove_short_words(text: str, min_len: int = 2) -> str:
    if not isinstance(text, str):
        text = str(text)
    words = text.split()
    long_words = [word for word in words if len(word) >= min_len]
    return " ".join(long_words)


def replace_elongated_words(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    regex_pattern = r"\b(\w+)((\w)\3{2,})(\w*)\b"
    return re.sub(regex_pattern, r"\1\3\4", text)


def remove_repeated_punctuation(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    return re.sub(r"[\?\.\!]+(?=[\?\.\!])", "", text)


def remove_extra_whitespace(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    return " ".join(text.split())


def remove_url_shorteners(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    return re.sub(
        r"(?:http[s]?://)?(?:www\.)?(?:bit\.ly|goo\.gl|t\.co|tinyurl\.com|"
        r"tr\.im|is\.gd|cli\.gs|u\.nu|url\.ie|tiny\.cc|alturl\.com|ow\.ly|"
        r"bit\.do|adoro\.to)\S+",
        "",
        text,
    )


def remove_spaces_tweets(tweet: str) -> str:
    if not isinstance(tweet, str):
        tweet = str(tweet)
    return tweet.strip()


def remove_short_tweets(tweet: str, min_words: int = 3) -> str:
    if not isinstance(tweet, str):
        tweet = str(tweet)
    words = tweet.split()
    return tweet if len(words) >= min_words else ""


def clean_tweet(tweet: str) -> str:
    """Apply the full text cleaning pipeline to a single tweet."""
    if not isinstance(tweet, str):
        tweet = str(tweet)
    # tweet = filter_non_english(tweet)
    tweet = strip_emoji(tweet)
    tweet = expand_contractions_text(tweet)
    tweet = strip_all_entities(tweet)
    tweet = clean_hashtags(tweet)
    tweet = filter_chars(tweet)
    tweet = remove_mult_spaces(tweet)
    tweet = remove_numbers(tweet)
    tweet = lemmatize(tweet)
    tweet = remove_short_words(tweet)
    tweet = replace_elongated_words(tweet)
    tweet = remove_repeated_punctuation(tweet)
    tweet = remove_extra_whitespace(tweet)
    tweet = remove_url_shorteners(tweet)
    tweet = remove_spaces_tweets(tweet)
    # tweet = remove_short_tweets(tweet)
    tweet = " ".join(tweet.split())
    return tweet


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning to the 'text' column, drop empty rows and duplicates."""
    df = df.copy()
    df["text"] = df["text"].apply(clean_tweet)
    # df = df[df["text"].str.strip().astype(bool)].reset_index(drop=False)
    # df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return df
