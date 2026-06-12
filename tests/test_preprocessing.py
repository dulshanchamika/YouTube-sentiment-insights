import pytest
from src.data.data_preprocessing import preprocess_comment

def test_preprocess_comment_lowercase():
    assert preprocess_comment("HELLO WORLD") == "hello world"

def test_preprocess_comment_strip():
    assert preprocess_comment("  hello  ") == "hello"

def test_preprocess_comment_newline():
    assert preprocess_comment("hello\nworld") == "hello world"

def test_preprocess_comment_special_chars():
    # It keeps A-Z, a-z, 0-9, space, !, ?, ., ,
    # It removes others
    assert preprocess_comment("hello @world#") == "hello world"

def test_preprocess_comment_stopwords():
    # 'is' is a stopword, 'not' is preserved
    result = preprocess_comment("this is not good")
    assert "is" not in result
    assert "not" in result

def test_preprocess_comment_lemmatization():
    # 'running' becomes 'running' or 'run' depending on lemmatizer
    # NLTK WordNetLemmatizer default is noun, so 'running' stays 'running' 
    # unless pos tagging is used. Let's test a simple plural.
    assert preprocess_comment("cats") == "cat"
