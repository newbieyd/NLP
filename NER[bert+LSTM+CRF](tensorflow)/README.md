tf1.12

预处理
tokenization.py是对输入的句子处理，包含两个主要类：BasickTokenizer, FullTokenizer

BasickTokenizer会对每个字做分割，会识别英文单词，对于数字会合并，例如：

query: 'Jack,请回答1988, UNwant\u00E9d,running'
token: ['jack', ',', '请', '回', '答', '1988', ',', 'unwanted', ',', 'running']
FullTokenizer会对英文字符做n-gram匹配，会将英文单词拆分，例如running会拆分为run、##ing，主要是针对英文。

query: 'UNwant\u00E9d,running'
token: ["un", "##want", "##ed", ",", "runn", "##ing"]
对于中文数据，特别是NER，如果数字和英文单词是整体的话，会出现大量UNK，所以要将其拆开，想要的结果：

query: 'Jack,请回答1988'
token:  ['j', 'a', 'c', 'k', ',', '请', '回', '答', '1', '9', '8', '8']
具体变动如下：

class CharTokenizer(object):
    """Runs end-to-end tokenziation."""
    def __init__(self, vocab_file, do_lower_case=True):
        self.vocab = load_vocab(vocab_file)
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in token:
                split_tokens.append(sub_token)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_tokens_to_ids(self.vocab, tokens)
