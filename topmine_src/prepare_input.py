from __future__ import division
import re
from collections import Counter
import math
import sys
from collections import OrderedDict
import re

import codecs

CH_PUNCTUATION = u"[＂＃＄％＆＇，：；＠［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。]"
EN_PUNCTUATION = u"['!#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~']"

symbol_pattern = re.compile(CH_PUNCTUATION)
ch_pattern = re.compile(u"[\u4e00-\u9fa5]+")

from hanziconv import HanziConv

import six
def convert_to_unicode(text):
	"""Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
	if six.PY3:
		if isinstance(text, str):
			return text
		elif isinstance(text, bytes):
			return text.decode("utf-8", "ignore")
		else:
			raise ValueError("Unsupported string type: %s" % (type(text)))
	elif six.PY2:
		if isinstance(text, str):
			return text.decode("utf-8", "ignore")
		elif isinstance(text, unicode):
			return text
		else:
			raise ValueError("Unsupported string type: %s" % (type(text)))
	else:
		raise ValueError("Not running on Python2 or Python 3?")

def full2half(s):
	n = []
	for char in s:
		num = ord(char)
		if num == 0x3000:
			num = 32
		elif 0xFF01 <= num <= 0xFF5E:
			num -= 0xfee0
		num = chr(num)
		n.append(num)
	return ''.join(n)

def clean(text):
	text = text.strip()
	text = HanziConv.toSimplified(text)
	text = full2half(text)
	text = re.sub("\\#.*?#|\\|.*?\\||\\[.*?]", "", text)
	text = re.sub("\s*", "", text)
	return text

class BaseReader(object):
	def __init__(self):
		pass
		
	def _read_data(self, input_file, **kargs):
		pass

	def _create_examples(self, lines, parser_fn, **kargs):
		pass

	def _preprocess_input(self, examples, stopwords, **kargs):
		pass

class TextReader(BaseReader):
	""""""
	def __init__(self):
		super(TextReader, self).__init__()

	def _read_data(self, input_file, **kargs):
		with cpdecs.open(input_file, "r", "utf-8") as frobj:
			lines = []
			for line in frobj:
				lines.append(line.strip())
			return lines

	def _create_examples(self, lines, parser_fn, **kargs):

		self.label2id = {}
		label_index = 0

		examples = []
		for (i, line) in enumerate(lines):
			guid = i
			text_a, input_labels = parser_fn(lines)
			text_a = convert_to_unicode(text_a)

			for l in input_labels:
				if l in self.label2id:
					continue
				else:
					self.label2id[l] = label_index
					label_index += 1

			data_dict = {
				"text_a":text_a,
				"text_b":None,
				"label":[self.label2id[label] for label in input_labels],
				"guid":i
			}
			examples.append(data_dict)
		return examples

	def _preprocess_input(self, examples, stopwords, tokenization_api, **kargs):
		i = 0
		self.doc_range2doc = OrderedDict()
		documents = []
		document_range = []
		i = 0
		num_docs = 0
		for index, example in enumerate(examples):
			text_a = example["text_a"]
			if index <= 10:
				print(text_a)
			sentences_no_punc = symbol_pattern.split(text_a)
			stripped_sentences = []
			for text in sentences_no_punc:
				sentence = tokenization_api.tokenize(sentence)
				sentence_no_stopword = " ".join([word for word in sentence.split() if word not in stopwords])
				stripped_sentences.append(sentence_no_stopword)

			sentences_no_punc = stripped_sentences

			self.doc_range2doc[i] = index

			i += len(sentences_no_punc)
			document_range.append(i)
			documents.extend(sentences_no_punc)
			num_docs += 1

		documents = [doc.strip() for doc in documents]

		# remove stop-words
		documents2 = []
		for doc in documents:
			documents2.append(' '.join([word for word in doc.split() if word not in stopwords]))

		assert len(documents) == len(documents2)

		documents = documents2[:]

		return documents, document_range, num_docs

DataPrepare = {
	"seqing":TextReader,
	"product_yancao":TextReader
}


