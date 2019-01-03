import sentencepiece as spm
import re

from jieba import Tokenizer
from jieba.posseg import POSTokenizer

class SPM(object):
	def __init__(self, config):
		self.config = config
		self.sp = spm.SentencePieceProcessor()

	def load_dict(self):
		self.dict = []
		try:
			with open(self.config.get("word_dict", None), "r") as frobj:
				for line in frobj:
					content = line.strip().split("\t")[0]
					# if "▁" in content:
					#   word = content.split("▁")[1]
					#   if word in self.dict:
					#     continue
					#   else:
					#     self.dict.append(word)
					# else:
					self.dict.append(content)
		except:
			raise ValueError("Not existed word piece dict")

	def add_extra_word(self, 
		extra_lst=["[PAD]", "[UNK]","[CLS]", "[SEP]", "[MASK]"]):
		extra_lst = extra_lst if extra_lst else self.config.get("extra_lst", [])
		if len(extra_lst) >= 1:
			for word in extra_lst:
				if word in self.dict:
					self.dict.remove(word)
			self.dict = extra_lst + self.dict
			with open("/data/xuht/tmp_vocab.txt", "w") as fwobj:
				for word in self.dict:
					fwobj.write(word+"\n")
		
	def build_word_id(self):
		self.word2id, self.id2word = {}, {}
		for index, word in enumerate(self.dict):
			self.word2id[word] = index
			self.id2word[index] = word
		
	def load_model(self):
		try:
			self.sp.Load(self.config.get("word_piece_model", None))
		except:
			raise ValueError('Not found word piece model')

	def train_model(self, train_config=None):
		config = train_config if train_config else self.config
		param = ""
		param += "--input={} ".format(self.config["corpus"])
		param += "--model_prefix={} ".format(self.config["model_prefix"])
		param += "--vocab_size={} ".format(self.config["vocab_size"])
		param += "--model_type={} ".format(self.config.get("model_type", "unigram"))
		param += "--character_coverage={}".format(self.config.get("character_coverage", "0.995"))
		try:
			spm.SentencePieceTrainer.Train(param)
			self.sp.Load(self.config["model_prefix"]+".model")
		except:
			raise ValueError(" training word piece model failed ")

	def tokenize(self, text):
		tokenized_text = self.sp.EncodeAsPieces(text)
		# tf.logging.info(" text {} token {}".format(text, tokenized_text))
		tmp_text = []
		for word in tokenized_text:
			word = re.sub("▁", "", word)
			if len(word) >= 1:
				tmp_text.append(word)
		return tmp_text

	def convert_tokens_to_ids(self, text, unk="[UNK]"):
		try:
			tokenized_text = self.tokenize(text)
		except:
			tokenized_text = text
		token_id_lst = [self.word2id.get(word, self.word2id[unk]) for word in tokenized_text]
		return token_id_lst

	def padding(self, token_id_lst, max_length, zero_padding=0):
		return token_id_lst + [zero_padding] * (max_length - len(token_id_lst))

class Jieba(object):
	def __init__(self, config):
		self.config = config
		self.dt = POSTokenizer()

	def load_dict(self):
		self.dict = []
		try:
			with open(self.config.get("word_dict", None), "r") as frobj:
				for line in frobj:
					content = line.strip().split("\t")[0]
					self.dict.append(content)
		except:
			raise ValueError("Not existed word piece dict")

	def load_model(self):
		for word in self.dict:
			self.dt.add_word(word)

	def build_word_id(self):
		self.word2id, self.id2word = {}, {}
		for index, word in enumerate(self.dict):
			self.word2id[word] = index
			self.id2word[index] = word

	def add_extra_word(self, 
			extra_lst=["[PAD]", "[UNK]","[CLS]", "[SEP]", "[MASK]"]):
		extra_lst = extra_lst if extra_lst else self.config.get("extra_lst", [])
		if len(extra_lst) >= 1:
			for word in extra_lst:
				if word in self.dict:
					self.dict.remove(word)
			self.dict = extra_lst + self.dict

	def train_model(self, train_config=None):
		config = train_config if train_config else self.config
		self.dict = []
		try:
			with open(config.get("word_dict", None)) as frobj:
				for line in frobj:
					content = line.strip().split("\t")[0]
					self.dict.append(content)
		except:
			raise ValueError(" not existed word dict")

	def tokenize(self, text):
		tokenized_text = self.dt.lcut(text)
		return [list(word)[0] for word in tokenized_text]

	def convert_tokens_to_ids(self, text, unk="[UNK]"):
		tokenized_text = self.tokenize(text)
		token_id_lst = [self.word2id.get(word, self.word2id[unk]) for word in tokenized_text]
		return token_id_lst