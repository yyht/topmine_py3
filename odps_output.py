import _pickle as pkl

import tensorflow as tf

import _pickle as pkl

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
	"phrase_filter", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"phrase_filter_risk", None,
	"Input TF example files (can be a glob or comma separated).")

data = pkl.load(open(FLAGS.phrase_filter, "rb"))
# data = pkl.load(open("/data/xuht/free_text_topmines/global_mining_unigram/phrase_filter.pkl", "rb"))

left = []
for key in data:
	if len(data[key]["label"]) == 0:
		left.append([key, data[key]])
print(len(left))

from collections import Counter
total_label = []
for key in data:
	total_label.extend(data[key]["label"])
from collections import OrderedDict
label_count = Counter(total_label)

print(list(label_count.keys()))

label_mapping = OrderedDict({
    u"一般辱骂":u"辱骂",
    u"严重辱骂":u"辱骂",
    u"辱骂":u"辱骂",
    u"轻微辱骂":u"辱骂",
    u"口头语":u"口头语",
    u"严重色情":u"色情",
    u"色情":u"色情",
    u"违禁商品":u"违禁商品",
    u"轻微色情":u"色情",
    u"内涵":u"色情",
    u"性知识":u"色情",
    u"色情违禁":u"色情",
    u"涉政":u"涉政",
    u"涉及政治细则":u"涉政",
    u"涉及政治负面细则":u"涉政",
    u"广告":u"广告",
    u"广告号":u"广告",
    u"广告语":u"广告",
    u"正常":u"正常",
    u"其他类":u"其他类"
})

from collections import Counter
risk_data = {}

import re
number_pattern = re.compile('[0-9]+')

for key in data:
	data[key]["score"] = {}
	sub_label = []
	for index, item in enumerate(data[key]["label"]):
		if item in label_mapping:
			sub_label.append(item)
	data[key]["label"] = sub_label
	data[key]["ratio"] = Counter(data[key]["label"])
	for sub_key in data[key]["ratio"]:
		data[key]["score"][sub_key] = float(data[key]["ratio"][sub_key]) / len(data[key]["label"])
	if "正常" in data[key]["score"]:
		if data[key]["score"]["正常"] == 1.0:
			continue
		else:
			number_len = sum([len(item) for item in number_pattern.findall(key)])
			if number_len == len(key):
				continue
			risk_data[key] = data[key]
	else:
		risk_data[key] = data[key]

# import math

# def get_number_of_docs(risk_data):
# 	doc_nums = 0
# 	for key in risk_data:
# 		doc_nums += len(set(risk_data[key]["doc_id"]))
# 	return doc_nums

# 	return math.log(float(1 + self.get_num_docs()) / 
#       (1 + self.term_num_docs[term]))

# def get_tfidf(risk_data):
# 	token_nums = len(risk_data)
# 	doc_nums = get_number_of_docs(risk_data)
# 	for key in risk_data:
# 		idf = math.log(float(1 + doc_nums) / (1 + len(set(risk_data[key]["doc_id"]))))
# 		tf = risk_data[key]["count"] / token_nums
# 		risk_data[key]["tfidf"] = tf*idf

# get_tfidf(risk_data)

# pkl.dump(risk_data, open('/data/xuht/free_text_topmines/global_mining_unigram/phrase_filter_risk.pkl', "wb"))
pkl.dump(risk_data, open(FLAGS.phrase_filter_risk, "wb"))



