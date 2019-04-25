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
	"一般辱骂":"辱骂",
	"严重辱骂":"辱骂",
	"辱骂":"辱骂",
	"轻微辱骂":"辱骂",
	"口头语":"口头语",
	"严重色情":"色情",
	"色情":"色情",
	"违禁商品":"色情",
	"轻微色情":"色情",
	"内涵":"色情",
	"性知识":"色情",
	"色情违禁":"色情",
	"涉政":"涉政",
	"涉及政治细则":"涉政",
	"涉及政治负面细则":"涉政",
	"广告":"广告",
	"广告号":"广告",
	"广告语":"广告",
	"正常":"正常",
	"其他类":"其他类"
})

from collections import Counter
risk_data = {}

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
			risk_data[key] = data[key]
	else:
		risk_data[key] = data[key]

# pkl.dump(risk_data, open('/data/xuht/free_text_topmines/global_mining_unigram/phrase_filter_risk.pkl', "wb"))
pkl.dump(risk_data, open(FLAGS.phrase_filter_risk, "wb"))



