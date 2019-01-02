python get_tokenization.py \
	--raw_corpus "/data/xuht/websiteanalyze-data-seqing20180821/yancao.csv" \
	--corpus "/data/xuht/product/yancao/corpus.txt" \
	--model_prefix "/data/xuht/product/yancao/unigram" \
	--vocab_size 100000 \
	--model_type "char" \
	--character_coverage 0.9995 \
	--tokenized_text "/data/xuht/product/yancao/unigram_tokenized.txt"
