python code_mining.py \
	--train_file "/data/xuht/oppo/pingfei_oppo.txt" \
	--stop_word_file "/data/xuht/stopwords-zh-master/stopwords-zh.txt" \
	--ouput_file "/data/xuht/oppo/pingfei" \
	--num_topics 2 \
	--iteration 1000 \
	--optimization_burnin 100 \
	--alpha 4 \
	--optimization_iterations 50 \
	--beta 0.02 \
	--min_support 5 \
	--max_phrase_size 6

