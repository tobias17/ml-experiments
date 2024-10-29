
CMD = python 13_token_cluster/train.py --restore weights/13_token_cluster/2024_10_25_22_03_38_606945 --decoded-loss --predict-loss

test:
	$(CMD)

beam:
	BEAM=20 $(CMD)
