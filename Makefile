
CMD = python 13_token_cluster/train.py --decoded-loss --predict-loss

test:
	$(CMD)

beam:
	BEAM=20 $(CMD)
