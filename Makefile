
CMD = python 16_cheat_sheet_v2/train.py
EXTRA_ARGS =
JITBEAM ?= 20

bake_all:
	$(MAKE) run_cheat_sheet EXTRA_ARGS=--only-bake
	$(MAKE) run_baseline_ctx EXTRA_ARGS=--only-bake
	$(MAKE) run_baseline_no_ctx EXTRA_ARGS=--only-bake

run_cheat_sheet:
	JITBEAM=$(JITBEAM) $(CMD) cheat_sheet $(EXTRA_ARGS)

run_baseline_ctx:
	JITBEAM=$(JITBEAM) $(CMD) baseline_ctx $(EXTRA_ARGS)

run_baseline_no_ctx:
	JITBEAM=$(JITBEAM) $(CMD) baseline_no_ctx $(EXTRA_ARGS)
