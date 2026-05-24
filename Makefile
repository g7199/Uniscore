.PHONY: all full score table recompute mapping clean

all: recompute table

full: score all

score:
	python3 scripts/score_qwen.py --datasets all

table: main_table.txt main_table_latex.txt

recompute: generated/recomputed_main_numbers.csv

mapping: generated/mapping_delta_ablation.csv

inputs/software_qwen3_1_7b/train_results_with_scores.csv inputs/software_qwen3_1_7b/test_results_with_scores.csv: | scripts/score_qwen.py
	python3 scripts/score_qwen.py --datasets software

inputs/samsum/train_results_with_scores.csv inputs/samsum/test_results_with_scores.csv: | scripts/score_qwen.py
	python3 scripts/score_qwen.py --datasets samsum

inputs/depression/train_high_scored.csv inputs/depression/train_low_scored.csv inputs/depression/test_results_with_scores.csv: | scripts/score_qwen.py
	python3 scripts/score_qwen.py --datasets depression

generated/recomputed_main_numbers.csv: scripts/recompute_main_numbers.py inputs/software_qwen3_1_7b/train_results_with_scores.csv inputs/software_qwen3_1_7b/test_results_with_scores.csv inputs/samsum/train_results_with_scores.csv inputs/samsum/test_results_with_scores.csv inputs/depression/train_high_scored.csv inputs/depression/train_low_scored.csv inputs/depression/test_results_with_scores.csv
	python3 scripts/recompute_main_numbers.py

generated/mapping_delta_ablation.csv: scripts/mapping_delta_ablation.py generated/recomputed_main_numbers.csv
	python3 scripts/mapping_delta_ablation.py

main_table.txt: data/main_table_values.csv scripts/render_main_table.py
	python3 scripts/render_main_table.py --format txt --out main_table.txt

main_table_latex.txt: data/main_table_values.csv scripts/render_main_table.py
	python3 scripts/render_main_table.py --format latex --out main_table_latex.txt

clean:
	rm -f main_table.txt main_table_latex.txt
	rm -f generated/recomputed_main_numbers.csv generated/mapping_delta_ablation.csv
	rm -rf scripts/__pycache__ src/uniscore/__pycache__
