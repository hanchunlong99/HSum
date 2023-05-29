from others.utils import test_rouge, rouge_results_to_str

temp_dir = '../temp'
can_path = '../results/cnndm.90000.candidate'
gold_path = '../results/cnndm.90000.gold'
rouges = test_rouge(temp_dir, can_path, gold_path)
print(rouge_results_to_str(rouges))