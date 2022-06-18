from transformers import pipeline
from rouge_score import rouge_scorer
import argparse, json

def main(args):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    model = pipeline('text-generation', model=args.model_path)
    scores = []
    fw_res = open('{}/test_generation.txt'.format(args.model_path), 'w')
    for line in open('{}/test_encoded.csv'.format(args.data_path)):
        if line.strip()[:4] == '[ID]':
            continue
        ref = line.split('\t')[5]
        prompt = line.split('\t')[5].split('<PLOT>')[0]
        hyp = model(prompt, max_length=1024)[0]['generated_text']
        scores.append(scorer.score(ref, hyp))
        fw_res.write('========== Reference =========\n')
        fw_res.write('{}\n'.format(ref))
        fw_res.write('========== Generation =========\n')
        fw_res.write('{}\n'.format(hyp))
    fw_res.close()
    aver_score = {}
    for metric in scores[0]:
        aver_score[metric] = {}
        aver_score[metric]['precision'] = 0
        aver_score[metric]['recall'] = 0
        aver_score[metric]['f1'] = 0
    for i in scores:
        for metric in scores[i]:
            aver_score[metric]['precision'] += scores[i][metric][0]
            aver_score[metric]['recall'] += scores[i][metric][1]
            aver_score[metric]['f1'] += scores[i][metric][2]
    aver_score[metric]['precision'] /= len(scores)
    aver_score[metric]['recall'] /= len(scores)
    aver_score[metric]['f1'] /= len(scores)
    json.dump(aver_score, open('{}/text_rouge.json', 'w'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--data_path', type=str)
    args = parser.parse_args()
    main(args)
