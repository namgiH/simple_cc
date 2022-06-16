import os, itertools 
import pandas as pd
target = ['bart', 'rake']
mode = ['cc', 'ncc']
for tm in itertools.product(target, mode):
    files = os.listdir('./data/en_{}'.format(tm[0]))
    for f in files:
        os.makedirs('sp/en_{}_{}'.format(tm[0], tm[1]), exist_ok=True)
        fw = open('sp/en_{}_{}/{}'.format(tm[0], tm[1], f), 'w')
        fw.write('[ID]\t[TITLE]\t[OUTLINE]\t[PLOT]\ttext\n')
        plot_id = ''
        data = ''
        for line in open('./data/en_{}/{}'.format(tm[0], f)):
            if line[:4] == '[ID]' or not line.strip():
                continue
            tokens = line.strip().split('\t')
            input_plot_id = tokens[0].split('_')[0]
            plot = tokens[5] if len(tokens) > 5 else ''
            if input_plot_id != plot_id:
                if data:
                    text = '<OUTLINE>{}</OUTLINE><PLOT>{}</PLOT>'.format(data.split('\t')[2], data.split('\t')[3])
                    fw.write('{}\t{}\n'.format(data.strip(), text))
                    if tm[1] == 'cc':
                        text = '<PLOT>{}</PLOT><OUTLINE>{}</OUTLINE>'.format(data.split('\t')[3], data.split('\t')[2])
                        fw.write('{}\t{}\n'.format(data.strip(), text))
                plot_id = input_plot_id
                data = '{}\t{}\t{}\t{}'.format(input_plot_id, tokens[2].split('[SEP]')[0], ' '.join(tokens[2].split('[SEP]')[1:]), plot)
            else:
                data = data + ' ' + plot
        text = '<OUTLINE>{}</OUTLINE><PLOT>{}</PLOT>'.format(data.split('\t')[2], data.split('\t')[3])
        fw.write('{}\t{}\n'.format(data.strip(), text))
        if tm[1] == 'cc':
            text = '<PLOT>{}</PLOT><OUTLINE>{}</OUTLINE>'.format(data.split('\t')[3], data.split('\t')[2])
            fw.write('{}\t{}\n'.format(data.strip(), text))
        fw.close()
        tmp = pd.read_csv('sp/en_{}_{}/{}'.format(tm[0], tm[1], f), sep='\t')
        tmp = tmp.dropna()
        tmp.to_csv('sp/en_{}_{}/{}'.format(tm[0], tm[1], f), sep='\t')
