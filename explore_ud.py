import argparse
import re
import os
import sys
import glob
import json
import traceback
from collections import defaultdict
from pathlib import Path
from pprint import pprint

import numpy as np
from numpy.random import default_rng
import stanza
import udon2
from udon2.visual import render_dep_tree
from udon2.kernels import ConvPartialTreeKernel


def exclude_mwts(stanza_obj):
    return [[w for w in sent if type(w['id']) == int] for sent in stanza_obj]



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--treebank', type=str, required=True, help="A path to UD treebank")
    parser.add_argument('-l', '--language', type=str, required=True, help="Language")
    parser.add_argument('-ap', '--augmented-parser', action='store_true')
    args = parser.parse_args()

    base_dir = '/home/dmytro/oss/stanza/saved_models'

    mwt_required = ['uk']
    dep_proc = 'tokenize,lemma,mwt,pos,depparse' if args.language in mwt_required else 'tokenize,lemma,pos,depparse'
    if args.augmented_parser:
        # for some reason they don't need MWT for the pretrained English models
        # maybe because they trained on combined treebanks and not all of them have MWT?
        mwt_required.append('en') 

        if args.language in mwt_required:
            ud_parser = stanza.Pipeline(
                lang=args.language, processors=dep_proc,
                tokenize_model_path=glob.glob(os.path.join(base_dir, 'tokenize', '{}_*_tokenizer.pt'.format(args.language)))[0],
                depparse_model_path=glob.glob(os.path.join(base_dir, 'depparse', '{}_*_parser.pt'.format(args.language)))[0],
                pos_model_path=glob.glob(os.path.join(base_dir, 'pos', '{}_*_tagger.pt'.format(args.language)))[0],
                lemma_model_path=glob.glob(os.path.join(base_dir, 'lemma', '{}_*_lemmatizer.pt'.format(args.language)))[0],
                mwt_model_path=glob.glob(os.path.join(base_dir, 'mwt', '{}_*_mwt_expander.pt'.format(args.language)))[0]
            )
        else:
            ud_parser = stanza.Pipeline(
                lang=args.language, processors=dep_proc,
                tokenize_model_path=glob.glob(os.path.join(base_dir, 'tokenize', '{}_*_tokenizer.pt'.format(args.language)))[0],
                depparse_model_path=glob.glob(os.path.join(base_dir, 'depparse', '{}_*_parser.pt'.format(args.language)))[0],
                pos_model_path=glob.glob(os.path.join(base_dir, 'pos', '{}_*_tagger.pt'.format(args.language)))[0],
                lemma_model_path=glob.glob(os.path.join(base_dir, 'lemma', '{}_*_lemmatizer.pt'.format(args.language)))[0]
            )
    else:
        ud_parser = stanza.Pipeline(lang=args.language, processors=dep_proc)

    prefix = Path(args.treebank).stem

    YEAR = re.compile(r"(?<= )\d{4}(?= )")

    kernel = ConvPartialTreeKernel("GRCT", includeFeats=True, includeForm=False)

    trees = udon2.Importer.from_conll_file(args.treebank)
    print("Total # of trees: {}".format(len(trees)))

    finalists = []
    for t in trees:
    	match = YEAR.search(t.get_subtree_text())
    	if match:
    		finalists.append((t, match))

    K = 50
    rng = default_rng(7919) # seed sequence is set the same for reproducibility
    years = rng.integers(1100, 2100, size=(K,))

    # report variable
    r = {
        'original': {
            'total': len(finalists),
            'corr_parsed': 0,
            'assertion_errors': 0,
            'gold': udon2.TreeList(),
            'parsed': udon2.TreeList(),
            'wrong_sent_segm': 0,
            'conv_tree_kernels': []
        },
        'augmented': {
            'total': 0,
            'corr_parsed': 0,
            'assertion_errors': 0,
            'same_as_first': 0,
            'gold': udon2.TreeList(),
            'parsed': udon2.TreeList(),
            'wrong_sent_segm': 0,
            'upos': defaultdict(lambda: defaultdict(int)),
            'deprel': defaultdict(lambda: defaultdict(int)),
            'feats': defaultdict(lambda: defaultdict(int)),
            'batches': []
        }
    }

    results_dir = '{}_results'.format(prefix)
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    batches_dir = os.path.join(results_dir, '{}_batches'.format(prefix))
    if not os.path.exists(batches_dir):
        os.mkdir(batches_dir)

    for batch_id, (f, match) in enumerate(finalists):
        sentence = f.get_subtree_text()

        try:
            trees = udon2.Importer.from_stanza(exclude_mwts(ud_parser(sentence).to_dict()))
        except AssertionError as e:
            # This will fire most probably if MWT module is necessary
            print("Original assertion error:", sentence)
            print(traceback.format_exc())
            r['original']['assertion_errors'] += 1

        if len(trees) > 1:
            r['original']['wrong_sent_segm'] += 1
            continue
        else:
            tree = trees[0]

        r['original']['gold'].append(f)
        r['original']['parsed'].append(tree)
        f_norm, t_norm = kernel(f, f), kernel(tree, tree)
        original_kernel = float(round(kernel(f, tree) / ( np.sqrt(f_norm) * np.sqrt(t_norm) ), 4))
        original_identical = original_kernel == 1
        r['original']['corr_parsed'] += original_identical
        r['original']['conv_tree_kernels'].append(original_kernel)

        s, e = match.span()

        current_year = sentence[s:e]
        first, f1 = None, f.copy()
        batch = {
            'total': 0,
            'assertion_errors': 0,
            'corr_parsed': 0,
            'same_as_first': 0,
            'wrong_sent_segm': 0,
            'original_match': original_identical,
            'upos': defaultdict(lambda: defaultdict(int)),
            'deprel': defaultdict(lambda: defaultdict(int)),
            'feats': defaultdict(lambda: defaultdict(int)),
            'conv_kernel_matrix': None, # TBI later
            'conv_kernel_gold': []
        }
        batch_parsed_trees = udon2.TreeList()
        f1_norm = kernel(f1, f1)
        precomputed_norms = {} # for conv tree kernels
        for y in years:
            nodes = f1.select_by('form', current_year)
            y_str = str(y)
            for node in nodes:
                node.form = y_str
                node.lemma = y_str
            current_year = y_str

            batch['total'] += 1
            
            # slows down the process, but is currently the only way to know
            # which exact occurences of the number were replaced, since there can be some compounds
            # like 1960s, which won't be found by regex or select_by if the target is 1960
            # however, this will get replaced by string.replace or re.sub
            new_sentence = f1.get_subtree_text()
            try:
                trees = udon2.Importer.from_stanza(exclude_mwts(ud_parser(new_sentence).to_dict()))
            except AssertionError as e:
                # This will fire most probably if MWT module is necessary
                print("Augmented assertion error:", new_sentence)
                print(traceback.format_exc())
                batch['assertion_errors'] += 1
                r['augmented']['assertion_errors'] += 1

            r['augmented']['total'] += 1

            if len(trees) > 1:
                r['augmented']['wrong_sent_segm'] += 1
                batch['wrong_sent_segm'] += 1
                continue
            else:
                tree = trees[0]

            if not first:
                first = tree.copy()

            precomputed_norms[batch['total']-batch['wrong_sent_segm']-1] = kernel(tree, tree)

            norm = np.sqrt(f1_norm) * np.sqrt(precomputed_norms[batch['total']-batch['wrong_sent_segm']-1])
            trees_kernel = round( float(kernel(f1, tree) / norm), 4 )
            trees_identical = trees_kernel == 1
            r['augmented']['corr_parsed'] += trees_identical
            batch['corr_parsed'] += trees_identical
            batch['conv_kernel_gold'].append(trees_kernel)

            r['augmented']['gold'].append(f1.copy())
            r['augmented']['parsed'].append(tree)

            batch_parsed_trees.append(tree.copy())

            nodes2 = tree.select_by('form', current_year) # current_year is now y_str

            for n1, n2 in zip(nodes, nodes2):
                r['augmented']['upos'][n1.upos][n2.upos] += 1
                r['augmented']['deprel'][n1.deprel][n2.deprel] += 1
                r['augmented']['feats'][str(n1.feats)][str(n2.feats)] += 1

                batch['upos'][n1.upos][n2.upos] += 1
                batch['deprel'][n1.deprel][n2.deprel] += 1
                batch['feats'][str(n1.feats)][str(n2.feats)] += 1

        # if batch['original_match']:
        #     del batch['conv_kernel_matrix']
        # else:
        NB = len(batch_parsed_trees)
        batch['conv_kernel_matrix'] = np.zeros((NB, NB))
        for i in range(NB):
            for j in range(i, NB):
                if i == j:
                    batch['conv_kernel_matrix'][i][j] = 1.0
                else:
                    t1, t2 = batch_parsed_trees[i], batch_parsed_trees[j]
                    ck = kernel(t1, t2)
                    if np.isfinite(ck) and np.isfinite(precomputed_norms[i]) and np.isfinite(precomputed_norms[j]):
                        batch['conv_kernel_matrix'][i][j] = round(
                            float(ck / (np.sqrt(precomputed_norms[i]) * np.sqrt(precomputed_norms[j]))), 4)
                    else:
                        batch['conv_kernel_matrix'][i][j] = -1
        same_as_first_cnt = int(sum(batch['conv_kernel_matrix'][0][1:] == 1))
        r['augmented']['same_as_first'] += same_as_first_cnt
        batch['same_as_first'] += same_as_first_cnt
        batch['conv_kernel_matrix'] = batch['conv_kernel_matrix'].tolist()
        
        udon2.ConllWriter.write_to_file(batch_parsed_trees, os.path.join(
            batches_dir, "{}_augmented_parsed_b{}.conllu".format(prefix, batch_id)
        ))
        r['augmented']['batches'].append(batch)

    ind, outd = r['original'], r['augmented']
    udon2.ConllWriter.write_to_file(ind['gold'], os.path.join(results_dir, "{}_original_gold.conllu".format(prefix)))
    udon2.ConllWriter.write_to_file(ind['parsed'], os.path.join(results_dir, "{}_original_parsed.conllu".format(prefix)))

    udon2.ConllWriter.write_to_file(outd['gold'], os.path.join(results_dir, "{}_augmented_gold.conllu".format(prefix)))
    udon2.ConllWriter.write_to_file(outd['parsed'],os.path.join(results_dir,  "{}_augmented_parsed.conllu".format(prefix)))

    del r['original']['gold']
    del r['original']['parsed']

    del r['augmented']['gold']
    del r['augmented']['parsed']

    r['comments'] = [
        "Correct: {}% ({} out of {}, skipped {} with wrong sentence segmentation)".format(
            round(ind['corr_parsed'] * 100 / (ind['total'] - ind['wrong_sent_segm']), 2),
            ind['corr_parsed'], ind['total'] - ind['wrong_sent_segm'], ind['wrong_sent_segm']),
        "Correct (aug): {}% ({} out of {}, skipped {} with wrong sentence segmentation)".format(
            round(outd['corr_parsed'] * 100 / (outd['total'] - outd['wrong_sent_segm']), 2),
            outd['corr_parsed'], outd['total'] - outd['wrong_sent_segm'], outd['wrong_sent_segm'])
    ]

    json.dump( r, open( os.path.join(results_dir, "{}.json".format(prefix)), 'w' ) )