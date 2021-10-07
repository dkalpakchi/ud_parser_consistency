import argparse
import re
import json
from collections import defaultdict
from pathlib import Path
from pprint import pprint

import numpy as np
from numpy.random import default_rng
import stanza
import udon2
from udon2.visual import render_dep_tree
from udon2.kernels import ConvPartialTreeKernel


def is_identical(r1, r2):
    if r1.is_identical(r2, "") and str(r1.children) == str(r2.children):
        all_identical = True
        for c1, c2 in zip(r1.children, r2.children):
            all_identical = all_identical and is_identical(c1, c2)
            if not all_identical:
                return all_identical
        return all_identical
    else:
        return False


def nodelist2string(lst):
    return " ".join(["|".join([n.upos, n.deprel]) for n in lst])


def is_identical_except_form_and_lemma(r1, r2):
    if r1.is_identical(r2, "form,lemma") and nodelist2string(r1.children) == nodelist2string(r2.children):
        all_identical = True
        for c1, c2 in zip(r1.children, r2.children):
            all_identical = all_identical and is_identical_except_form_and_lemma(c1, c2)
            if not all_identical:
                return all_identical
        return all_identical
    else:
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--treebank', type=str, required=True, help="A path to UD treebank")
    parser.add_argument('-l', '--language', type=str, default='en', help="Language")
    args = parser.parse_args()

    dep_proc = 'tokenize,lemma,mwt,pos,depparse' if args.language in ['fi', 'ar'] else 'tokenize,lemma,pos,depparse'
    ud_parser = stanza.Pipeline(
        lang=args.language, processors=dep_proc
        # tokenize_model_path='saved_models/tokenize/sv_talbanken_tokenizer.pt',
        # depparse_model_path='saved_models/depparse/sv_talbanken_tokenizer.pt',
        # pos_model_path='saved_models/pos/sv_talbanken_tokenizer.pt',
        # lemma_model_path='saved_models/lemma/sv_talbanken_tokenizer.pt'
    )

    prefix = Path(args.treebank).stem

    YEAR = re.compile(r"(?<= )\d{4}(?= )")
    YEAR_EN = re.compile(r"(?<= )\d{4}(?= year)|(?<=year )\d{4}(?= )")

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
            'gold': udon2.TreeList(),
            'parsed': udon2.TreeList(),
            'wrong_sent_segm': 0
        },
        'augmented': {
            'total': 0,
            'corr_parsed': 0,
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

    for f, match in finalists:
        sentence = f.get_subtree_text()
        trees = udon2.Importer.from_stanza(ud_parser(sentence).to_dict())

        if len(trees) > 1:
            r['original']['wrong_sent_segm'] += 1
            continue
        else:
            tree = trees[0]

        r['original']['gold'].append(f)
        r['original']['parsed'].append(tree)
        original_identical = is_identical(f, tree)
        r['original']['corr_parsed'] += original_identical

        s, e = match.span()

        current_year = sentence[s:e]
        first, f1 = f.copy(), f.copy()
        batch = {
            'total': 0,
            'corr_parsed': 0,
            'same_as_first': 0,
            'wrong_sent_segm': 0,
            'original_match': original_identical,
            'upos': defaultdict(lambda: defaultdict(int)),
            'deprel': defaultdict(lambda: defaultdict(int)),
            'feats': defaultdict(lambda: defaultdict(int)),
            'conv_kernel_matrix': np.zeros((K, K))
        }
        batch_parsed_trees = udon2.TreeList()
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
            trees = udon2.Importer.from_stanza(ud_parser(new_sentence).to_dict())

            r['augmented']['total'] += 1

            if len(trees) > 1:
                r['augmented']['wrong_sent_segm'] += 1
                batch['wrong_sent_segm'] += 1
                continue
            else:
                tree = trees[0]

            trees_identical = is_identical(f1, tree)
            r['augmented']['corr_parsed'] += trees_identical
            batch['corr_parsed'] += trees_identical

            same_as_first = is_identical_except_form_and_lemma(first, tree)
            r['augmented']['same_as_first'] += same_as_first
            batch['same_as_first'] += same_as_first

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

        if batch['original_match']:
            del batch['conv_kernel_matrix']
        else:
            NB = len(batch_parsed_trees)
            if NB != K:
                batch['conv_kernel_matrix'] = np.zeros((NB, NB))
            precomputed_norms = {}
            for i in range(NB):
                for j in range(i, NB):
                    # if i == j:
                    #     batch['conv_kernel_matrix'][i][j] = 1.0
                    # else:
                    t1, t2 = batch_parsed_trees[i], batch_parsed_trees[j]
                    if i not in precomputed_norms:
                        precomputed_norms[i] = kernel(t1, t1)
                    if j not in precomputed_norms:
                        precomputed_norms[j] = kernel(t2, t2)
                    
                    batch['conv_kernel_matrix'][i][j] = round(kernel(t1, t2) / (np.sqrt(precomputed_norms[i]) * np.sqrt(precomputed_norms[j])), 4)
            batch['conv_kernel_matrix'] = batch['conv_kernel_matrix'].tolist()

        r['augmented']['batches'].append(batch)

    ind, outd = r['original'], r['augmented']
    udon2.ConllWriter.write_to_file(ind['gold'], "{}_original_gold.conllu".format(prefix))
    udon2.ConllWriter.write_to_file(ind['parsed'], "{}_original_parsed.conllu".format(prefix))

    udon2.ConllWriter.write_to_file(outd['gold'], "{}_augmented_gold.conllu".format(prefix))
    udon2.ConllWriter.write_to_file(outd['parsed'], "{}_augmented_parsed.conllu".format(prefix))

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
    json.dump(r, open("{}.json".format(prefix), 'w'))