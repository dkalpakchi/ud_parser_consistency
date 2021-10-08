import argparse
import re
import sys
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from pprint import pprint

import numpy as np
from terminaltables import AsciiTable

from maximal_cliques import find_cliques


def convert_to_adjacency_list(m):
    adj = defaultdict(list)
    for i, row in enumerate(m):
        for j, v in enumerate(row):
            if j > i and v == 1:
                adj[i].append(j)
                adj[j].append(i)
    return adj


def create_safe(func):
    def safe_func(x):
        if x:
            return func(x)
        else:
            return 'NA'
    return safe_func

safe_mean = create_safe(np.mean)
safe_std = create_safe(np.std)
safe_median = create_safe(np.median)
safe_min = create_safe(np.min)
safe_max = create_safe(np.max)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, required=True, help="A path to JSON log file")
    args = parser.parse_args()

    data = json.load(open(args.file))
    batches = data['augmented']['batches']

    r = {
        'corr_parsed': {
            True: [],
            False: []
        },
        'all_correct': {
            True: 0, # bools are for original_match
            False: 0
        },
        'same_as_first': {
            True: 0,
            False: 0
        },
        'deprel_mismatch': {
            True: 0,
            False: 0
        },
        'upos_mismatch': {
            True: 0,
            False: 0
        },
        'feats_mismatch': {
            True: 0,
            False: 0
        },
        'original_match': 0,
        'total': len(batches),
        # 'error_clusters': [],
        'error_clusters_sim': {
            True: [],
            False: []
        },
        'error_clusters_num': {
            True: [],
            False: []
        }
    }
    for b in batches:
        r['original_match'] += b['original_match']
        r['all_correct'][b['original_match']] += b['corr_parsed'] == b['total']
        r['corr_parsed'][b['original_match']].append(b['corr_parsed'])

        # Only for those where original was incorrectly parsed
        consistent_errors = b['same_as_first'] == (b['total'] - b['wrong_sent_segm'] - 1) # since first is obviously same as first :)
        r['same_as_first'][b['original_match']] += consistent_errors
        if not consistent_errors:
            cliques = find_cliques( convert_to_adjacency_list(b['conv_kernel_matrix']) )
            if len(cliques) == 1 and len(cliques[0]) != (b['total'] - b['wrong_sent_segm']):
                # means some node is not connected to any other, we need to account for that
                cliques.append(set(range(b['total'] - b['wrong_sent_segm'])) - cliques[0])
            # r['error_clusters'].append(cliques)
            references = [list(clique)[0] for clique in cliques]
            for ref1, ref2 in combinations(references, 2):
                r['error_clusters_sim'][b['original_match']].append(b['conv_kernel_matrix'][ref1][ref2])

            r['error_clusters_num'][b['original_match']].append(len(cliques))

        # if b['corr_parsed'] != b['total']:
        #     r['all_same_as_first'][b['original_match']] += b['same_as_first'] == b['total']
        #     if b['same_as_first'] > 0:
        #         print(b)
        # r['deprel_mismatch'][b['original_match']] += len(b['deprel'][list(b['deprel'].keys())[0]].keys()) > 1
        # r['upos_mismatch'][b['original_match']] += len(b['upos'][list(b['upos'].keys())[0]].keys()) > 1
        # r['feats_mismatch'][b['original_match']] += len(b['feats'][list(b['feats'].keys())[0]].keys()) > 1

    print("Total:", data['original']['total'])
    print("Total (aug):", data['augmented']['total'])
    pprint(data['comments'])
    print("Batches:", r['total'])
    table_data = [
        ['Metric', 'Original correctly parsed', 'Original incorrectly parsed'],
        ['Total', r['original_match'], r['total'] - r['original_match']],
        ['All correct in a batch', r['all_correct'][True], r['all_correct'][False]],
        ['Correctly parsed in a batch (out of 50): MEAN (SD)', 
            "{} ({})".format(round(safe_mean(r['corr_parsed'][True]), 2), round(safe_std(r['corr_parsed'][True]), 2)), 
            "{} ({})".format(round(safe_mean(r['corr_parsed'][False]), 2), round(safe_std(r['corr_parsed'][False]), 2)) ],
        ['Correctly parsed in a batch (out of 50): MEDIAN (MIN - MAX)', 
            "{} ({} - {})".format(
                safe_median(r['corr_parsed'][True]),
                safe_min(r['corr_parsed'][True]),
                safe_max(r['corr_parsed'][True])), 
            "{} ({} - {})".format(
                safe_median(r['corr_parsed'][False]),
                safe_min(r['corr_parsed'][False]),
                safe_max(r['corr_parsed'][False]))],
        ['Consistent errors', 'NA', r['same_as_first']],
        ['Number of error clusters: MEAN (STD)',
            "{} ({})".format(safe_mean(r['error_clusters_num'][True]), safe_std(r['error_clusters_num'][True])),
            "{} ({})".format(safe_mean(r['error_clusters_num'][False]), safe_std(r['error_clusters_num'][False]))],
        ['Number of error clusters: MEDIAN (MIN - MAX)',
            "{} ({} - {})".format(
                safe_median(r['error_clusters_num'][True]), 
                safe_min(r['error_clusters_num'][True]),
                safe_max(r['error_clusters_num'][True])),
            "{} ({} - {})".format(
                safe_median(r['error_clusters_num'][False]), 
                safe_min(r['error_clusters_num'][False]),
                safe_max(r['error_clusters_num'][False]))],
        ['Between-cluster tree similarity: MEAN (STD)', 
            "{} ({})".format(safe_mean(r['error_clusters_sim'][True]), safe_std(r['error_clusters_sim'][True])),
            "{} ({})".format(safe_mean(r['error_clusters_sim'][False]), safe_std(r['error_clusters_sim'][False]))],
        ['Between-cluster tree similarity: MEDIAN (MIN - MAX)', 
            "{} ({} - {})".format(
                safe_median(r['error_clusters_sim'][True]),
                safe_min(r['error_clusters_sim'][True]),
                safe_max(r['error_clusters_sim'][True])),
            "{} ({} - {})".format(
                safe_median(r['error_clusters_sim'][False]),
                safe_min(r['error_clusters_sim'][False]),
                safe_max(r['error_clusters_sim'][False]))]
    ]
    print(AsciiTable(table_data).table)



