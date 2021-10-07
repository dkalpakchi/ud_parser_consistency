import argparse
import re
import sys
import json
from collections import defaultdict
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
        'same_as_first': 0,
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
        'error_clusters': []
    }
    for b in batches:
        r['original_match'] += b['original_match']
        r['all_correct'][b['original_match']] += b['corr_parsed'] == b['total']
        r['corr_parsed'][b['original_match']].append(b['corr_parsed'])

        if not b['original_match']:
            # Only for those where original was incorrectly parsed
            r['same_as_first'] += b['same_as_first'] == b['total']
            r['error_clusters'].append(
                find_cliques( convert_to_adjacency_list(b['conv_kernel_matrix']) )
            )

        # if b['corr_parsed'] != b['total']:
        #     r['all_same_as_first'][b['original_match']] += b['same_as_first'] == b['total']
        #     if b['same_as_first'] > 0:
        #         print(b)
        # r['deprel_mismatch'][b['original_match']] += len(b['deprel'][list(b['deprel'].keys())[0]].keys()) > 1
        # r['upos_mismatch'][b['original_match']] += len(b['upos'][list(b['upos'].keys())[0]].keys()) > 1
        # r['feats_mismatch'][b['original_match']] += len(b['feats'][list(b['feats'].keys())[0]].keys()) > 1

    print(r['total'])
    print(r['error_clusters'])
    table_data = [
        ['Metric', 'Original correctly parsed', 'Original incorrectly parsed'],
        ['Total', r['original_match'], r['total'] - r['original_match']],
        ['All correct in a batch', r['all_correct'][True], r['all_correct'][False]],
        ['Correctly parsed in a batch (out of 50): MEAN (SD)', 
            "{} ({})".format(round(np.mean(r['corr_parsed'][True]), 2), round(np.std(r['corr_parsed'][True]), 2)), 
            "{} ({})".format(round(np.mean(r['corr_parsed'][False]), 2), round(np.std(r['corr_parsed'][False]), 2)) ],
        ['Correctly parsed in a batch (out of 50): MEDIAN (MIN - MAX)', 
            "{} ({} - {})".format(np.median(r['corr_parsed'][True]), np.min(r['corr_parsed'][True]), np.max(r['corr_parsed'][True])), 
            "{} ({} - {})".format(np.median(r['corr_parsed'][False]), np.min(r['corr_parsed'][False]), np.max(r['corr_parsed'][False]))],
        ['Consistent errors', 'NA', r['same_as_first']],
        # ['Number of error clusters: MEAN (STD)', 'NA', "{} ({})".format(
        #     np.mean(r['error_clusters']), np.std(r['error_clusters']))],
        # ['Number of error clusters: MEDIAN (MIN - MAX)', 'NA', "{} ({} - {})".format(
        #     np.median(r['error_clusters']), np.min(r['error_clusters']), np.max(r['error_clusters']))]
    ]
    print(AsciiTable(table_data).table)



