import argparse
import os
import re
import json
from collections import defaultdict
from pathlib import Path
from pprint import pprint

from numpy.random import default_rng
import udon2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--treebank', type=str, required=True, help="A path to UD treebank directory")
    args = parser.parse_args()

    treebank_dir = Path(args.treebank)

    for treebank in treebank_dir.glob("*.conllu"):
        print(treebank)
        prefix = treebank.stem

        YEAR = re.compile(r"(?<= )\d{4}(?= )")

        trees = udon2.Importer.from_conll_file(str(treebank))
        print("Total # of trees: {}".format(len(trees)))

        comments = []

        finalists = []
        for t in trees:
            sent = t.get_subtree_text()
            comments.append(sent)
            match = YEAR.search(sent)
            if match:
                finalists.append((t, match))

        # Years used for testing later
        rng = default_rng(7919) # seed sequence is set the same for reproducibility -- 100th prime number
        test_years = rng.integers(1100, 2100, size=(50,))

        # Years used for training
        rng = default_rng(7907) # seed sequence is set the same for reproducibility -- 99th prime number
        train_years = rng.integers(1100, 2100, size=(100,))
        train_years = [y for y in train_years if y not in test_years]
        print("Found {} years for augmentation".format(len(train_years)))

        for f, match in finalists:
            sentence = f.get_subtree_text()
            s, e = match.span()
            current_year = sentence[s:e]
            f1 = f.copy()
            for y in train_years[:20]:
                nodes = f1.select_by('form', current_year)
                y_str = str(y)
                for node in nodes:
                    node.form = y_str
                    node.lemma = y_str
                current_year = y_str

                trees.append(f1.copy())
                comments.append(f1.get_subtree_text())

        print("Total # of trees after augmentation: {}".format(len(trees)))
        udon2.ConllWriter.write_to_file(trees, "augm_{}.conllu".format(prefix))

        # Additional code to add `# text` comment in front of every recorded tree as required by Stanza
        new_sentence, cur_idx = True, 0
        with open("augm_{}.conllu".format(prefix)) as f1, open("augmented_{}.conllu".format(prefix), "w") as f2:
            for line in f1:
                if not line.strip():
                    new_sentence = True
                    f2.write(line)
                    continue
                if new_sentence:
                    f2.write("# text = {}\n".format(comments[cur_idx]))
                    new_sentence = False
                    cur_idx += 1
                f2.write(line)
        os.remove("augm_{}.conllu".format(prefix))