"""Convert the `# %%` percent-format sources into .ipynb files for Kaggle (File -> Import Notebook)."""
import json
import re
import sys

for py in sys.argv[1:] or ['01_preprocess.py', '02_train_eval.py', '03_fusion_ensemble.py']:
    cells = []
    for chunk in re.split(r'^# %%', open(py).read(), flags=re.M)[1:]:
        head, _, body = chunk.partition('\n')
        if '[markdown]' in head:
            cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': re.sub(r'^# ?', '', body, flags=re.M).strip()})
        else:
            cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': body.strip()})
    nb = {'cells': cells, 'nbformat': 4, 'nbformat_minor': 4,
          'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}}}
    json.dump(nb, open(py[:-3] + '.ipynb', 'w'), indent=1)
    print(py[:-3] + '.ipynb', len(cells), 'cells')
