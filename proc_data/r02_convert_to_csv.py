import glob
import os
import json
import pandas as pd


def convert_preds(subm_path):
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
    df = pd.read_csv(root_path + 'input/Test_phase2.csv')

    in1 = open(subm_path, 'r', encoding='utf-8')
    lines = in1.readlines()
    in1.close()
    res_pred = {}
    for line in lines:
        arr = json.loads(line)
        res_pred[arr['id']] = str(arr['text'])

    df['Target'] = df['ID'].map(res_pred)
    df.to_csv(subm_path[:-6] + '.csv', index=False)


if __name__ == "__main__":
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
    files = glob.glob(root_path + 'submission/*.jsonl')
    print("Found {} jsonl files".format(len(files)))
    for f in files:
        convert_preds(f)

