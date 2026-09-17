import os
import pandas as pd
from asr_ensemble import ensemble_text_lists


def tst_ensemble(input_files, weights, out_path):
    # Load data from all files into a dictionary grouped by audio UID

    texts = []
    ids = None
    for f in input_files:
        df = pd.read_csv(f)
        df = df.sort_values(by='ID').reset_index(drop=True)
        df.fillna("", inplace=True)
        texts.append(df['Target'])
        if ids is None:
            ids = tuple(df['ID'].values)
        else:
            if ids != tuple(df['ID'].values):
                print("Different IDs order!")
                exit()

    texts_list = []
    for i in range(len(texts[0])):
        texts_list.append([])
        for j in range(len(texts)):
            texts_list[i].append(texts[j][i])

    text_pred = ensemble_text_lists(
        texts_list,
        normalize=False,
        char_level=True,
        weights=weights,
        ensemble_type='median_extended',
        language=None,
        max_workers=8,
        verbose=True,
    )
    text_pred = [t.strip() for t in text_pred]
    df['Target'] = text_pred
    df.to_csv(out_path, index=False)
    print("Saved to {}".format(out_path))


if __name__ == "__main__":
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
    folder = root_path + 'submission/'
    subm_list = [
        folder + 'submission_whisper_salt.csv',
        folder + 'submission_facebook_mms.csv',
        folder + 'submission_wav2vec.csv',
        folder + 'submission_whisper.csv',
        folder + 'submission_qwen.csv',
        folder + 'submission_parakeet.csv',
        folder + 'submission_ibm_granite.csv',
        folder + 'submission_whisper_salt_student.csv',
    ]
    out_path = folder + 'submission_final_{}.csv'.format(len(subm_list))
    weights = [10.0, 12.0, 11.0, 9.0, 6.0, 4.0, 4.0, 10.0]
    tst_ensemble(subm_list, weights, out_path)
