import os
import sys
from glob import glob
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    raw_train_folder = sys.argv[1]
    train_outfolder = sys.argv[2]

    train_files = glob(os.path.join(raw_train_folder, '*'))
    print('Getting subfolder images...')
    train_imgs = [glob(os.path.join(t, '*.jpg')) for t in tqdm(train_files)]
    print('Getting train identities...')
    train_ids = [t.split(os.sep)[-1] for t in tqdm(train_files)]

    os.makedirs(train_outfolder, exist_ok=True)
    train_img_scp_path = os.path.join(train_outfolder, 'img.scp')
    train_img2id_path = os.path.join(train_outfolder, 'img2id')
    train_id2img_path = os.path.join(train_outfolder, 'id2img')

    all_train_imgs = np.concatenate(train_imgs)
    print('Formatting image ids...')
    tr_imgs = ['{}_{}'.format(x.split(os.sep)[-2], os.path.basename(x[:-4])) for x in tqdm(all_train_imgs)]

    with open(train_img_scp_path, 'w+') as fp:
        for img, file in zip(tr_imgs, all_train_imgs):
            line = '{} {}\n'.format(img, file)
            fp.write(line)

    with open(train_img2id_path, 'w+') as fp:
        for img in tr_imgs:
            iden = img.split('_')[0]
            line = '{} {}\n'.format(img, iden)
            fp.write(line)

    with open(train_id2img_path, 'w+') as fp:
        for iden, files in zip(train_ids, train_imgs):
            timgs = ['_'.join([iden, i.split(os.sep)[-1][:-4]]) for i in files]
            line = '{} {}\n'.format(iden, ' '.join(timgs))
            fp.write(line)
