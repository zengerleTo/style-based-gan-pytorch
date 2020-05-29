import argparse
from io import BytesIO
import multiprocessing
from functools import partial

from PIL import Image
import lmdb
from tqdm import tqdm
from torchvision import datasets
from torchvision.transforms import functional as trans_fn


def resize_worker(img_file):
    i, file = img_file
    img = Image.open(file)
    img = img.convert('RGB')

    return i, img


def prepare(transaction, dataset, n_worker, size=128):
    resize_fn = partial(resize_worker)

    files = sorted(dataset.imgs, key=lambda x: x[0])
    files = [(i, file) for i, (file, label) in enumerate(files)]
    total = 0

    with multiprocessing.Pool(n_worker) as pool:
        for i, imgs in tqdm(pool.imap_unordered(resize_fn, files)):
            for img in imgs:
                key = f'{size}-{str(i).zfill(5)}'.encode('utf-8')
                transaction.put(key, img)

            total += 1

        transaction.put('length'.encode('utf-8'), str(total).encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str)
    parser.add_argument('--n_worker', type=int, default=8)
    parser.add_argument('path', type=str)
    parser.add_argument('--resolution', type=int, default=128)

    args = parser.parse_args()

    imgset = datasets.ImageFolder(args.path)

    with lmdb.open(args.out, map_size=args.resolution, readahead=False) as env:
        with env.begin(write=True) as txn:
            prepare(txn, imgset, args.n_worker, args.resolution)
