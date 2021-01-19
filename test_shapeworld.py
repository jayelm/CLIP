import torch
import clip
from PIL import Image
import os
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from visualize import vis


class AverageMeter:
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self, running_avg=False):
        self.reset()
        self.compute_running_avg = running_avg
        if self.compute_running_avg:
            self.reset_running_avg()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset_running_avg(self):
        self.running_val = 0
        self.running_avg = 0
        self.running_sum = 0
        self.running_count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if self.compute_running_avg:
            self.update_running_avg(val, n)

    def update_running_avg(self, val, n):
        self.running_val = val
        self.running_sum += val * n
        self.running_count += n
        self.running_avg = self.running_sum / self.running_count

    def __str__(self):
        return f"AverageMeter(mean={self.avg:f}, count={self.count:d})"

    def __repr__(self):
        return str(self)


class PosNeg:
    def __init__(self, pos_imgs, neg_imgs, langs, transform=None):
        self.pos_imgs = pos_imgs
        self.neg_imgs = neg_imgs
        self.langs = langs
        self.transform = transform

    def __getitem__(self, i):
        pi = Image.fromarray(self.pos_imgs[i])
        ni = Image.fromarray(self.neg_imgs[i])
        if self.transform is not None:
            pi = self.transform(pi)
            ni = self.transform(ni)

        lang = self.langs[i]

        return pi, ni, lang, i

    def __len__(self):
        return len(self.pos_imgs)


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description='test shapeworld',
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('dataset', nargs='?', help='which dataset to load', default='l3')

    args = parser.parse_args()

    train_file = os.path.join(args.dataset, 'test.npz')

    train = np.load(train_file)

    imgs = train['imgs']
    labels = train['labels']
    langs = train['langs']
    # 0th image is positive example. 1th image is negative example
    # get all negative examples
    is_negative = labels[:, -1] == 0

    pos_imgs, neg_imgs = imgs[is_negative, 0], imgs[is_negative, -1]
    pos_imgs = np.transpose(pos_imgs, (0, 2, 3, 1))
    neg_imgs = np.transpose(neg_imgs, (0, 2, 3, 1))
    langs = langs[is_negative]
    langs_pp = clip.tokenize(langs)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    data = PosNeg(pos_imgs, neg_imgs, langs_pp, transform=preprocess)
    dataloader = DataLoader(data, batch_size=32, num_workers=0)

    def score_encoded(model, image_features, text_features):
        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = model.logit_scale.exp()
        dot_products = torch.bmm(image_features.unsqueeze(1), text_features.unsqueeze(2))
        dot_products = dot_products.squeeze(-1).squeeze(-1)
        scores = logit_scale * dot_products.squeeze(-1)

        # shape = [global_batch_size, global_batch_size]
        return scores

    acc_meter = AverageMeter()
    for pos_img, neg_img, txt, idx in tqdm(dataloader, desc='eval'):
        batch_size = pos_img.shape[0]
        pos_img = pos_img.to(device)
        neg_img = neg_img.to(device)
        txt = txt.to(device)

        with torch.no_grad():
            pos_img_features = model.encode_image(pos_img)
            neg_img_features = model.encode_image(neg_img)
            txt_features = model.encode_text(txt)

            pos_scores = score_encoded(model, pos_img_features, txt_features)
            neg_scores = score_encoded(model, neg_img_features, txt_features)

        # Score for the positive image should be greater than the negative image.
        hits = pos_scores > neg_scores
        acc_meter.update(hits.float().mean(), batch_size)

    # Visualize the last batch
    print(acc_meter.avg)

    pos_img_raw = [Image.fromarray(data.pos_imgs[i]) for i in idx.numpy()]
    neg_img_raw = [Image.fromarray(data.neg_imgs[i]) for i in idx.numpy()]
    txt_raw = [langs[i] for i in idx.numpy()]

    comb_scores = torch.stack((pos_scores, neg_scores), 1).softmax(-1).cpu().numpy()
    hits = hits.cpu().numpy()
    exs = zip(pos_img_raw, neg_img_raw, txt_raw, comb_scores, hits)

    vis(exs, acc_meter.avg.item(), dname=args.dataset)
