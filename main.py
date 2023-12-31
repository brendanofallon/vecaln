
import os
import itertools
import random

import torch
from torch.optim import lr_scheduler
import torch.nn.functional as F
import numpy as np
import faiss
import pysam

from model import DNAEnc
from transforms import *

import byol_pytorch
import contrastive

DEVICE = torch.device("cuda") if hasattr(torch, 'cuda') and torch.cuda.is_available() else torch.device("cpu")


BASE_IDX = ['A', 'C', 'G', 'T']

BASEMAP = {
    'A': 0,
    'C': 1,
    'G': 2,
    'T': 3,
}

BASE_ONEHOT={
    'A': np.array([1, 0, 0, 0]),
    'C': np.array([0, 1, 0, 0]),
    'G': np.array([0, 0, 1, 0]),
    'T': np.array([0, 0, 0, 1]),
}

#REF_PATH="/Users/brendanofallon/data/ref/human_g1k_v37_decoy_phiXAdaptr.fasta.gz"
REF_PATH=os.getenv('REF_GENOME', "/Users/brendanofallon/data/ref/human_g1k_v37_decoy_phiXAdaptr.fasta.gz")
REF_GENOME=pysam.Fastafile(REF_PATH)


KMER_SIZE=1


def vectorize_onehot(c):
    return np.stack([BASE_ONEHOT[b] for b in c])


def base_labels(c):
    return np.stack([BASEMAP[b] for b in c])


def make_kmer_lookups(size):
    """
    Generate forward and reverse lookups for kmers
    str2index returns the index int for a given kmer,
    index2str returns the kmer string for a given index
    """
    bases = "ACGT"
    baselist = [bases] * size
    str2index = {}
    index2str = [None] * (len(bases) ** size)
    for i, combo in enumerate(itertools.product(*baselist)):
        s = ''.join(combo)
        str2index[s] = i
        index2str[i] = s
    return str2index, index2str


def pull_chunk():
    return REF_GENOME.fetch("1", 2000000, 2050000)


def vectorize_chunk(c, **kwargs):
    s2i = kwargs.get('kmer_lookup')
    return np.array([s2i[c[j:j + KMER_SIZE]] for j in range(0, len(c), KMER_SIZE)]).astype("uint8")


def vectors_from_chunk(chunk, chunksize, stepsize, **kwargs):
    s2i = kwargs.get('kmer_lookup')
    offsets = []
    vectors = []
    for i in range(0, len(chunk), stepsize):
        c = chunk[i:i+chunksize]
        if len(c) != chunksize:
            continue
        v = vectorize_chunk(c, kmer_lookup=s2i)
        offsets.append(i)
        vectors.append(v)
    return np.array(offsets), np.stack(vectors)


def vectorize_batch(batch):
    result = [torch.tensor(vectorize_onehot(b)) for b in batch]
    return torch.stack(result, dim=0).float().to(DEVICE)


def make_batch(batchsize, bases):
    batch = []
    labels = []
    chunk = pull_chunk()
    for b in range(batchsize):
        i = np.random.randint(0, len(chunk) - bases)
        s = chunk[i:i + bases]
        v = torch.tensor(vectorize_onehot(s))
        labels.append(torch.tensor(base_labels(s)))
        batch.append(v)

    return torch.stack(labels, dim=0).long(), torch.stack(batch, dim=0).float()

def raw_batch(batchsize, seqlen):
    batch = []
    chunk = pull_chunk()
    for b in range(batchsize):
        i = np.random.randint(0, len(chunk) - seqlen)
        batch.append(chunk[i:i + seqlen])
    return batch



def train_byol():
    net = DNAEnc(inlen=64).to(DEVICE)

    tr = SequentialTransform(
        transforms=[
            PickTransform(transforms=[SnpTransform(), InsTransform(), DelTransform()]),
            #RotateTransform(min_len=0, max_len=2),
            vectorize_batch,
        ])

    seqs = raw_batch(5, 64)
    y = net(tr(seqs))
    pcount = sum(p.numel() for p in net.parameters())
    print(f"Model has {pcount} tot params")

    # byol = byol_pytorch.BYOL(
    #     net=net,
    #     seq_len=64,
    #     augment_fn=vectorize_batch,
    #     hidden_layer=-1,
    #     augment_fn2=tr,
    #     projection_size=1024, # Default 256
    #     projection_hidden_size=1024 # Default 4096
    # ).to(DEVICE)

    cont = contrastive.Contrastive(model=net, transforms=vectorize_batch, augs=tr)

    opt = torch.optim.Adam(net.parameters(), lr=0.0002)

    orig = raw_batch(100, 64)
    snps = SnpTransform()(orig)
    dels = DelTransform()(orig)
    inss = InsTransform()(orig)
    rots = RotateTransform()(orig)
    rnd = raw_batch(100, 64)

    origt = vectorize_batch(orig)
    snpst = vectorize_batch(snps)
    delst = vectorize_batch(dels)
    insst = vectorize_batch(inss)
    rotst = vectorize_batch(rots)
    rndt = vectorize_batch(rnd)

    o = net(origt)
    s = net(snpst)
    # dsnp = byol_pytorch.loss_fn(o, s)

    warmup_iters = 400

    lrsched = lr_scheduler.SequentialLR(
        opt,
        schedulers=[lr_scheduler.LinearLR(opt, start_factor=0.00001, end_factor=0.999, total_iters=warmup_iters),
                    lr_scheduler.ExponentialLR(opt, gamma=0.9999)],
        milestones=[warmup_iters]
        )
    for t in range(10000):
        seqs = raw_batch(256, 64)
        neg_examples = vectorize_batch(raw_batch(256, 64))
        # loss = byol(seqs, negative_examples=neg_examples, negative_factor=2.0)
        loss = cont(seqs, neg_examples, neg_factor=t/1000.0)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        # byol.update_moving_average()
        lrsched.step()
        if t % 50 == 0:
            pre = f"Step {t} lr: {lrsched.get_last_lr()[0] :.5f} loss: {loss.item() :.3f} "

            with torch.no_grad():
                o = net(origt)
                rl = byol_pytorch.loss_fn(o, net(rndt)).mean().item()
                if rl < 0:
                    print(f"Whoa, random loss is < 0!: {rl}")
                msg = [pre]
                loss_ratios = []
                for key, val in zip(['snps', 'dels', 'ins', 'rots'], [snpst, delst, insst, rotst]):
                    l = byol_pytorch.loss_fn(o, net(val))
                    if l.mean().item() < 0:
                        print(f"Whoa, {key} loss is < 0! {l.mean().item()}")
                    loss_ratios.append(l.mean().item() / rl)
                    msg.append(f"{key} : {l.mean().item() / rl :.4f} ")
                print(''.join(msg))

            early_stop_threshold = 0.025
            if all(0 < r < early_stop_threshold for r in loss_ratios):
                print(f"Wow, all loss ratios are < {early_stop_threshold}, we are aborting")
                break


        if t > 500 and t % 1000 == 0:
            sd = net.state_dict()
            dest = f"vecaln_ep{t}_nf2_proj1024_1d.pyt"
            print(f"Saving model to {dest}")
            torch.save(sd, dest)


def load_ref_sequences(ref, chrom, start, end, step, seq_len, batchsize):
    batch = []
    starts = []
    for pos in range(start, end, step):
        seq = ref.fetch(chrom, pos, pos+seq_len)
        batch.append(seq)
        starts.append(pos)
        if len(batch) >= batchsize:
            yield batch, starts
            batch = []
            starts = []

    if batch:
        yield batch, starts


def build_index(index_dim, encoder, refpath, chrom, start, end, step, seq_len):
    ref = pysam.Fastafile(refpath)
    index = faiss.IndexFlatL2(index_dim)

    with torch.no_grad():
        offsets = []
        for batch, starts in load_ref_sequences(ref, chrom, start, end, step, seq_len, batchsize=64):
            bv = vectorize_batch(batch)
            batchenc = encoder(bv)
            index.add(batchenc.detach().cpu().numpy())
            offsets.extend(starts)

    print(f"Build index with {index.ntotal} entries")
    return index, offsets


def dist(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


def path_cost(path, rd, indices, scores):
    if len(path) == 1:
        return scores[path[0]]
    steps = len(path)
    scorecost = np.sum(scores[i, path[i]] for i in range(steps))
    distcost = np.sum((rd[i-1] - (indices[i, path[i]] - indices[i-1, path[i-1]])) for i in range(1, steps))
    return scorecost + distcost


def map_chunk(read, index, net, inlen=64):
    readpos = [0, 25, 50, 75, 150-64]
    with torch.no_grad():
        subreads = vectorize_batch([read[i:i+inlen] for i in readpos])
        emb = net(subreads).cpu().numpy()
        scores, indices = index.search(emb, k=4)
        scores = scores * 1e7
        print(scores)
        print(indices)

        rd = np.array(readpos[1:]) - readpos[0]
        cost = path_cost([0, 0, 2, 0], rd, indices, scores)
    return dist, indices


def infer():
    random.seed(13276)
    sd = torch.load("vecaln_ep6000_nf2_proj1024_1d.pyt", map_location='cpu')
    net = DNAEnc(inlen=64)
    net.load_state_dict(sd)
    net.eval()
    index, offsets = build_index(index_dim=32, encoder=net, refpath=REF_PATH, chrom="2", start=2000000, end=2050000, step=1, seq_len=64)
    # print(idx)

    query = REF_GENOME.fetch("2", 2000100, 2000250)
    query = DelTransform()(query)
    bigchunk = REF_GENOME.fetch("2", 2000000, 2000256)
    d, i = map_chunk(query, index, net, inlen=64)


    # for i in range(len(bigchunk)):
    #     t = bigchunk[i:i+64]
    #     vb = vectorize_batch([query, t])
    #     emb = net(vb)
    #
    #     d = dist(emb[0, :], emb[1,:]).detach().numpy()
    #     print(f"{i}: {d * 1e6 :.6f}")

    # e = net(qb)
    # D, I = idx.search(e.detach().cpu().numpy(), k=4)
    # print(I)
    # print(D)


if __name__=="__main__":
    train_byol()
    # infer()
