import os
import csv
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import spacy
from torchtext.vocab import build_vocab_from_iterator, Vocab

from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
from network import Transformer
from torch.utils.tensorboard import SummaryWriter

# ============ Tokenizers ============
spacy_zh = spacy.load("zh_core_web_sm")
spacy_en = spacy.load("en_core_web_sm")

def tokenize_zh(text: str) -> List[str]:
    return [tok.text for tok in spacy_zh.tokenizer(text)]

def tokenize_en(text: str) -> List[str]:
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

# ============ Special tokens ============
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"

# ============ Dataset ============
class TSVSeq2SeqDataset(Dataset):
    def __init__(self, tsv_path: str, src_tokenize, trg_tokenize):
        self.examples: List[Tuple[List[str], List[str]]] = []
        self.src_tok = src_tokenize
        self.trg_tok = trg_tokenize
        with open(tsv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                src = [SOS_TOKEN] + self.src_tok(row["src"]) + [EOS_TOKEN]
                trg = [SOS_TOKEN] + self.trg_tok(row["trg"]) + [EOS_TOKEN]
                self.examples.append((src, trg))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

# ============ Vocab builders ============
def yield_tokens(dataset: Dataset, idx: int):
    # idx=0 for src, 1 for trg
    for src, trg in dataset:
        yield src if idx == 0 else trg

def build_vocab(dataset: Dataset, max_size: int = 10000, min_freq: int = 2) -> Vocab:
    vocab = build_vocab_from_iterator(
        yield_tokens(dataset, 0),  # dummy, replaced by caller
        specials=[PAD_TOKEN, SOS_TOKEN, EOS_TOKEN],
        special_first=True,
        max_tokens=None  # we’ll trim manually
    )
    return vocab

def build_src_trg_vocabs(train_ds: Dataset, max_size=10000, min_freq=2):
    src_vocab = build_vocab_from_iterator(
        yield_tokens(train_ds, 0),
        specials=[PAD_TOKEN, SOS_TOKEN, EOS_TOKEN],
        special_first=True,
        min_freq=min_freq,
    )
    trg_vocab = build_vocab_from_iterator(
        yield_tokens(train_ds, 1),
        specials=[PAD_TOKEN, SOS_TOKEN, EOS_TOKEN],
        special_first=True,
        min_freq=min_freq,
    )
    src_vocab.set_default_index(src_vocab[PAD_TOKEN])
    trg_vocab.set_default_index(trg_vocab[PAD_TOKEN])

    # Enforce max_size (keep most frequent)
    def trim(v: Vocab, max_size: int):
        if max_size is None:
            return v
        itos = v.get_itos()
        if len(itos) <= max_size:
            return v
        # keep specials + most frequent remaining
        specials = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]
        specials_idx = [itos.index(tok) for tok in specials]
        specials_set = set(specials_idx)
        # frequency info exists in v.vocab (Counter-like)
        freqs = v.vocab
        # Build tokens sorted by frequency (excluding specials)
        tokens = [t for t in itos if t not in specials]
        tokens.sort(key=lambda t: -freqs[t])
        keep = specials + tokens[: max_size - len(specials)]
        # rebuild
        new_v = build_vocab_from_iterator([keep], specials=[], special_first=True)
        new_v.set_default_index(v[PAD_TOKEN] if PAD_TOKEN in v else 0)
        return new_v

    src_vocab = trim(src_vocab, max_size)
    trg_vocab = trim(trg_vocab, max_size)
    return src_vocab, trg_vocab

# ============ Numericalization and Collate ============
def numericalize(tokens: List[str], vocab: Vocab) -> torch.Tensor:
    return torch.tensor([vocab[token] for token in tokens], dtype=torch.long)

def collate_fn(batch, src_vocab: Vocab, trg_vocab: Vocab, device: torch.device):
    src_seqs = [numericalize(src, src_vocab) for src, _ in batch]
    trg_seqs = [numericalize(trg, trg_vocab) for _, trg in batch]

    src_padded = pad_sequence(src_seqs, padding_value=src_vocab[PAD_TOKEN])
    trg_padded = pad_sequence(trg_seqs, padding_value=trg_vocab[PAD_TOKEN])

    return src_padded.to(device), trg_padded.to(device)

# ============ Load data ============
data_dir = "data/zh_en"
train_path = os.path.join(data_dir, "train.tsv")
valid_path = os.path.join(data_dir, "valid.tsv")
test_path  = os.path.join(data_dir, "test.tsv")

train_ds = TSVSeq2SeqDataset(train_path, tokenize_zh, tokenize_en)
valid_ds = TSVSeq2SeqDataset(valid_path, tokenize_zh, tokenize_en)
test_ds  = TSVSeq2SeqDataset(test_path,  tokenize_zh, tokenize_en)

src_vocab, trg_vocab = build_src_trg_vocabs(train_ds, max_size=10000, min_freq=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32

train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=lambda b: collate_fn(b, src_vocab, trg_vocab, device),
)

valid_loader = DataLoader(
    valid_ds,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=lambda b: collate_fn(b, src_vocab, trg_vocab, device),
)

test_loader = DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=lambda b: collate_fn(b, src_vocab, trg_vocab, device),
)

# ============ Model setup (unchanged except vocab refs) ============
src_vocab_size = len(src_vocab)
trg_vocab_size = len(trg_vocab)

embedding_size = 512
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10
max_len = 100
forward_expansion = 4

src_pad_idx = src_vocab[PAD_TOKEN]
trg_pad_idx = trg_vocab[PAD_TOKEN]

model = Transformer(
    embedding_size,
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    forward_expansion,
    dropout,
    max_len,
    device,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=True)
criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

# ============ Optional checkpoint ============
load_model = True
save_model = True

if load_model:
    try:
        ckpt = torch.load("my_checkpoint.pth.tar", map_location=device)
        load_checkpoint(ckpt, model, optimizer)
    except FileNotFoundError:
        print("No checkpoint found, starting from scratch.")

# ============ Training loop (adapted to DataLoader) ============
writer = SummaryWriter("runs/loss_plot")
step = 0
num_epochs = 10000

sentence = "一匹馬走在 bridge 旁邊的船下面。"

def tensor_to_tokens(t: torch.Tensor, vocab: Vocab) -> List[str]:
    itos = vocab.get_itos()
    return [itos[i] for i in t.tolist()]

for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")

    if save_model:
        checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        save_checkpoint(checkpoint)

    model.eval()
    # translate_sentence expects legacy Field objects; we provide wrappers with needed attrs
    class DummyField:
        def __init__(self, vocab, tokenize):
            self.vocab = vocab
            self.init_token = SOS_TOKEN
            self.eos_token = EOS_TOKEN
            self.tokenize = tokenize
            self.vocab.stoi = {tok: i for i, tok in enumerate(vocab.get_itos())}
            self.vocab.itos = vocab.get_itos()
    chinese = DummyField(src_vocab, tokenize_zh)
    english = DummyField(trg_vocab, tokenize_en)

    translated_sentence = translate_sentence(model, sentence, chinese, english, device, max_length=50)
    print(f"Translated example sentence:\n {translated_sentence}")

    model.train()
    losses = []

    for batch_idx, (inp_data, target) in enumerate(train_loader):
        # inp_data, target shapes: [seq_len, batch]
        output = model(inp_data, target[:-1, :])  # teacher forcing
        output = output.reshape(-1, output.shape[2])  # [(seq_len-1)*batch, vocab]
        target_flat = target[1:].reshape(-1)         # [(seq_len-1)*batch]

        optimizer.zero_grad()
        loss = criterion(output, target_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        losses.append(loss.item())
        writer.add_scalar("Training loss", loss.item(), global_step=step)
        step += 1

    mean_loss = sum(losses) / max(1, len(losses))
    scheduler.step(mean_loss)

# ============ BLEU ============
# If your bleu() expects legacy datasets, adapt by creating a list of (src, trg) pairs:
test_pairs = []
for src_tokens, trg_tokens in test_ds:
    test_pairs.append((" ".join(src_tokens[1:-1]), " ".join(trg_tokens[1:-1])))

score = bleu(test_pairs[:100], model, chinese, english, device)
print(f"Bleu score {score * 100:.2f}")