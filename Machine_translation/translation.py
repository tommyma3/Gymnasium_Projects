import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
from network import Transformer
from torch.utils.tensorboard import SummaryWriter
from torchtext.data import Field, BucketIterator, TabularDataset

spacy_zh = spacy.load("zh_core_web_sm")
spacy_en = spacy.load("en_core_web_sm")

def tokenize_zh(text):
    return [tok.text for tok in spacy_zh.tokenizer(text)]

def tokenize_en(text):
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

chinese = Field(tokenize=tokenize_zh, lower=False, init_token="<sos>", eos_token="<eos>")
english = Field(tokenize=tokenize_en, lower=True, init_token="<sos>", eos_token="<eos>")

train_data, valid_data, test_data = TabularDataset.splits(
    path="data/zh_en",      # folder with train.tsv, valid.tsv, test.tsv
    train="train.tsv",
    validation="valid.tsv",
    test="test.tsv",
    format="tsv",
    fields=[("src", chinese), ("trg", english)],
)

chinese.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_model = True
save_model = True

num_epochs = 10000
learning_rate = 3e-4
batch_size = 32

src_vocab_size = len(chinese.vocab)
trg_vocab_size = len(english.vocab)
embedding_size = 512
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10
max_len = 100
forward_expansion = 4

src_pad_idx = chinese.vocab.stoi["<pad>"]

writer = SummaryWriter(f"runs/loss_plot")
step = 0

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device,
)

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

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=10, verbose=True
)

pad_idx = english.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if load_model:
    try:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)
    except FileNotFoundError:
        print("No checkpoint found, starting from scratch.")

sentence = "一匹馬走在 bridge 旁邊的船下面。"

for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")

    if save_model:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

    model.eval()
    translated_sentence = translate_sentence(
        model, sentence, chinese, english, device, max_length=50
    )

    print(f"Translated example sentence: \n {translated_sentence}")
    model.train()
    losses = []

    for batch_idx, batch in enumerate(train_iterator):
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        output = model(inp_data, target[:-1, :])

        output = output.reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()

        loss = criterion(output, target)
        losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1

    mean_loss = sum(losses) / len(losses)
    scheduler.step(mean_loss)

# Evaluate BLEU on a small slice (you can adapt this)
score = bleu(test_data[1:100], model, chinese, english, device)
print(f"Bleu score {score * 100:.2f}")