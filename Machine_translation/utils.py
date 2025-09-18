import torch
import spacy
import sacrebleu
import sys


def translate_sentence(model, sentence, src_field, trg_field, device, max_length=50):
    """
    model: your seq2seq/transformer model
    sentence: list of tokens (Chinese tokens if src=zh)
    src_field: tokenizer/vocab for source language
    trg_field: tokenizer/vocab for target language
    """
    model.eval()

    # numericalize
    src_indexes = [src_field.vocab.stoi[token] for token in sentence]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

    with torch.no_grad():
        encoder_outputs = model.encoder(src_tensor)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_length):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)

        with torch.no_grad():
            output = model.decoder(trg_tensor, encoder_outputs)

        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    return trg_tokens[1:]  # remove <sos>


# -----------------------------
# BLEU Score with SacreBLEU
# -----------------------------
def bleu(data, model, src_field, trg_field, device, max_examples=100):
    """
    data: dataset with (src, trg) pairs
    """
    preds = []
    refs = []

    for idx, example in enumerate(data):
        if idx > max_examples:
            break

        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence(model, src, src_field, trg_field, device)

        preds.append(" ".join(prediction))
        refs.append([" ".join(trg)])  # sacrebleu expects list of lists

    bleu = sacrebleu.corpus_bleu(preds, refs)
    return bleu.score


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])