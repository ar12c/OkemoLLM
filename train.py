# train.py -- M2 Max / MPS-safe Transformer training script
# - disables torch.compile on MPS
# - uses autocast on MPS (no GradScaler there)
# - uses boolean causal mask and matching key_padding_mask types
# - target shifting, padding, gradient accumulation included

import os
import re
import sys
import math
import time
import random
import warnings
import multiprocessing
from functools import partial
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import load_dataset
import nltk
from nltk.tokenize import sent_tokenize
from huggingface_hub.errors import LocalEntryNotFoundError
import requests
import urllib3

# ----------------- QoL / macOS warnings -----------------
try:
    warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)
except Exception:
    pass

# ----------------- Reproducibility -----------------
def set_seed(seed: int = 1337):
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

set_seed(1337)

# ----------------- NLTK -----------------
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizers...")
    nltk.download('punkt')

# ----------------- Tokenizer -----------------
_token_re = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)

def tokenize(text: str):
    return _token_re.findall(text.lower())

# ----------------- Vocabulary -----------------
class Voc:
    def __init__(self):
        self.PAD, self.SOS, self.EOS, self.UNK = "<PAD>", "<SOS>", "<EOS>", "<UNK>"
        self.word2index = {self.PAD: 0, self.SOS: 1, self.EOS: 2, self.UNK: 3}
        self.index2word = {0: self.PAD, 1: self.SOS, 2: self.EOS, 3: self.UNK}
        self.word_count = defaultdict(int)
        self.n_words = 4

    def add_tokens(self, tokens):
        for w in tokens:
            self.word_count[w] += 1

    def trim(self, min_count):
        keep = [w for w, c in self.word_count.items() if c >= min_count]
        print(f"Keeping {len(keep)} of {len(self.word_count)} total words with min_count={min_count}")
        self.word2index = {self.PAD: 0, self.SOS: 1, self.EOS: 2, self.UNK: 3}
        self.index2word = {0: self.PAD, 1: self.SOS, 2: self.EOS, 3: self.UNK}
        self.n_words = 4
        for w in keep:
            self.word2index[w] = self.n_words
            self.index2word[self.n_words] = w
            self.n_words += 1

# ----------------- Data Loading & Processing -----------------
def load_and_process_local_data(data_path):
    print(f"Loading local data from {data_path}...")
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        print(f"❌ Error: The file '{data_path}' was not found.")
        sys.exit(1)

    pairs = []
    cur = []
    for line in lines:
        if line.strip():
            text = line.split("]", 1)[-1].strip() if "]" in line else line.strip()
            cur.append(text)
        else:
            if len(cur) > 1:
                for i in range(len(cur) - 1):
                    pairs.append([cur[i], cur[i + 1]])
            cur = []
    if len(cur) > 1:
        for i in range(len(cur) - 1):
            pairs.append([cur[i], cur[i + 1]])
    return pairs

def load_and_process_wikipedia_data(num_samples):
    print(f"Loading {num_samples} articles from Wikipedia dataset...")
    max_retries = 5
    dataset = None
    for attempt in range(max_retries):
        try:
            dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split=f"train[:{num_samples}]")
            break
        except (requests.exceptions.ConnectionError, LocalEntryNotFoundError) as e:
            print(f"❌ Download failed on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                delay = 2 ** (attempt + 1)
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("❌ Max retries exceeded. Please check your connection.")
                sys.exit(1)

    pairs = []
    for article in dataset:
        text = article.get("text", "") or ""
        sentences = sent_tokenize(text)
        for i in range(len(sentences) - 1):
            s1, s2 = sentences[i].strip(), sentences[i + 1].strip()
            if s1 and s2:
                pairs.append([s1, s2])
    return pairs

def load_all_data(local_data_path, num_wikipedia_samples):
    local_pairs = load_and_process_local_data(local_data_path)
    wiki_pairs = load_and_process_wikipedia_data(num_wikipedia_samples)
    print(f"✅ Loaded {len(local_pairs)} pairs from local data.")
    print(f"✅ Loaded {len(wiki_pairs)} pairs from Wikipedia data.")
    all_pairs = local_pairs + wiki_pairs
    print(f"✅ Total conversational pairs: {len(all_pairs)}")
    return all_pairs

def filter_pairs(pairs, max_length):
    out = []
    for a, b in pairs:
        ta, tb = tokenize(a), tokenize(b)
        if len(ta) < max_length and len(tb) < max_length:
            out.append((ta, tb))
    return out

def build_vocabulary(pairs, min_count):
    voc = Voc()
    for ta, tb in pairs:
        voc.add_tokens(ta)
        voc.add_tokens(tb)
    voc.trim(min_count)
    return voc

def tensor_from_tokens(voc: Voc, tokens, add_sos=False):
    idxs = [voc.word2index.get(tok, voc.word2index[voc.UNK]) for tok in tokens]
    if add_sos:
        idxs = [voc.word2index[voc.SOS]] + idxs
    idxs = idxs + [voc.word2index[voc.EOS]]
    return torch.tensor(idxs, dtype=torch.long)

def pad_sequence(seqs, pad_idx):
    max_len = max(s.size(0) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_idx, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : s.size(0)] = s
    return out

def collate_fn(batch_data, voc: Voc):
    src_tensors = [tensor_from_tokens(voc, ta, add_sos=False) for ta, _ in batch_data]
    tgt_tensors = [tensor_from_tokens(voc, tb, add_sos=True) for _, tb in batch_data]
    src = pad_sequence(src_tensors, voc.word2index[voc.PAD])
    tgt = pad_sequence(tgt_tensors, voc.word2index[voc.PAD])
    return src, tgt

# ----------------- Model -----------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]

class TransformerEncoder(nn.Module):
    def __init__(self, n_words, d_model, n_layers, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(n_words, d_model)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, n_head, d_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)

    def forward(self, src, src_key_padding_mask=None):
        x = self.embedding(src)
        x = self.pos(x)
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)

class TransformerDecoder(nn.Module):
    def __init__(self, n_words, d_model, n_layers, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(n_words, d_model)
        self.pos = PositionalEncoding(d_model)
        dec_layer = nn.TransformerDecoderLayer(d_model, n_head, d_ff, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, n_layers)
        self.fc = nn.Linear(d_model, n_words)

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        t = self.embedding(tgt)
        t = self.pos(t)
        out = self.decoder(
            t, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        return self.fc(out)

# ----------------- Training utils -----------------
def make_bool_causal_mask(sz, device):
    # True = masked out (compatible with key_padding_mask boolean type)
    return torch.triu(torch.ones((sz, sz), dtype=torch.bool, device=device), diagonal=1)

def train_step(input_tensor, target_tensor, encoder, decoder, criterion,
               device, voc, use_autocast, scaler, accumulation_steps):
    decoder_input = target_tensor[:, :-1]    # includes <SOS>
    decoder_target = target_tensor[:, 1:]    # shifted targets

    src_key_padding_mask = (input_tensor == voc.word2index[voc.PAD])       # (B, S) bool
    tgt_key_padding_mask = (decoder_input == voc.word2index[voc.PAD])     # (B, T) bool
    tgt_mask = make_bool_causal_mask(decoder_input.size(1), device)       # (T, T) bool

    if use_autocast:
        # For MPS/CUDA we use autocast to reduce memory. GradScaler only on CUDA (scaler != None).
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            memory = encoder(input_tensor, src_key_padding_mask=src_key_padding_mask)
            logits = decoder(decoder_input, memory,
                             tgt_mask=tgt_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             memory_key_padding_mask=src_key_padding_mask)
            loss = criterion(logits.reshape(-1, logits.size(-1)), decoder_target.reshape(-1))
            loss = loss / accumulation_steps
    else:
        memory = encoder(input_tensor, src_key_padding_mask=src_key_padding_mask)
        logits = decoder(decoder_input, memory,
                         tgt_mask=tgt_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=src_key_padding_mask)
        loss = criterion(logits.reshape(-1, logits.size(-1)), decoder_target.reshape(-1))
        loss = loss / accumulation_steps

    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    return loss.item() * accumulation_steps  # return effective loss (before division)

def save_checkpoint(model_dir, encoder, decoder, enc_opt, dec_opt, voc: Voc):
    print("Saving model checkpoint...")
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "checkpoint.pth")
    torch.save({
        "encoder_state_dict": encoder.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "encoder_optimizer_state_dict": enc_opt.state_dict(),
        "decoder_optimizer_state_dict": dec_opt.state_dict(),
        "voc_word2index": voc.word2index,
        "voc_index2word": voc.index2word,
        "special_tokens": {"PAD": voc.PAD, "SOS": voc.SOS, "EOS": voc.EOS, "UNK": voc.UNK},
    }, path)
    print(f"✅ Model saved to {path}")

# ----------------- Main -----------------
def main():
    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"Using device: {device}")

    # M2-friendly defaults (tweak if you know what you're doing)
    data_file_path = "combined_data.txt"
    num_wikipedia_samples = 1000        # start small on MPS, raise later
    d_model = 192
    n_head = 4                         # even number to avoid nested-tensor warning
    d_ff = 384
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1
    n_epochs = 20
    learning_rate = 0.000001
    max_length = 1000
    print_every_steps = 200
    save_every_epochs = 5
    min_count = 3
    batch_size = 4                     # small on MPS to avoid kernel complexity
    accumulation_steps = 2             # effective batch = 8
    num_workers_to_use = 0
    pin_memory = (device.type == "cuda")

    # Mixed precision / scaler policy:
    # - autocast used on MPS and CUDA (device.type check)
    # - GradScaler only used on CUDA (safer)
    use_autocast = device.type in ["cuda", "mps"]
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # Do NOT compile on MPS (torch.compile can produce symbolic-size kernels that crash)
    use_compile = torch.cuda.is_available()  # only true if CUDA is present

    # Load data
    all_pairs = load_all_data(data_file_path, num_wikipedia_samples)
    filtered_pairs = filter_pairs(all_pairs, max_length=max_length)
    print(f"Found {len(filtered_pairs)} conversational pairs after filtering.")

    # Build vocab
    voc = build_vocabulary(filtered_pairs, min_count)
    print(f"Vocabulary size: {voc.n_words}")

    # DataLoader
    data_loader = DataLoader(
        filtered_pairs,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, voc=voc),
        num_workers=num_workers_to_use,
        pin_memory=pin_memory,
        drop_last=True,
    )
    print(f"DataLoader created with batch_size={batch_size}, workers={num_workers_to_use}.")

    # Models
    encoder = TransformerEncoder(voc.n_words, d_model, encoder_n_layers, n_head, d_ff, dropout).to(device)
    decoder = TransformerDecoder(voc.n_words, d_model, decoder_n_layers, n_head, d_ff, dropout).to(device)

    # Try compile only on CUDA
    if use_compile:
        try:
            encoder = torch.compile(encoder)
            decoder = torch.compile(decoder)
            print("Models compiled with torch.compile (CUDA).")
        except Exception as e:
            print(f"torch.compile failed or is disabled: {e}")
    else:
        print("Skipping torch.compile on non-CUDA device (recommended for MPS).")

    enc_opt = optim.Adam(encoder.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    dec_opt = optim.Adam(decoder.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=voc.word2index[voc.PAD])

    print("Starting training...")
    global_step = 0

    for epoch in range(1, n_epochs + 1):
        epoch_start = time.time()
        total_epoch_loss = 0.0

        enc_opt.zero_grad(set_to_none=True)
        dec_opt.zero_grad(set_to_none=True)

        for batch_idx, (src, tgt) in enumerate(data_loader):
            src = src.to(device, non_blocking=pin_memory)
            tgt = tgt.to(device, non_blocking=pin_memory)

            loss_val = train_step(src, tgt, encoder, decoder, criterion,
                                  device, voc, use_autocast, scaler, accumulation_steps)
            total_epoch_loss += loss_val
            global_step += 1

            # Optimizer step after accumulation_steps micro-batches
            if (batch_idx + 1) % accumulation_steps == 0:
                if scaler is not None:
                    # CUDA path with GradScaler
                    scaler.step(enc_opt)
                    scaler.step(dec_opt)
                    scaler.update()
                else:
                    enc_opt.step()
                    dec_opt.step()
                enc_opt.zero_grad(set_to_none=True)
                dec_opt.zero_grad(set_to_none=True)

            if global_step % print_every_steps == 0:
                avg = total_epoch_loss / (batch_idx + 1)
                print(f"[Epoch {epoch}/{n_epochs} | Step {global_step} | Batch {batch_idx+1}/{len(data_loader)}] "
                      f"Avg Loss: {avg:.4f}")

        avg_epoch = total_epoch_loss / len(data_loader)
        dur = time.time() - epoch_start
        print(f"Epoch {epoch}/{n_epochs} done in {dur:.2f}s | Avg Loss: {avg_epoch:.4f}")

        if epoch % save_every_epochs == 0:
            save_checkpoint("model_mps_safe", encoder, decoder, enc_opt, dec_opt, voc)

    print("✅ Training complete.")
    save_checkpoint("model_mps_safe", encoder, decoder, enc_opt, dec_opt, voc)


if __name__ == "__main__":
    # Use spawn to avoid sometimes-broken forking behavior on macOS
    multiprocessing.set_start_method('spawn', force=True)
    main()