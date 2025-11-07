import matplotlib
matplotlib.use('Agg')
import os, torch, matplotlib.pyplot as plt
from model import TransformerEncoder
from torch.utils.data import DataLoader
import torch.nn as nn, torch.optim as optim
from transformers import GPT2TokenizerFast
import random, numpy as np, requests
from utils.config import load_config
def prepare_datasets_by_sequences(data, train_ratio=0.8, val_ratio=0.1,seed=42):
    """按序列分割，避免切断上下文"""
    # 设置随机种子确保可重复性
    random.seed(seed)
    # 先打乱数据顺序
    shuffled_data = data.copy()  # 创建副本避免修改原数据
    random.shuffle(shuffled_data)
    total_sequences = len(shuffled_data)
    train_size = int(total_sequences * train_ratio)
    val_size = int(total_sequences * val_ratio)

    train_data = shuffled_data[:train_size]
    val_data = shuffled_data[train_size:train_size + val_size]
    test_data = shuffled_data[train_size + val_size:]

    return train_data, val_data, test_data

def evaluate_model(model, loader, criterion, device):
    """在测试集上评估模型"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out.view(-1, out.size(-1)), y.reshape(-1))
            total_loss += loss.item()
    return total_loss / len(loader)
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_tiny_shakespeare():
    os.makedirs("data", exist_ok=True)
    path = "../data/tiny_shakespeare.txt"
    if not os.path.exists(path):
        print("📥 下载 tiny_shakespeare.txt ...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        r = requests.get(url)
        with open(path, "w", encoding="utf-8") as f:
            f.write(r.text)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text[:400000]

def collate_fn(batch, pad_id=0):
    batch = nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=pad_id)
    return batch[:, :-1], batch[:, 1:]

def run_experiment(use_positional=True, num_heads=4, label="base"):
    # 加载配置
    cfg = load_config()
    batch_size = cfg['batch_size']
    learning_rate = cfg['learning_rate']
    set_seed(42)
    text = get_tiny_shakespeare()
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # 把文本切块成小句子
    def split_text_by_tokens(text, tokenizer, max_tokens=128, overlap=32):
        """按token边界切块，避免切断单词"""
        # 先tokenize整个文本
        tokens = tokenizer.encode(text)

        chunks = []
        start = 0
        while start < len(tokens):
            # 取一个块
            end = min(start + max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]

            # 解码回文本（确保完整性）
            chunk_text = tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)

            # 滑动窗口，使用重叠
            start += (max_tokens - overlap)

        return chunks

    # 使用改进的切块
    samples = split_text_by_tokens(text, tokenizer, max_tokens=128, overlap=32)
    # 分词
    data = [torch.tensor(tokenizer.encode(s), dtype=torch.long) for s in samples]

    # 填充并形成输入输出对
    def collate_fn(batch, pad_id=0):
        batch = nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=pad_id)
        return batch[:, :-1], batch[:, 1:]

    train_data, val_data, test_data = prepare_datasets_by_sequences(data)

    train_loader = DataLoader(train_data, batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size, shuffle=False, collate_fn=collate_fn)

    model = TransformerEncoder(vocab_size=tokenizer.vocab_size, num_heads=num_heads)
    if not use_positional:
        model.pos = nn.Identity()  # 移除位置编码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=float(learning_rate))
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    epochs=60
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out.view(-1, out.size(-1)), y.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 验证阶段
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out.view(-1, out.size(-1)), y.reshape(-1))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch + 1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

        # 早停机制
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "../results/best_model_ablation.pt")
            print("  保存最佳模型！")

    # 最终测试
    model.load_state_dict(torch.load("../results/best_model_ablation.pt"))
    test_loss = evaluate_model(model, test_loader, criterion, device)
    print(f"最终测试损失: {test_loss:.4f}")

    return train_losses

def main():
    os.makedirs("../results", exist_ok=True)
    runs = {
        "no_pos": run_experiment(use_positional=False, label="no_pos"),
    }

    plt.figure()
    for k, v in runs.items():
        plt.plot(v, label=k)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Ablation Study: Effect of Position Encoding and #Heads")
    plt.legend()
    plt.savefig("../results/ablation.png")


if __name__ == "__main__":
    main()



