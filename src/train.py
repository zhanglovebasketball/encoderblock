import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import TransformerEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random, numpy as np, os, requests
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
    random.seed(seed)#设置Python内置随机数生成器
    np.random.seed(seed)#设置NumPy随机数生成器
    torch.manual_seed(seed)#设置PyTorch CPU随机种子
    torch.cuda.manual_seed_all(seed)#设置PyTorch GPU随机种子

def get_tiny_shakespeare():
    os.makedirs("../data", exist_ok=True)
    path = "../data/tiny_shakespeare.txt"
    if not os.path.exists(path):
        print("📥 下载 tiny_shakespeare.txt ...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        r = requests.get(url)
        with open(path, "w", encoding="utf-8") as f:
            f.write(r.text)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    # 截取一小部分，训练快
    return text[:400000]


def plot_losses(train_losses, val_losses, save_path="../results/train_val_curve.png"):
    """绘制训练和验证损失曲线"""
    plt.figure(figsize=(10, 6))

    # 绘制损失曲线
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)

    # 设置图表属性
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"损失曲线已保存到: {save_path}")
def main():
    # 加载配置
    cfg = load_config()
    batch_size = cfg['batch_size']
    learning_rate = cfg['learning_rate']
    epochs = cfg['epochs']
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
    #分词
    data = [torch.tensor(tokenizer.encode(s), dtype=torch.long) for s in samples]
    #填充并形成输入输出对
    def collate_fn(batch, pad_id=0):
        batch = nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=pad_id)
        return batch[:, :-1], batch[:, 1:]

    train_data, val_data, test_data = prepare_datasets_by_sequences(data)

    train_loader = DataLoader(train_data, batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size, shuffle=False, collate_fn=collate_fn)
    model = TransformerEncoder(vocab_size=tokenizer.vocab_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model_path = "../results/best_model.pt"
    if os.path.exists(model_path):
        print("📥 加载之前训练的最佳模型...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ 模型加载成功！")
    else:
        print("🆕 未找到预训练模型，从头开始训练")
    optimizer = optim.AdamW(model.parameters(), lr=float(learning_rate),weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # 训练循环（包含验证）
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

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
            torch.save(model.state_dict(), "../results/best_model.pt")
            print("  保存最佳模型！")

    # 最终测试
    model.load_state_dict(torch.load("../results/best_model.pt"))
    test_loss = evaluate_model(model, test_loader, criterion, device)
    print(f"最终测试损失: {test_loss:.4f}")

    # 绘制损失曲线
    plot_losses(train_losses, val_losses)


def evaluate_model_comprehensive(model, loader, criterion, device, tokenizer):
    """综合评估模型"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    correct_tokens = 0
    correct_top5 = 0

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            out = model(x)
            loss = criterion(out.view(-1, out.size(-1)), y.reshape(-1))
            total_loss += loss.item()

            # 计算准确率
            predictions = out.argmax(dim=-1)

            # 现在pad_token_id应该有效了
            mask = (y != tokenizer.pad_token_id)
            print(f"Debug: mask shape: {mask.shape}")  # 现在应该正常了

            correct_tokens += ((predictions == y) & mask).sum().item()
            total_tokens += mask.sum().item()

            # 计算Top-5准确率
            top5_pred = out.topk(5, dim=-1).indices
            top5_correct = torch.any(top5_pred == y.unsqueeze(-1), dim=-1)
            correct_top5 += (top5_correct & mask).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
    top5_accuracy = correct_top5 / total_tokens if total_tokens > 0 else 0
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'accuracy': accuracy,
        'top5_accuracy': top5_accuracy,
        'total_tokens': total_tokens
    }

def generate_text_simple(model, tokenizer, device, prompt_tokens, max_length=50):
    """简化版文本生成"""
    model.eval()
    generated = prompt_tokens.clone().to(device)

    with torch.no_grad():
        for _ in range(max_length):
            if generated.size(1) >= 512:  # 不超过模型最大长度
                break

            outputs = model(generated)
            next_token_logits = outputs[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            # 如果生成了结束符则停止
            if next_token.item() == tokenizer.eos_token_id:
                break

            generated = torch.cat([generated, next_token], dim=1)

    # 解码整个生成序列（包含原始提示）
    full_text = tokenizer.decode(generated[0].cpu().numpy(), skip_special_tokens=True)
    return full_text


def evaluate_with_text_comparison(model, loader, criterion, device, tokenizer):
    """评估并包含文本生成对比"""
    # 先进行数值评估
    metrics = evaluate_model_comprehensive(model, loader, criterion, device, tokenizer)

    # 添加文本生成对比
    model.eval()
    with torch.no_grad():
        # 取第一个batch的第一个样本
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # 原始文本
            original_tokens = x[0]
            # 去掉padding
            non_padding_tokens = original_tokens[original_tokens != tokenizer.pad_token_id]
            if len(non_padding_tokens) > 10:
                original_text = tokenizer.decode(non_padding_tokens.cpu().numpy())

                # 使用前10个token作为提示生成文本
                prompt_tokens = non_padding_tokens[:10]
                prompt_text = tokenizer.decode(prompt_tokens.cpu().numpy())

                # 改进的生成：使用温度调节和top-k采样
                generated_tokens = prompt_tokens.unsqueeze(0).to(device)
                for _ in range(50):  # 生成50个token
                    outputs = model(generated_tokens)
                    next_token_logits = outputs[:, -1, :]

                    # 温度调节和top-k采样
                    next_token_logits = next_token_logits / 0.8  # 温度=0.8
                    top_k = 40
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')
                    import torch.nn.functional as F
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                    generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

                    if next_token.item() == tokenizer.eos_token_id:
                        break

                generated_text = tokenizer.decode(generated_tokens[0].cpu().numpy())

                metrics['text_comparison'] = {
                    'prompt': prompt_text,
                    'original': original_text[:150] + '...' if len(original_text) > 150 else original_text,
                    'generated': generated_text
                }
            break  # 只处理第一个样本

    return metrics

if __name__ == "__main__":
    #使用示例
    # text = get_tiny_shakespeare()
    # from transformers import GPT2TokenizerFast
    # tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token  # 使用eos_token作为pad_token
    # print(f"Debug: pad_token_id after fix: {tokenizer.pad_token_id}")
    # # 把文本切块成小句子
    # def split_text_by_tokens(text, tokenizer, max_tokens=128, overlap=32):
    #     """按token边界切块，避免切断单词"""
    #     # 先tokenize整个文本
    #     tokens = tokenizer.encode(text)
    #
    #     chunks = []
    #     start = 0
    #     while start < len(tokens):
    #         # 取一个块
    #         end = min(start + max_tokens, len(tokens))
    #         chunk_tokens = tokens[start:end]
    #
    #         # 解码回文本（确保完整性）
    #         chunk_text = tokenizer.decode(chunk_tokens)
    #         chunks.append(chunk_text)
    #
    #         # 滑动窗口，使用重叠
    #         start += (max_tokens - overlap)
    #
    #     return chunks
    # # 使用改进的切块
    # samples = split_text_by_tokens(text, tokenizer, max_tokens=128, overlap=32)
    # # 分词
    # data = [torch.tensor(tokenizer.encode(s), dtype=torch.long) for s in samples]
    #
    #
    # # 填充并形成输入输出对
    # def collate_fn(batch, tokenizer):
    #     """使用tokenizer的pad_token_id"""
    #     batch = nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=tokenizer.pad_token_id)
    #     return batch[:, :-1], batch[:, 1:]
    #
    # train_data, val_data, test_data = prepare_datasets_by_sequences(data)
    # criterion = nn.CrossEntropyLoss()
    # batch_size=32
    # train_loader = DataLoader(train_data, batch_size, shuffle=True,
    #                           collate_fn=lambda b: collate_fn(b, tokenizer))
    # val_loader = DataLoader(val_data, batch_size, shuffle=False,
    #                         collate_fn=lambda b: collate_fn(b, tokenizer))
    # test_loader = DataLoader(test_data, batch_size, shuffle=False,
    #                          collate_fn=lambda b: collate_fn(b, tokenizer))
    # model = TransformerEncoder(vocab_size=tokenizer.vocab_size)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    # model.load_state_dict(torch.load("../results/best_model.pt"))
    # metrics = evaluate_with_text_comparison(model, test_loader, criterion, device, tokenizer)
    #
    # # 打印结果
    # print(f"最终测试损失: {metrics['loss']:.4f}")
    # print(f"困惑度: {metrics['perplexity']:.2f}")
    # print(f"准确率: {metrics['accuracy']:.4f}")
    # print(f"Top-5准确率: {metrics['top5_accuracy']:.4f}")
    main()