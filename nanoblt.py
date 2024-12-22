import os
import logging
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Configuration
@dataclass
class BLTConfig:
    # Data and training
    vocab_size: int = 256
    seq_len: int = 1024
    batch_size: int = 4
    lr: float = 1e-3
    max_steps: int = 5000
    eval_interval: int = 500

    # Model dimension parameters
    dim: int = 256
    dim_local_encoder: int = None  # will be set to dim if None
    dim_local_decoder: int = None  # will be set to dim if None
    dim_token_emb: int = None  # will be set to dim // 2 if None
    dim_patch_emb: int = None  # will be computed if None

    # Architecture parameters
    num_layers_local_enc: int = 4
    num_layers_local_dec: int = 4
    num_layers_global: int = 6
    num_heads_global: int = 8
    n_heads_local_encoder: int = 8
    n_heads_local_decoder: int = 8
    patch_size: int = 32

    # Other parameters
    dropout: float = 0.1
    norm_eps: float = 1e-5

    # Optimization parameters
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    grad_clip: float = 1.0

    # Features (not implemented in MVP)
    use_dynamic_patching: bool = False
    n_gram_sizes: list = None

    # Commented out parameters (to be implemented in future versions)
    # attn_bias_type: str = None  # Future feature: type of attention bias to use
    # alpha_depth: float = 1.0  # Future feature: depth scaling factor for initialization
    # norm_type: str = 'rms'  # Future feature: type of normalization to use (e.g., 'rms' or 'layernorm')
    # norm_affine: bool = True  # Future feature: whether to use affine transformation in normalization
    # pre_norm: bool = True  # Future feature: whether to use pre-norm or post-norm in transformer blocks
    # patching_device: str = 'cuda'  # Future feature: device to use for patching operations
    # realtime_patching: bool = False  # Future feature: whether to use real-time patching during training
    # patching_batch_size: int = 1  # Future feature: batch size for patching operations
    # data_loader_patching: bool = False  # Future feature: whether to use data loader for patching
    # max_patch_length: int = 128  # Future feature: maximum patch length for dynamic patching
    # monotonicity: bool = False  # Future feature: whether to enforce monotonicity in patching
    # log_time: bool = False  # Future feature: whether to log time for various operations
    # output_size: int = None  # Future feature: size of the output layer if different from vocab_size
    # tie_local_encoder_decoder_logits: bool = False  # Future feature: tie embeddings between encoder and decoder
    # share_encoder_decoder_emb: bool = False  # Future feature: share embeddings between encoder and decoder
    # global_local_decoder_residual_layer: bool = False  # Future feature: add residual connection between global and local decoder
    # tokenize_with_bpe_delimiter: bool = False  # Future feature: use BPE delimiter for tokenization
    # patching_thresholds_str: str = None  # Future feature: thresholds for dynamic patching
    # tie_local_encoder_decoder: bool = False  # Future feature: tie weights between local encoder and decoder
    # encoder_preds_low_entropy_toks: int = 0  # Future feature: number of low entropy tokens to predict in encoder
    # encoder_preds_random_toks: int = 0  # Future feature: number of random tokens to predict in encoder
    # encoder_ngram_table_dir: str = None  # Future feature: directory for encoder n-gram table
    # encoder_ngram_to_size_str: str = None  # Future feature: string mapping n-grams to sizes
    # entropy_model_checkpoint_dir: str = None  # Future feature: directory for entropy model checkpoint
    # entropy_model_is_ngram_model: bool = False  # Future feature: whether entropy model is n-gram model
    # downsampling_by_pooling: bool = False  # Future feature: whether to use pooling for downsampling
    # n_kv_heads: int = None  # Future feature: number of key/value heads for grouped query attention
    # n_kv_heads_global: int = None  # Future feature: number of key/value heads for global attention
    # conv_kernel_size: int = 1  # Future feature: kernel size for convolutional layers
    # sequence_parallel: bool = False  # Future feature: whether to use sequence parallelism
    # loss_parallel: bool = False  # Future feature: whether to parallelize loss computation
    # fuse_sequence_parallel: bool = False  # Future feature: whether to fuse sequence parallel operations
    # use_fsdp: bool = False  # Future feature: whether to use Fully Sharded Data Parallel
    # attn_to_keep: float = 1.0  # Future feature: fraction of attention to keep
    # rope_use_fp32_in_outer_product: bool = False  # Future feature: use fp32 for rotary embedding outer product
    # full_logging_n_layers: int = 0  # Future feature: number of layers to log fully
    # eos_id: int = None  # Future feature: end of sequence token ID
    # cross_attn_encoder: bool = False  # Future feature: use cross attention in encoder
    # cross_attn_decoder: bool = False  # Future feature: use cross attention in decoder
    # cross_attn_window_encoder: int = 0  # Future feature: window size for encoder cross attention
    # cross_attn_window_decoder: int = 0  # Future feature: window size for decoder cross attention
    # cross_attn_k: int = 0  # Future feature: dimension of keys in cross attention
    # cross_attn_nheads: int = 0  # Future feature: number of heads in cross attention
    # cross_attn_all_layers_decoder: bool = False  # Future feature: use cross attention in all decoder layers
    # cross_attn_all_layers_encoder: bool = False  # Future feature: use cross attention in all encoder layers
    # cross_attn_use_flex_attention: bool = False  # Future feature: use flex attention for cross attention
    # cross_attn_init_by_pooling: bool = False  # Future feature: initialize cross attention by pooling
    # patch_only_encoder: bool = False  # Future feature: use patching only in encoder
    # patch_only_decoder: bool = False  # Future feature: use patching only in decoder
    # init_use_gaussian: bool = False  # Future feature: use Gaussian initialization
    # init_use_depth: bool = False  # Future feature: use depth-based initialization
    # multiple_of: int = 1  # Future feature: ensure dimensions are multiples of this value
    # ffn_dim_multiplier: float = 4.0  # Future feature: multiplier for FFN hidden dimension
    # rope_theta: float = 10000.0  # Future feature: base for rotary position embedding

    def __post_init__(self):
        self.dim_local_encoder = self.dim_local_encoder or self.dim
        self.dim_local_decoder = self.dim_local_decoder or self.dim
        self.dim_token_emb = self.dim_token_emb or self.dim // 2
        self.dim_patch_emb = self.dim_patch_emb or (self.patch_size * self.dim_token_emb)


# Data loading and preprocessing
def load_data(file_path):
    """
    Load data from a file.

    Args:
        file_path (str): Path to the data file.

    Returns:
        torch.Tensor: Tensor of loaded data.

    Raises:
        ValueError: If the data file is empty.
        FileNotFoundError: If the data file does not exist.

    Note:
        For large-scale data, this function could be adapted to use a data loader
        that reads data in chunks or implements a generator-based approach.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    with open(file_path, 'rb') as f:
        data = f.read()
    if len(data) == 0:
        raise ValueError("Data file is empty.")
    return torch.tensor(list(data), dtype=torch.long)


def get_batch(data, block_size, batch_size):
    """
    Get a batch of data.

    Args:
        data (torch.Tensor): The full dataset.
        block_size (int): The size of each sequence in the batch.
        batch_size (int): The number of sequences in the batch.

    Returns:
        tuple: A tuple (x, y) where x is the input sequences and y is the target sequences.

    Raises:
        ValueError: If the block size is too large for the given data.

    Note:
        For large datasets, this function could be adapted to use a data loader
        that reads from disk on-the-fly or implements a streaming approach.
    """
    if block_size + 1 > len(data):
        raise ValueError("Block size is too large for the given data.")
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


# Unimplemented functions as commented stubs
"""
def extract_ngrams(tokens, n_gram_sizes):
    # Implementation from the plan:
    # For each n in n_gram_sizes, consider sliding windows of length n.
    # Map them to IDs using hashing or a lookup table.
    # Merge the n-gram embeddings with the byte embeddings.
    # Fallback to smaller n-grams or individual bytes if necessary.
    pass

def calculate_local_entropy(byte_seq, window=32):
    # Implementation from the plan:
    # For each position i, consider the slice [i-window//2 : i+window//2].
    # Compute frequencies using a sliding window approach.
    # Calculate local entropy as -sum(p(byte) * log(p(byte))).
    pass

def create_dynamic_patches(x, entropy_threshold, max_patch_size=512):
    # Implementation from the plan:
    # 1) entropies = calculate_local_entropy(...)
    # 2) Start a new patch at position 0
    # 3) For i in range(1, len(x)):
    #    if entropies[i] > entropy_threshold or current_patch_length >= max_patch_size:
    #       Close the current patch and start a new one
    # 4) If the last patch is too small, merge it with the previous one
    # 5) Return patch boundaries for re-gathering enc_out into patch-level embeddings
    pass
"""


# Model Components
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization

    Note: In future versions, this could be toggled with LayerNorm based on the
    'norm_type' config parameter. The 'norm_affine' parameter could also be used
    to determine whether to use an affine transformation.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding

    Note: In future versions, the 'rope_theta' config parameter could be used
    to set the base value for the inverse frequency bands.
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiheadAttention(nn.Module):
    """
    Multihead Attention module

    Note: In future versions, 'n_kv_heads' could be used for grouped query attention,
    and 'attn_bias_type' could determine the type of attention bias to use.
    """

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads_global
        self.hidden_size = config.dim
        self.head_dim = self.hidden_size // self.num_heads

        self.wq = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.wk = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.wv = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.wo = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(self, x, mask=None):
        b, t, c = x.size()
        q = self.wq(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(q, seq_len=t)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if mask is not None:
            att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.wo(y)


class FeedForward(nn.Module):
    """
    Feed Forward module

    Note: In future versions, 'ffn_dim_multiplier' could be used to adjust
    the hidden dimension of the feed-forward layer.
    """

    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.dim, 4 * config.dim)
        self.w2 = nn.Linear(4 * config.dim, config.dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.relu(self.w1(x))))


class TransformerBlock(nn.Module):
    """
    Transformer Block

    Note: In future versions, 'recompute_fc1_out', 'recompute_fc3_out', 'recompute_attn'
    could be used for memory-efficient backpropagation.
    """

    def __init__(self, config):
        super().__init__()
        self.attn = MultiheadAttention(config)
        self.ff = FeedForward(config)
        self.ln1 = RMSNorm(config.dim, eps=config.norm_eps)
        self.ln2 = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class ByteEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.dim_token_emb)

    def forward(self, x):
        return self.embedding(x)


class LocalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers_local_enc)])
        self.project = nn.Linear(config.dim_token_emb, config.dim_local_encoder)

    def forward(self, x):
        x = self.project(x)
        for layer in self.layers:
            x = layer(x)
        return x


class GlobalTransformer(nn.Module):
    """
    Global Transformer module

    Note: In future versions, this module would use the output of 'create_dynamic_patches'
    for variable-sized patches instead of fixed-size patches. The 'downsampling_by_pooling'
    config parameter could also be used to determine the pooling strategy.
    """

    def __init__(self, config):
        super().__init__()
        self.patch_size = config.patch_size
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers_global)])

    def forward(self, x):
        B, T, D = x.shape
        num_patches = T // self.patch_size
        # Simple pooling for patch formation
        x = x.view(B, num_patches, self.patch_size, D).mean(dim=2)
        for layer in self.layers:
            x = layer(x)
        return x


class LocalDecoder(nn.Module):
    """
    Local Decoder module using standard cross-attention.

    Note: This approach doesn't expand memory; we let the cross-attention
    module handle the differences in sequence length between tokens (T)
    and patch embeddings (num_patches).
    """

    def __init__(self, config):
        super().__init__()

        self.num_layers = config.num_layers_local_dec
        self.dim_local_decoder = config.dim_local_decoder
        self.patch_size = config.patch_size

        # A standard nn.TransformerDecoder uses cross-attention under the hood
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.dim_local_decoder,
            nhead=config.n_heads_local_decoder,
            dropout=config.dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_layers)

        # Project the token-level embeddings to decoder dimension
        self.project_in = nn.Linear(config.dim_token_emb, config.dim_local_decoder)
        # Project the patch embeddings to decoder dimension
        self.project_mem = nn.Linear(config.dim_patch_emb, config.dim_local_decoder)

    def forward(self, tgt_tokens, patch_memory):
        """
        Args:
            tgt_tokens: Byte-level tokens shape (B, T).
            patch_memory: Patch-level embeddings shape (B, num_patches, dim_patch_emb).

        Returns:
            The decoded output, shape (B, T, dim_local_decoder), where
            cross-attention attends each of the T tokens to the num_patches memory.
        """

        B, T = tgt_tokens.shape
        # 1) Project token inputs to local decoder dimension
        tgt_emb = self.project_in(tgt_tokens)  # (B, T, dim_local_decoder)

        # 2) Project patch embeddings to local decoder dimension
        mem_emb = self.project_mem(patch_memory)  # (B, num_patches, dim_local_decoder)

        # 3) Generate a causal mask for the tokens if we want autoregressive decoding
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(tgt_tokens.device)

        # 4) Forward pass through the TransformerDecoder
        # The decoder internally does:
        #    - self-attention on (tgt_emb)
        #    - cross-attention on (mem_emb)
        out = self.decoder(
            tgt=tgt_emb,
            memory=mem_emb,
            tgt_mask=causal_mask
        )
        return out


class BLTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.byte_embedding = ByteEmbedding(config)
        self.local_encoder = LocalEncoder(config)
        self.global_transformer = GlobalTransformer(config)
        self.local_decoder = LocalDecoder(config)
        self.output_head = nn.Linear(config.dim_local_decoder, config.vocab_size)

    def forward(self, x, y=None):
        # Embeddings
        byte_emb = self.byte_embedding(x)

        # Local encoding
        enc_out = self.local_encoder(byte_emb)

        # Global transformer
        global_embs = self.global_transformer(enc_out)

        # Local decoder
        dec_out = self.local_decoder(byte_emb, global_embs)

        # Output logits and optional loss
        logits = self.output_head(dec_out)

        if y is None:
            loss = None
        else:
            B, T = y.shape
            logits = logits.view(B * T, -1)
            y = y.view(-1)
            loss = F.cross_entropy(logits, y)

        return logits, loss

    def configure_optimizers(self):
        """
        Configure the optimizer for the model.

        Returns:
            torch.optim.Optimizer: The configured optimizer.

        Note:
            This method uses AdamW with betas and weight decay as specified in the config.
            Future versions might incorporate more advanced optimization techniques or
            learning rate schedules.
        """
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.config.lr,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.config.weight_decay
        )


# Training and evaluation
def save_checkpoint(model, optimizer, step, path="checkpoint.pt"):
    """
    Save a checkpoint of the model and optimizer states.

    Args:
        model (nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        step (int): The current training step.
        path (str): The path to save the checkpoint to.

    Note:
        This function saves the model state dict, optimizer state dict, and current step.
        For distributed training scenarios, additional logic would be needed to handle
        saving and loading of distributed model states.
    """
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step
    }, path)
    logging.info(f"Saved checkpoint to {path}")


def load_checkpoint(model, optimizer, path="checkpoint.pt"):
    """
    Load a checkpoint of the model and optimizer states.

    Args:
        model (nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.
        path (str): The path to load the checkpoint from.

    Returns:
        int: The training step at which the checkpoint was saved.

    Note:
        For distributed training scenarios, additional logic would be needed to handle
        loading of distributed model states.
    """
    if not os.path.exists(path):
        return 0
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['step']


@torch.no_grad()
def evaluate(model, data, config):
    """
    Evaluate the model on the validation data.

    Args:
        model (nn.Module): The model to evaluate.
        data (torch.Tensor): The validation data.
        config (BLTConfig): The configuration object.

    Returns:
        float: The perplexity of the model on the validation data.

    Note:
        This function computes the average loss over 10 batches and returns the perplexity.
        Future versions might implement more sophisticated evaluation metrics or
        use a larger validation set.
    """
    model.eval()
    losses = []
    for _ in range(10):
        x, y = get_batch(data, config.seq_len, config.batch_size)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return torch.exp(torch.tensor(losses).mean())


@torch.no_grad()
def generate(model, start_ids, max_new_tokens=100):
    """
    Generate text from the model.

    Args:
        model (nn.Module): The trained model.
        start_ids (torch.Tensor): The starting token IDs.
        max_new_tokens (int): The maximum number of new tokens to generate.

    Returns:
        torch.Tensor: The generated token IDs.

    Note:
        This function uses simple greedy decoding. Future versions might implement
        more advanced decoding methods like top-k sampling or beam search.
    """
    x = start_ids.unsqueeze(0)
    for _ in range(max_new_tokens):
        logits, _ = model(x)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, next_id), dim=1)
    return x.squeeze()


def train(model, config, data):
    """
    Train the model.

    Args:
        model (nn.Module): The model to train.
        config (BLTConfig): The configuration object.
        data (torch.Tensor): The training data.

    Note:
        This function implements a simple training loop inspired by 'nanogpt.py'.
        It can be extended to include more advanced features like learning rate
        scheduling, gradient accumulation, etc. in future versions.
    """
    optimizer = model.configure_optimizers()
    step = load_checkpoint(model, optimizer)

    while step < config.max_steps:
        x, y = get_batch(data, config.seq_len, config.batch_size)
        logits, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        optimizer.step()

        if step % 100 == 0:
            logging.info(f"Step {step}, train loss: {loss.item():.4f}")

        if step % config.eval_interval == 0:
            ppl = evaluate(model, data, config)
            logging.info(f"Step {step}, validation perplexity: {ppl:.2f}")
            save_checkpoint(model, optimizer, step)

        step += 1


# Main
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = BLTConfig()

    # Check for data file
    data_file = "data.txt"
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")

    data = load_data(data_file)
    model = BLTModel(config)

    # Log basic info
    n_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model parameters: {n_params / 1e6:.2f}M")

    # Train the model
    train(model, config, data)

    # Generation example (after training)
    context = "Hello world"
    context_ids = torch.tensor([ord(c) for c in context])
    output_ids = generate(model, context_ids)
    output_text = ''.join([chr(i) for i in output_ids.tolist()])
    print(f"Generated text:\n{output_text}")