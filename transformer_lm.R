# ============================================================
# 迷你 Transformer 字符级语言模型 (GPT-style, Decoder-Only)
# 从零手写实现 —— 核心展示 Self-Attention 机制
# ============================================================
# 课设：生成式模型探索
# 使用 R torch 包，不依赖任何预置 Transformer 模块
# ============================================================

# ============================================================
# 0. 环境准备
# ============================================================
cat("\n========== 环境准备 ==========\n")

if (!require("torch")) {
  install.packages("torch")
  library(torch)
}

# 检查 LibTorch + Lantern 是否已安装
tryCatch({
  torch_tensor(c(1, 2, 3))
}, error = function(e) {
  if (grepl("Lantern is not loaded", e$message, fixed = TRUE)) {
    stop("R 会话中 torch 状态过期。请按 Ctrl+Shift+F10 重启 R，然后重新 source 本脚本。")
  }
  stop("LibTorch 未安装或不可用！请在 RStudio Console 中运行：\n  torch::install_torch()\n",
       "下载约 800MB，仅需运行一次。完成后重新执行本脚本。\n原始错误: ", e$message)
})

cat("torch 包版本:", as.character(packageVersion("torch")), "\n")

# ============================================================
# 1. 数据准备 — 内置唐诗数据集 + 字符级 Tokenization
# ============================================================
cat("\n========== 数据准备 ==========\n")

# ============================================================
# 训练语料加载（内置唐诗 + 可选外部 .txt 文件）
# ============================================================
EOS_TOKEN <- "█"  # 终止符（单个特殊字符，不存在于唐诗中）

# 内置唐诗库（60 首，覆盖五言/七言绝句律诗多种体裁）
tang_poems <- c(
  # === 五言绝句 ===
  "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
  "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
  "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。",
  "锄禾日当午，汗滴禾下土。谁知盘中餐，粒粒皆辛苦。",
  "千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪。",
  "鹅鹅鹅，曲项向天歌。白毛浮绿水，红掌拨清波。",
  "松下问童子，言师采药去。只在此山中，云深不知处。",
  "移舟泊烟渚，日暮客愁新。野旷天低树，江清月近人。",
  "红豆生南国，春来发几枝。愿君多采撷，此物最相思。",
  "空山不见人，但闻人语响。返景入深林，复照青苔上。",
  "泠泠七弦上，静听松风寒。古调虽自爱，今人多不弹。",
  "归山深浅去，须尽丘壑美。莫学武陵人，暂游桃源里。",
  "孤云将野鹤，岂向人间住。莫买沃洲山，时人已知处。",
  "三日入厨下，洗手作羹汤。未谙姑食性，先遣小姑尝。",
  "君自故乡来，应知故乡事。来日绮窗前，寒梅著花未。",
  "山中相送罢，日暮掩柴扉。春草年年绿，王孙归不归。",
  # === 七言绝句 ===
  "日照香炉生紫烟，遥看瀑布挂前川。飞流直下三千尺，疑是银河落九天。",
  "朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。",
  "两个黄鹂鸣翠柳，一行白鹭上青天。窗含西岭千秋雪，门泊东吴万里船。",
  "月落乌啼霜满天，江枫渔火对愁眠。姑苏城外寒山寺，夜半钟声到客船。",
  "远上寒山石径斜，白云生处有人家。停车坐爱枫林晚，霜叶红于二月花。",
  "故人西辞黄鹤楼，烟花三月下扬州。孤帆远影碧空尽，唯见长江天际流。",
  "烟笼寒水月笼沙，夜泊秦淮近酒家。商女不知亡国恨，隔江犹唱后庭花。",
  "独在异乡为异客，每逢佳节倍思亲。遥知兄弟登高处，遍插茱萸少一人。",
  "渭城朝雨浥轻尘，客舍青青柳色新。劝君更尽一杯酒，西出阳关无故人。",
  "寒雨连江夜入吴，平明送客楚山孤。洛阳亲友如相问，一片冰心在玉壶。",
  "春城无处不飞花，寒食东风御柳斜。日暮汉宫传蜡烛，轻烟散入五侯家。",
  "银烛秋光冷画屏，轻罗小扇扑流萤。天阶夜色凉如水，卧看牵牛织女星。",
  "少小离家老大回，乡音无改鬓毛衰。儿童相见不相识，笑问客从何处来。",
  "葡萄美酒夜光杯，欲饮琵琶马上催。醉卧沙场君莫笑，古来征战几人回。",
  "黄河远上白云间，一片孤城万仞山。羌笛何须怨杨柳，春风不度玉门关。",
  "秦时明月汉时关，万里长征人未还。但使龙城飞将在，不教胡马度阴山。",
  # === 五言律诗 ===
  "国破山河在，城春草木深。感时花溅泪，恨别鸟惊心。",
  "离离原上草，一岁一枯荣。野火烧不尽，春风吹又生。",
  "好雨知时节，当春乃发生。随风潜入夜，润物细无声。",
  "细草微风岸，危樯独夜舟。星垂平野阔，月涌大江流。",
  "青山横北郭，白水绕东城。此地一为别，孤蓬万里征。",
  "渡远荆门外，来从楚国游。山随平野尽，江入大荒流。",
  "空山新雨后，天气晚来秋。明月松间照，清泉石上流。",
  "单车欲问边，属国过居延。征蓬出汉塞，归雁入胡天。",
  "楚塞三湘接，荆门九派通。江流天地外，山色有无中。",
  # === 七言律诗 ===
  "昔人已乘黄鹤去，此地空余黄鹤楼。黄鹤一去不复返，白云千载空悠悠。",
  "风急天高猿啸哀，渚清沙白鸟飞回。无边落木萧萧下，不尽长江滚滚来。",
  "丞相祠堂何处寻，锦官城外柏森森。映阶碧草自春色，隔叶黄鹂空好音。",
  # === 附加名句 ===
  "海上生明月，天涯共此时。情人怨遥夜，竟夕起相思。",
  "春蚕到死丝方尽，蜡炬成灰泪始干。晓镜但愁云鬓改，夜吟应觉月光寒。",
  "相见时难别亦难，东风无力百花残。春蚕到死丝方尽，蜡炬成灰泪始干。",
  "醉里挑灯看剑，梦回吹角连营。八百里分麾下炙，五十弦翻塞外声。"
)

# 外部 .txt 文件支持：放在同目录下自动加载
# 每行一首诗/一段文本即可，脚本会自动拼接
external_file <- "d:/R/RStudio/works/poems_extra.txt"
if (file.exists(external_file)) {
  cat(sprintf(">>> 发现外部语料文件: %s\n", external_file))
  ext_lines <- readLines(external_file, encoding = "UTF-8", warn = FALSE)
  ext_texts <- ext_lines[nchar(trimws(ext_lines)) > 0]  # 跳过空行
  tang_poems <- c(tang_poems, ext_texts)
  cat(sprintf(">>> 已加载 %d 行外部文本\n", length(ext_texts)))
}

# 每首诗末尾拼接 EOS 终止符
full_text <- paste(paste0(tang_poems, EOS_TOKEN), collapse = "")

# 提取所有唯一字符（EOS 会被自动包含）
all_chars <- strsplit(full_text, "")[[1]]
unique_chars <- sort(unique(all_chars))

cat(sprintf("训练文本总字符数: %d\n", length(all_chars)))
cat(sprintf("唯一字符数 (vocab_size): %d\n", length(unique_chars)))

vocab_size <- length(unique_chars)

# 模型实际词表大小（+1 因为索引 0 保留，实际使用 1:vocab_size）
vocab_size_model <- vocab_size + 1L

# 字符 ↔ 索引 映射（1-index，兼容 R torch 的 nn_embedding）
# 索引 0 保留不用（避免 R torch 的 1-base 索引报错）
char2idx <- setNames(1:vocab_size, unique_chars)

# EOS 的索引位置（用于 generate 中判断停止）
EOS_IDX <- as.integer(char2idx[EOS_TOKEN])
cat(sprintf("EOS 索引: %d\n", EOS_IDX))

# 索引 → 字符
decode_chars <- function(indices) {
  paste(unique_chars[as.integer(indices)], collapse = "")
}

# 整段文本编码为整数序列
encoded <- torch_tensor(as.integer(char2idx[all_chars]), dtype = torch_long())
cat(sprintf("编码后序列长度: %d\n", length(encoded)))

# 划分训练集/验证集 (80% / 20%)
n_total <- length(encoded)
split_idx <- as.integer(n_total * 0.8)
train_data <- encoded[1:split_idx]
val_data   <- encoded[(split_idx + 1):n_total]

cat(sprintf("训练集大小: %d, 验证集大小: %d\n",
            length(train_data), length(val_data)))

# ============================================================
# 2. 超参数配置
# ============================================================
cat("\n========== 超参数配置 ==========\n")

block_size    <- 64L    # 上下文窗口长度
batch_size    <- 16L    # 批量大小
d_model       <- 64L    # 嵌入维度
n_heads       <- 2L     # 注意力头数
n_layers      <- 2L     # Transformer 层数
d_ff          <- 256L   # FeedForward 中间层维度
dropout_rate  <- 0.1    # Dropout
learning_rate <- 3e-4   # 学习率
warmup_iters  <- 200L   # warmup 步数
lr_min        <- 3e-5   # 最低学习率
max_iters     <- 3000L  # 训练迭代次数
eval_interval <- 300L   # 每隔多少步评估一次

cat(sprintf("d_model=%d | n_heads=%d | n_layers=%d | block_size=%d\n",
            d_model, n_heads, n_layers, block_size))

# ============================================================
# 3. 批量数据获取函数
# ============================================================
get_batch <- function(split = "train") {
  # 随机采样 batch_size 个起始位置，构造 (input, target) 对
  data <- if (split == "train") train_data else val_data
  n_data <- as.integer(length(data))
  max_start <- n_data - block_size      # 最后一个合法起始位置

  # 随机起始索引（1-base，直接兼容 R torch 张量索引）
  starts <- as.integer(torch_randint(
    1L, max_start + 1L, as.integer(c(batch_size)), dtype = torch_long()
  ))

  # 用循环构造 batch — 避免 lapply+torch_stack 可能导致的 0-index 问题
  x <- torch_zeros(c(batch_size, block_size), dtype = torch_long())
  y <- torch_zeros(c(batch_size, block_size), dtype = torch_long())
  for (j in seq_along(starts)) {
    s <- starts[j]
    x[j, ] <- data[s:(s + block_size - 1L)]
    y[j, ] <- data[(s + 1L):(s + block_size)]
  }

  list(x = x, y = y)
}

# ============================================================
# 4. 位置编码 (Sinusoidal Positional Encoding)
# ============================================================
# 公式：PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
#       PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
# 不使用可学习的 Embedding，而是固定的正弦/余弦波
# 这样即使序列比训练时更长，编码也能自然外推

PositionalEncoding <- nn_module(
  "PositionalEncoding",
  initialize = function(max_len, d_model) {
    # 位置索引 [0, 1, 2, ..., max_len-1]
    position <- 0:(max_len - 1)

    # 维度频率项：1 / 10000^(2i / d_model)，i = 0, 1, ..., d_model/2 - 1
    i <- 0:(d_model/2 - 1)
    div_term <- 1 / (10000 ^ (2 * i / d_model))

    # 角度矩阵: outer(position, div_term) → [max_len, d_model/2]
    angles <- outer(position, div_term)

    # 填充 sin 到偶数维，cos 到奇数维
    pe <- matrix(0, nrow = max_len, ncol = d_model)
    pe[, 2 * (1:(d_model/2)) - 1] <- sin(angles)   # 2i-1 → 1,3,5,...
    pe[, 2 * (1:(d_model/2))]     <- cos(angles)   # 2i   → 2,4,6,...

    # 注册为 buffer（非训练参数，但随模型移动设备）
    self$pe <- torch_tensor(pe)$unsqueeze(1)  # [1, max_len, d_model]
  },

  forward = function(x) {
    # x: [B, T, d_model]  — token embedding 的输出
    seq_len <- dim(x)[2]
    # 截取前 seq_len 个位置，相加
    x + self$pe[, 1:seq_len, , drop = FALSE]
  }
)

# ============================================================
# 5. 缩放点积注意力 (Scaled Dot-Product Attention)
# ============================================================
# ★ 这是整个 Transformer 的核心！★
#
# Attention(Q, K, V) = softmax( Q·K^T / √d_k  +  mask ) · V
#
# 直觉解释（以语言模型为例）：
# ──────────────────────────────────────────────
# Q (Query):   "我现在是什么词，我想查询什么信息？"
# K (Key):     "序列中每个位置的词，能提供什么信息？"
# V (Value):   "序列中每个位置的词，实际包含什么内容？"
#
# 1. Q·K^T → 计算 Query 和所有 Key 的"相似度分数"
#    每个位置对序列中所有其他位置打分，形成 [T, T] 的分数矩阵
#
# 2. /√d_k → 缩放：防止 d_k 过大时，点积方差太大导致
#    softmax 梯度消失（梯度流到"极软"区域）
#
# 3. + causal_mask → 加上因果遮罩：
#    位置 t 只能看到 ≤ t 的位置（不能偷看"未来"）
#    上三角填 -∞，经 softmax 后变为 0
#
# 4. softmax → 将分数转成概率分布（每行求和 = 1）
#    每个位置对过去所有位置的注意力权重
#
# 5. ·V → 用注意力权重对 Value 加权求和
#    输出 = "综合了过去所有相关位置信息的新表示"
# ──────────────────────────────────────────────

scaled_dot_product_attention <- function(Q, K, V, causal_mask) {
  # Q, K, V: [B, n_heads, T, head_dim]
  d_k <- dim(Q)[4]  # head_dim

  # Step 1+2: 计算缩放点积分数
  # Q @ K^T  →  [B, n_heads, T, T]
  scores <- torch_matmul(Q, K$transpose(3, 4)) / sqrt(d_k)

  # Step 3: 叠加因果遮罩（上三角为 -∞）
  scores <- scores + causal_mask

  # Step 4: Softmax 归一化 → 注意力权重
  attn_weights <- nnf_softmax(scores, dim = 4)

  # Step 5: 注意力加权求和
  output <- torch_matmul(attn_weights, V)  # [B, n_heads, T, head_dim]

  list(output = output, attn_weights = attn_weights)
}

# ============================================================
# 6. 多头自注意力 (Multi-Head Self-Attention)
# ============================================================
# 将 d_model 拆分成 n_heads 个"头"，每个头独立做注意力
# 不同头关注不同模式（如一个头关注句法，另一个关注语义）
# 最后拼接所有头的结果

MultiHeadSelfAttention <- nn_module(
  "MultiHeadSelfAttention",
  initialize = function(d_model, n_heads, block_size, dropout = 0.1) {
    self$d_model  <- d_model
    self$n_heads  <- n_heads
    self$head_dim <- as.integer(d_model / n_heads)  # 每个头的维度
    stopifnot(d_model %% n_heads == 0)  # 必须整除

    # 三个独立的线性投影：输入 → Q, K, V
    self$w_q <- nn_linear(d_model, d_model, bias = FALSE)
    self$w_k <- nn_linear(d_model, d_model, bias = FALSE)
    self$w_v <- nn_linear(d_model, d_model, bias = FALSE)

    # 输出投影：拼接后 → d_model
    self$w_o <- nn_linear(d_model, d_model)

    self$dropout <- nn_dropout(dropout)

    # 预计算因果遮罩（不可训练，固定值）
    # 下三角 = 0（允许注意），上三角 = -1e9（禁止注意）
    tril_mask <- torch_tril(torch_ones(c(block_size, block_size)))
    causal_mask <- (1 - tril_mask) * -1e9
    # 注册为 buffer
    self$register_buffer("causal_mask", causal_mask$unsqueeze(1)$unsqueeze(2))
    # 形状: [1, 1, block_size, block_size] 便于广播到 [B, n_heads, T, T]

    # 用于可视化的开关
    self$capture_attn <- FALSE
    self$saved_attn   <- NULL
  },

  forward = function(x) {
    B      <- dim(x)[1]   # batch size
    T_seq  <- dim(x)[2]   # 当前序列长度

    # ---- 线性投影得到 Q, K, V ----
    Q <- self$w_q(x)  # [B, T, d_model]
    K <- self$w_k(x)
    V <- self$w_v(x)

    # ---- 拆分为多头 ----
    # 将 d_model 重塑为 [n_heads, head_dim]
    # [B, T, d_model] → [B, T, n_heads, head_dim] → [B, n_heads, T, head_dim]
    Q <- Q$view(c(B, T_seq, self$n_heads, self$head_dim))$transpose(2, 3)
    K <- K$view(c(B, T_seq, self$n_heads, self$head_dim))$transpose(2, 3)
    V <- V$view(c(B, T_seq, self$n_heads, self$head_dim))$transpose(2, 3)

    # ---- 缩放点积注意力 ----
    current_mask <- self$causal_mask[, , 1:T_seq, 1:T_seq]
    attn_result <- scaled_dot_product_attention(Q, K, V, current_mask)

    # ---- 保存注意力权重（用于可视化） ----
    if (self$capture_attn) {
      self$saved_attn <- attn_result$attn_weights$detach()
    }

    attended <- attn_result$output  # [B, n_heads, T, head_dim]

    # ---- 拼接多头 ----
    # [B, n_heads, T, head_dim] → [B, T, n_heads, head_dim] → [B, T, d_model]
    attended <- attended$transpose(2, 3)$reshape(c(B, T_seq, self$d_model))

    # ---- 输出投影 ----
    output <- self$w_o(attended)
    output
  }
)

# ============================================================
# 7. 前馈网络 (Position-wise Feed-Forward Network)
# ============================================================
# 两层 MLP + GELU 激活，对每个位置独立作用
# d_model → d_ff → d_model （先扩张再压缩）

FeedForward <- nn_module(
  "FeedForward",
  initialize = function(d_model, d_ff, dropout = 0.1) {
    self$linear1 <- nn_linear(d_model, d_ff)    # 扩张
    self$linear2 <- nn_linear(d_ff, d_model)    # 压缩回原维度
    self$dropout <- nn_dropout(dropout)
  },

  forward = function(x) {
    x %>%
      self$linear1() %>%
      nnf_gelu() %>%          # GELU 激活（比 ReLU 更平滑）
      self$linear2() %>%
      self$dropout()
  }
)

# ============================================================
# 8. Transformer Block (Pre-Norm 架构)
# ============================================================
# Pre-Norm: LayerNorm → Sublayer → Residual Add
# 相比 Post-Norm 更稳定，训练收敛更快
#
# 一个 Block = MHA子层 + FFN子层，每个子层= Pre-Norm + Residual

TransformerBlock <- nn_module(
  "TransformerBlock",
  initialize = function(d_model, n_heads, block_size, d_ff, dropout = 0.1) {
    self$ln1 <- nn_layer_norm(d_model)       # MHA 前的 LayerNorm
    self$mha <- MultiHeadSelfAttention(d_model, n_heads, block_size, dropout)
    self$ln2 <- nn_layer_norm(d_model)       # FFN 前的 LayerNorm
    self$ffn <- FeedForward(d_model, d_ff, dropout)
  },

  forward = function(x) {
    # 子层1：自注意力 + 残差连接
    x <- x + self$mha(self$ln1(x))

    # 子层2：前馈网络 + 残差连接
    x <- x + self$ffn(self$ln2(x))

    x
  }
)

# ============================================================
# 9. 迷你 Transformer 语言模型 (Decoder-Only)
# ============================================================
# 整体架构（GPT 风格）：
#   Input IDs → TokenEmbedding + PositionalEncoding
#   → TransformerBlock × n_layers
#   → Final LayerNorm → LM Head (Linear) → Logits

MiniTransformer <- nn_module(
  "MiniTransformer",
  initialize = function(vocab_size, d_model, n_heads, n_layers,
                         block_size, d_ff, dropout = 0.1) {
    self$n_layers <- n_layers  # 保存层数，用于 forward 中迭代

    # 词嵌入：把字符 ID 映射到 d_model 维向量
    self$token_embedding <- nn_embedding(vocab_size, d_model)

    # 位置编码：注入位置信息（"第几个字"）
    self$pos_encoding <- PositionalEncoding(block_size, d_model)

    # 堆叠 Transformer Block（分别命名的子模块）
    for (i in 1:n_layers) {
      self[[paste0("block_", i)]] <- TransformerBlock(
        d_model, n_heads, block_size, d_ff, dropout
      )
    }

    # 最终归一化
    self$ln_final <- nn_layer_norm(d_model)

    # 语言模型头：d_model → vocab_size，预测下一个字符
    self$lm_head <- nn_linear(d_model, vocab_size)

    # 参数统计
    total_params <- 0L
    for (nm in names(self$parameters)) {
      total_params <- total_params + self$parameters[[nm]]$numel()
    }
    cat(sprintf(">>> 模型参数量: %d\n", total_params))
  },

  forward = function(x) {
    # x: [B, T]  — 字符 ID 索引

    # Token Embedding
    x <- self$token_embedding(x)  # [B, T, d_model]

    # + Positional Encoding
    x <- self$pos_encoding(x)     # [B, T, d_model]

    # 通过所有 Transformer Block
    for (i in 1:self$n_layers) {
      x <- self[[paste0("block_", i)]](x)
    }

    # 最终 LayerNorm
    x <- self$ln_final(x)         # [B, T, d_model]

    # LM Head → logits
    logits <- self$lm_head(x)     # [B, T, vocab_size]

    logits
  }
)

# ============================================================
# 10. 实例化模型 & 优化器
# ============================================================
cat("\n========== 构建模型 ==========\n")

model <- MiniTransformer(
  vocab_size = vocab_size_model,
  d_model    = d_model,
  n_heads    = n_heads,
  n_layers   = n_layers,
  block_size = block_size,
  d_ff       = d_ff,
  dropout    = dropout_rate
)

# Adam 优化器
optimizer <- optim_adam(model$parameters, lr = learning_rate)

# 损失函数：交叉熵（内部含 softmax，所以模型输出 logits 即可）
compute_loss <- function(logits, targets) {
  B     <- dim(logits)[1]
  T_seq <- dim(logits)[2]

  # nnf_cross_entropy 需要 [N, C] 的 logits 和 [N] 的 targets
  nnf_cross_entropy(
    logits$view(c(B * T_seq, vocab_size_model)),
    targets$view(c(B * T_seq))
  )
}

# ============================================================
# 11. 评估函数
# ============================================================
estimate_loss <- function() {
  model$eval()

  result <- list()
  for (split in c("train", "val")) {
    losses <- numeric(10)
    for (i in 1:10) {
      batch <- get_batch(split)
      with_no_grad({
        logits <- model(batch$x)
        loss <- compute_loss(logits, batch$y)
      })
      losses[i] <- as.numeric(loss$item())
    }
    result[[split]] <- mean(losses)
  }

  model$train()
  result
}

# ============================================================
# 12. 训练循环
# ============================================================
# ============================================================
# 生成函数（提前定义，供训练前后调用）
# ============================================================
generate <- function(model, prompt, max_new_tokens = 100, temperature = 0.8,
                      top_k = 20L, repeat_penalty = 1.2) {
  model$eval()

  prompt_chars <- strsplit(prompt, "")[[1]]
  indices <- sapply(prompt_chars, function(c) {
    if (c %in% names(char2idx)) char2idx[[c]] else 0L
  })
  idx <- torch_tensor(matrix(indices, nrow = 1), dtype = torch_long())

  cat(sprintf("\nPrompt: \"%s\"\n", prompt))
  cat("Generated: \"")

  for (i in 1:max_new_tokens) {
    idx_len <- dim(idx)[2]
    start_pos <- max(1L, idx_len - block_size + 1L)
    idx_cond <- idx[, start_pos:idx_len]

    with_no_grad({
      logits <- model(idx_cond)
      # 取出最后一帧的 logits（显式索引避免 drop=TRUE 兼容问题）
      logits <- logits[1, dim(logits)[2], ]
    })

    # ---- 重复惩罚 ----
    if (repeat_penalty > 1.0) {
      generated <- as.integer(idx[1, ])
      for (gid in generated) {
        if (gid >= 1) logits[gid] <- logits[gid] / repeat_penalty
      }
    }

    # ---- 屏蔽索引 0 + 温度缩放 ----
    logits[1] <- -Inf
    logits <- logits / temperature

    # ---- Top-K 过滤（用 R 整数索引，兼容 R torch）----
    if (top_k > 0) {
      topk_result <- logits$topk(top_k)
      topk_vals <- topk_result[[1]]            # [K] top-K 值
      topk_idx_r <- as.integer(topk_result[[2]]) # [K] 转为 R 整数向量
      new_logits <- rep(-Inf, length(logits))   # R 向量初始化为 -Inf
      for (j in seq_along(topk_idx_r)) {
        new_logits[topk_idx_r[j]] <- as.numeric(topk_vals[j])
      }
      logits <- torch_tensor(new_logits)
    }

    probs <- nnf_softmax(logits, dim = -1)
    next_idx <- torch_multinomial(probs, num_samples = 1L, replacement = TRUE)

    nid <- as.integer(next_idx)
    if (nid == EOS_IDX) { cat("█"); break }

    idx <- torch_cat(list(idx, next_idx$view(c(1L, 1L))), dim = 2)
    if (nid >= 1 && nid <= vocab_size) cat(unique_chars[nid])
  }
  cat("\"\n")
  model$train()

  invisible(paste(sapply(as.integer(idx[1, ]), function(i) {
    if (i >= 1 && i <= vocab_size) unique_chars[i] else ""
  }), collapse = ""))
}

# ---- 训练前生成（随机权重，展示 baseline）----
cat("\n========== 训练前生成（随机权重）==========\n")
set.seed(42)
generate(model, "春", temperature = 0.9, repeat_penalty = 1.0, max_new_tokens = 60)

cat("\n========== 开始训练 ==========\n")

model$train()
loss_history <- numeric(max_iters)
train_losses <- numeric(max_iters %/% eval_interval + 1)
val_losses   <- numeric(max_iters %/% eval_interval + 1)
eval_step    <- 1

t_start <- Sys.time()

for (iter in 1:max_iters) {

  # 获取一个 batch
  batch <- get_batch("train")

  # 前向传播
  optimizer$zero_grad()
  logits <- model(batch$x)
  loss <- compute_loss(logits, batch$y)

  # 反向传播
  loss$backward()

  # 梯度裁剪（手动实现，防止梯度爆炸）
  total_norm <- 0.0
  for (p in model$parameters) {
    if (!is.null(p$grad)) {
      total_norm <- total_norm + as.numeric(p$grad$norm()$item())^2
    }
  }
  total_norm <- sqrt(total_norm)
  max_norm <- 1.0
  if (total_norm > max_norm) {
    clip_coef <- max_norm / (total_norm + 1e-6)
    for (p in model$parameters) {
      if (!is.null(p$grad)) {
        p$grad$mul_(clip_coef)
      }
    }
  }

  # ---- 学习率调度（Warmup + 余弦衰减）----
  if (iter <= warmup_iters) {
    lr <- learning_rate * iter / warmup_iters          # 线性 warmup
  } else {
    progress <- (iter - warmup_iters) / (max_iters - warmup_iters)
    lr <- lr_min + 0.5 * (learning_rate - lr_min) * (1 + cos(pi * progress))
  }
  for (pg in optimizer$param_groups) {
    pg$lr <- lr
  }

  # 更新参数
  optimizer$step()

  # 记录损失
  loss_history[iter] <- as.numeric(loss$item())

  # 定期评估
  if (iter %% eval_interval == 0 || iter == 1) {
    est <- estimate_loss()
    train_losses[eval_step] <- est$train
    val_losses[eval_step]   <- est$val

    elapsed <- difftime(Sys.time(), t_start, units = "secs")
    cat(sprintf(
      "Iter %4d/%d | Train Loss: %.4f | Val Loss: %.4f | Time: %.1fs\n",
      iter, max_iters, est$train, est$val, as.numeric(elapsed)
    ))
    eval_step <- eval_step + 1
  }
}

t_end <- Sys.time()
cat(sprintf("\n>>> 训练完成！总耗时: %.1f 秒\n",
            as.numeric(difftime(t_end, t_start, units = "secs"))))

# ============================================================
# 13. 损失曲线可视化
# ============================================================
cat("\n========== 损失曲线 ==========\n")

# 平滑函数（移动平均）
smooth <- function(x, window = 20) {
  n <- length(x)
  if (n < window) return(x)
  result <- numeric(n)
  for (i in 1:n) {
    start <- max(1, i - window + 1)
    result[i] <- mean(x[start:i])
  }
  result
}

# 画出两个图：平滑损失曲线 + 原始损失（取子集）
par(mfrow = c(1, 2))

# 左图：训练集平滑曲线
eval_points <- seq(1, max_iters, by = eval_interval)
eval_points <- eval_points[1:(eval_step - 1)]
plot(eval_points, train_losses[1:(eval_step - 1)],
     type = "o", col = "steelblue", pch = 16, cex = 0.6,
     main = "Training & Validation Loss",
     xlab = "Iteration", ylab = "Loss",
     ylim = range(c(train_losses[1:(eval_step-1)], val_losses[1:(eval_step-1)])))
lines(eval_points, val_losses[1:(eval_step - 1)],
      type = "o", col = "tomato", pch = 17, cex = 0.6)
legend("topright", legend = c("Train", "Val"),
       col = c("steelblue", "tomato"), pch = c(16, 17), lty = 1)

# 右图：平滑后的逐步训练损失
smoothed <- smooth(loss_history, window = 50)
plot(smoothed, type = "l", col = "darkgreen", lwd = 1.5,
     main = "Smoothed Training Loss",
     xlab = "Iteration", ylab = "Loss (smoothed)")

par(mfrow = c(1, 1))

# ============================================================
# 14. 训练后文本生成（使用全局已定义的 generate 函数）
# ============================================================
cat("\n========== 训练后生成（对比改进效果）==========\n")

# 用不同 prompt 测试生成
set.seed(42)  # 固定随机种子便于复现
generate(model, "春", temperature = 0.9, repeat_penalty = 1.0, max_new_tokens = 60)
generate(model, "月", temperature = 0.9, repeat_penalty = 1.0, max_new_tokens = 60)
generate(model, "白日", temperature = 0.8, repeat_penalty = 1.0, max_new_tokens = 60)

# ============================================================
# 15. Attention 权重可视化
# ============================================================
cat("\n========== Attention 可视化 ==========\n")

visualize_attention <- function(model, text, layer = 1) {
  model$eval()

  # 编码输入文本
  chars <- strsplit(text, "")[[1]]
  indices <- sapply(chars, function(c) {
    if (c %in% names(char2idx)) char2idx[[c]] else 0L
  })
  x <- torch_tensor(matrix(indices, nrow = 1), dtype = torch_long())

  T_seq <- length(indices)

  # 只取前 block_size 个字符
  if (T_seq > block_size) {
    x <- x[, 1:block_size]
    chars <- chars[1:block_size]
    T_seq <- block_size
  }

  # 打开 attention 捕获（指定层的第 1 个 head）
  block <- model[[paste0("block_", layer)]]
  block$mha$capture_attn <- TRUE

  with_no_grad({
    logits <- model(x)
  })

  # 提取注意力权重（as.array 保持 4D 形状）
  attn_arr <- as.array(block$mha$saved_attn$cpu())
  block$mha$capture_attn <- FALSE

  # attn_arr: [B, n_heads, T, T]
  # 画所有 head 的热力图
  n_h <- dim(attn_arr)[2]
  par(mfrow = c(1, min(n_h, 4)))

  for (h in 1:min(n_h, 4)) {
    head_attn <- attn_arr[1, h, 1:T_seq, 1:T_seq]

    # 只显示前若干字符的标签（避免重叠）
    label_step <- max(1, floor(T_seq / 15))
    labels <- rep("", T_seq)
    labels[seq(1, T_seq, by = label_step)] <- chars[seq(1, T_seq, by = label_step)]

    image(
      t(head_attn[T_seq:1, ]),  # 翻转 y 轴使原点在左上角
      col  = heat.colors(128),
      main = paste0("Layer ", layer, ", Head ", h),
      xlab = "Key Position",
      ylab = "Query Position",
      xaxt = "n", yaxt = "n"
    )

    # 简化的轴标注
    at_pos <- seq(0, 1, length.out = sum(labels != ""))
    axis(1, at = seq(0, 1, length.out = T_seq)[labels != ""],
         labels = labels[labels != ""], las = 2, cex.axis = 0.7)
    axis(2, at = seq(0, 1, length.out = T_seq)[labels != ""],
         labels = labels[labels != ""], las = 2, cex.axis = 0.7)
  }

  par(mfrow = c(1, 1))
  model$train()
}

# 用一句诗做可视化
cat("绘制 Attention 权重热力图...\n")
visualize_attention(model, "春眠不觉晓处处闻啼鸟夜来风雨声", layer = 1)

# ============================================================
# 16. 冒烟测试 — overfit 小样本验证模型正确性
# ============================================================
cat("\n========== 冒烟测试 (Overfit Test) ==========\n")
cat("目标：在一首短诗上过拟合，验证模型是否有足够容量学到数据\n")

overfit_test <- function() {
  # 只用一首诗
  test_text <- "床前明月光，疑是地上霜。举头望明月，低头思故乡。"
  test_chars <- strsplit(test_text, "")[[1]]

  # 构建小词表（1-index）
  test_unique <- sort(unique(test_chars))
  test_vocab <- length(test_unique)
  test_c2i <- setNames(1:test_vocab, test_unique)

  test_encoded <- torch_tensor(as.integer(test_c2i[test_chars]), dtype = torch_long())

  # 微型模型（vocab_size+1 因为索引从 1 开始，0 保留）
  test_model <- MiniTransformer(
    vocab_size = test_vocab + 1L, d_model = 32, n_heads = 2, n_layers = 2,
    block_size = 64, d_ff = 128, dropout = 0.0
  )

  test_opt <- optim_adam(test_model$parameters, lr = 5e-3)

  test_model$train()
  n_iter <- 300

  for (iter in 1:n_iter) {
    # 每条样本: input[1:(n-1)], target[2:n]
    x <- test_encoded[1:(length(test_encoded) - 1)]$unsqueeze(1)
    y <- test_encoded[2:length(test_encoded)]$unsqueeze(1)

    T_len <- min(dim(x)[2], 64L)
    x <- x[, 1:T_len]
    y <- y[, 1:T_len]

    test_opt$zero_grad()
    logits <- test_model(x)
    B <- dim(logits)[1]; T_seq <- dim(logits)[2]
    loss <- nnf_cross_entropy(
      logits$view(c(B * T_seq, test_vocab + 1L)),
      y$view(c(B * T_seq))
    )
    loss$backward()
    test_opt$step()

    if (iter %% 30 == 0 || iter == 1) {
      cat(sprintf("  Overfit iter %3d | Loss: %.4f\n", iter,
                  as.numeric(loss$item())))
    }
  }

  final_loss <- as.numeric(loss$item())
  if (final_loss < 0.1) {
    cat(sprintf(">>> ✓ 测试通过！最终 loss = %.6f（已接近 0）\n", final_loss))
  } else {
    cat(sprintf(">>> ✗ 测试未通过，最终 loss = %.4f（期望 < 0.1）\n", final_loss))
  }
}

overfit_test()

# ============================================================
# 17. 总结输出
# ============================================================
cat("\n========================================\n")
cat("  Transformer 字符级语言模型 — 完成！\n")
cat("========================================\n")
cat(sprintf("  词表大小:    %d\n", vocab_size))
cat(sprintf("  模型维度:    d_model=%d, n_heads=%d, n_layers=%d\n",
            d_model, n_heads, n_layers))
cat(sprintf("  最终训练损失: %.4f\n", tail(loss_history, 1)))
cat(sprintf("  最终验证损失: %.4f\n", tail(val_losses[!is.na(val_losses)], 1)))
cat("\n  核心组件（全部从零手写）：\n")
cat("    - Sinusoidal Positional Encoding\n")
cat("    - Scaled Dot-Product Attention (with Causal Mask)\n")
cat("    - Multi-Head Self-Attention\n")
cat("    - Feed-Forward Network (GELU)\n")
cat("    - Transformer Block (Pre-Norm)\n")
cat("    - Autoregressive Text Generation\n")
cat("========================================\n")
