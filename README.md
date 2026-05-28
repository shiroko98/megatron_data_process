# megatron_data_process

一个用于将 `.jsonl` 语料批量转换为 Megatron 风格 `.bin/.idx` 数据集的脚本仓库。

当前仓库的主要目标是：

- 读取单个 `.jsonl` 文件，或递归读取目录下的多个 `.jsonl` 文件
- 从每一行 JSON 中抽取指定字段，例如 `text`
- 使用 tokenizer 将文本编码成 token id
- 输出 Megatron / IndexedDataset 风格的 `.bin` 和 `.idx`
- 支持按文件分片、多进程处理、可选分句

这个仓库当前最主要、也最稳定的使用方式是：

- `RWKVTokenizer` + RWKV 词表文件
- `HFTokenizer` + Hugging Face tokenizer 目录

下面的命令示例默认你已经在仓库根目录下执行。
多行示例使用的是 Windows `cmd` 的 `^` 续行写法；如果你在 PowerShell 里执行，可以改成单行，或者把 `^` 换成 PowerShell 的反引号续行。


## 仓库现状

当前代码里，CLI 参数还保留了一些历史上来自 Megatron 的名字，但这个仓库实际长期维护、测试覆盖过的核心路径主要是：

- `RWKVTokenizer`
- `HFTokenizer`
- 输入格式只支持 `.jsonl`

也就是说，如果你现在准备稳定使用这个仓库，建议优先按这两种 tokenizer 方式来用。


## 输入格式

输入必须是 `.jsonl`。

每一行都应该是一个 JSON 对象，例如：

```json
{"text": "hello world"}
{"text": "another sample"}
```

如果你想从多个字段里取文本，也可以这样：

```json
{"title": "标题", "content": "正文"}
{"title": "第二篇", "content": "更多正文"}
```

然后在命令行里通过 `--json-keys title content` 指定需要处理的字段。

当前仓库只会扫描 `.jsonl` 文件，不会自动读取 `.jsonl.gz`。


## 输出格式

输出是 Megatron / IndexedDataset 风格的数据集：

- `xxx_text_document.bin`
- `xxx_text_document.idx`

如果启用了分句：

- `xxx_text_sentence.bin`
- `xxx_text_sentence.idx`

如果一次输入多个文件且 `partitions=1`，每个源文件会各自产出一份结果，名字会基于源文件名自动拼接。


## 处理流程

高层流程如下：

1. 扫描输入路径，收集所有 `.jsonl`
2. 如果 `partitions > 1`，先把输入重新分桶成若干临时分片
3. 如果开启 `--split-sentences`，先生成分句后的中间 `.jsonl`
4. 对每个待处理文件做 tokenizer 编码
5. 将编码结果写入 `.bin/.idx`
6. 如果启用了分片合并，再把分片结果合并成最终输出
7. 成功后尝试删除空的 intermediate 目录

中间文件会统一放到：

- `输出前缀_intermediate`

例如：

- 输出前缀是 `D:\codes\megatron_data_process\out`
- 中间目录就是 `D:\codes\megatron_data_process\out_intermediate`

这样做是为了避免中间 `.jsonl` 污染原始语料目录。


## 主要参数说明

### 输入输出

- `--input`
  - 输入文件或输入目录
- `--output-prefix`
  - 输出前缀，不带 `.bin/.idx` 后缀
- `--json-keys`
  - 每条 JSON 里要抽取的字段名，默认是 `text`

### tokenizer

- `--tokenizer-type`
  - 当前建议只使用 `RWKVTokenizer` 或 `HFTokenizer`
- `--vocab-file`
  - 对 `RWKVTokenizer` 来说，这是必须的 RWKV 词表文件路径
  - 对 `HFTokenizer` 来说，这是兼容旧写法的回退路径
- `--tokenizer-model`
  - 对 `HFTokenizer` 来说，这是更推荐的 tokenizer / model 目录路径
- `--append-eod`
  - 在每个样本末尾追加 `<eod>`

### 并行和分片

- `--workers`
  - tokenizer worker 总预算
  - 它不是“每个文件固定开多少 worker”，而是整个运行期可分配的总并发预算
- `--max-processes`
  - 同时并发处理多少个文件任务
- `--partitions`
  - 先把输入逻辑上拆成多少份再处理
- `--merge-partitions`
  - 处理完后是否把多个分片结果合并成一个最终数据集
- `--keep-sequential-samples`
  - 仅在 `partitions > 1` 时有意义
  - 保证样本按顺序整块进入不同 partition，而不是轮转分配

### 分句

- `--split-sentences`
  - 使用 NLTK 分句后再编码
- `--keep-newlines`
  - 尽量保留换行对分句的影响
- `--lang`
  - NLTK 使用的语言，默认 `english`


## tokenizer 参数语义

这是这个仓库里最容易混淆的地方。

### 1. RWKVTokenizer

RWKV 使用的是“词表文件”。

所以参数语义非常直接：

- `--vocab-file` = RWKV vocab 文件路径

例如当前仓库里就有两个 RWKV 词表文件：

- [rwkv_vocab_v20240530.txt](rwkv_vocab_v20240530.txt)
- [rwkv_vocab_v20250609.txt](rwkv_vocab_v20250609.txt)

其中：

- `rwkv_vocab_v20240530.txt` 是当前测试覆盖过、可直接拿来跑这套脚本的默认示例词表
- `rwkv_vocab_v20250609.txt` 是候选新词表，但在当前环境的 `pyrwkv_tokenizer` 后端下仍需要额外兼容性确认

推荐用法示例：

```bash
python preprocess_data.py ^
  --input data\train-00000-of-00048-ab2b35705f029d94.parquet.jsonl ^
  --output-prefix data\minipile ^
  --tokenizer-type RWKVTokenizer ^
  --vocab-file rwkv_vocab_v20240530.txt ^
  --workers 6 ^
  --max-processes 2 ^
  --append-eod
```

### 2. HFTokenizer

Hugging Face tokenizer 通常不是只靠一个 vocab 文件，而是依赖一个目录中的多个文件，例如：

- `tokenizer.json`
- `tokenizer_config.json`
- `vocab.json`
- `merges.txt`
- `tokenizer.model`

所以对 HFTokenizer，更合理的参数语义是：

- `--tokenizer-model` = Hugging Face tokenizer 目录

例如当前仓库里的这些目录都符合 HF tokenizer 目录的风格：

- [Qwen1.5-14B-Chat](Qwen1.5-14B-Chat/)
- [chatglm3-6b-128k](chatglm3-6b-128k/)
- [llama-3-8b-Instruct-chinese](llama-3-8b-Instruct-chinese/)
- [yi](yi/)

推荐写法：

```bash
python preprocess_data.py ^
  --input data\train-00000-of-00048-ab2b35705f029d94.parquet.jsonl ^
  --output-prefix data\qwen_out ^
  --tokenizer-type HFTokenizer ^
  --tokenizer-model Qwen1.5-14B-Chat ^
  --workers 6 ^
  --max-processes 2
```

ChatGLM 示例：

```bash
python preprocess_data.py ^
  --input data\train-00000-of-00048-ab2b35705f029d94.parquet.jsonl ^
  --output-prefix data\chatglm_out ^
  --tokenizer-type HFTokenizer ^
  --tokenizer-model chatglm3-6b-128k ^
  --workers 6 ^
  --max-processes 2
```

Llama 示例：

```bash
python preprocess_data.py ^
  --input data\train-00000-of-00048-ab2b35705f029d94.parquet.jsonl ^
  --output-prefix data\llama_out ^
  --tokenizer-type HFTokenizer ^
  --tokenizer-model llama-3-8b-Instruct-chinese ^
  --workers 6 ^
  --max-processes 2
```

Yi 示例：

```bash
python preprocess_data.py ^
  --input data\train-00000-of-00048-ab2b35705f029d94.parquet.jsonl ^
  --output-prefix data\yi_out ^
  --tokenizer-type HFTokenizer ^
  --tokenizer-model yi ^
  --workers 6 ^
  --max-processes 2
```

### 3. 和旧用法的区别

历史上，这个仓库的 `HFTokenizer` 实际上把：

- `--vocab-file`

当成了 Hugging Face tokenizer 目录来用。

这会导致参数名和真实语义不一致。

现在代码已经调整为：

- `HFTokenizer` 优先使用 `--tokenizer-model`
- 如果没有传 `--tokenizer-model`，仍然兼容回退到 `--vocab-file`

所以：

- 旧命令大多数还能继续跑
- 但新写法建议改成 `--tokenizer-model`

例如旧写法：

```bash
python preprocess_data.py ^
  --input data\train-00000-of-00048-ab2b35705f029d94.parquet.jsonl ^
  --output-prefix data\qwen_out ^
  --tokenizer-type HFTokenizer ^
  --vocab-file Qwen1.5-14B-Chat ^
  --workers 6
```

现在仍然兼容，但不再是推荐写法。


## 使用示例

### 1. 最常见：RWKV 处理单个文件

```bash
python preprocess_data.py ^
  --input data\train-00000-of-00048-ab2b35705f029d94.parquet.jsonl ^
  --output-prefix data\rwkv_train ^
  --tokenizer-type RWKVTokenizer ^
  --vocab-file rwkv_vocab_v20240530.txt ^
  --json-keys text ^
  --workers 6 ^
  --max-processes 2 ^
  --append-eod
```

输出大致会是：

- `rwkv_train_text_document.bin`
- `rwkv_train_text_document.idx`

### 2. 目录输入，递归处理多个 `.jsonl`

```bash
python preprocess_data.py ^
  --input data ^
  --output-prefix data\merged ^
  --tokenizer-type RWKVTokenizer ^
  --vocab-file rwkv_vocab_v20240530.txt ^
  --workers 6 ^
  --max-processes 2
```

注意：

- 当前目录扫描会递归读取子目录中的 `.jsonl`
- 只会扫描 `.jsonl`
- 顺序是稳定排序过的

### 3. 单个文件，多分片处理

```bash
python preprocess_data.py ^
  --input data\train-00000-of-00048-ab2b35705f029d94.parquet.jsonl ^
  --output-prefix data\partitioned ^
  --tokenizer-type RWKVTokenizer ^
  --vocab-file rwkv_vocab_v20240530.txt ^
  --workers 8 ^
  --max-processes 2 ^
  --partitions 4 ^
  --merge-partitions ^
  --append-eod
```

这时流程会先生成 4 个逻辑 partition，再分别处理，最后合并。

### 4. 保持样本顺序地分片

```bash
python preprocess_data.py ^
  --input data\train-00000-of-00048-ab2b35705f029d94.parquet.jsonl ^
  --output-prefix data\partitioned_seq ^
  --tokenizer-type RWKVTokenizer ^
  --vocab-file rwkv_vocab_v20240530.txt ^
  --workers 8 ^
  --max-processes 2 ^
  --partitions 4 ^
  --merge-partitions ^
  --keep-sequential-samples
```

不开 `--keep-sequential-samples` 时，样本默认按轮转方式分配到不同 partition。

### 5. 使用 HF tokenizer

```bash
python preprocess_data.py ^
  --input data\train-00000-of-00048-ab2b35705f029d94.parquet.jsonl ^
  --output-prefix data\hf_qwen ^
  --tokenizer-type HFTokenizer ^
  --tokenizer-model Qwen1.5-14B-Chat ^
  --workers 6 ^
  --max-processes 2
```


## 关于 `workers` 和 `max_processes`

这是另一个容易误解的点。

当前实现中：

- `workers` 表示总 tokenizer worker 预算
- `max_processes` 表示同时最多跑几个“文件任务”

程序会根据任务数动态分配：

- 实际并发文件数 = `min(max_processes, task_count, workers)`
- 每个文件任务分到的 worker 数 = `workers // 实际并发文件数`

举例：

- `workers=8`
- `max_processes=4`
- 当前有 2 个文件任务

那么最终会变成：

- 并发 2 个文件任务
- 每个文件任务分到 4 个 tokenizer worker

这样可以避免“多文件输入时总 worker 数爆炸”。


## 关于分句

开启 `--split-sentences` 后，流程会先把文本切分成句子，再把句子列表交给 tokenizer。

适用场景：

- 你希望 `.idx/.bin` 中按 sentence level 组织
- 你后续训练或分析逻辑依赖句边界

注意：

- 这个路径依赖 `nltk`
- 如果没有 `nltk`，只要不启用 `--split-sentences`，当前仓库仍然可以正常使用


## 中间文件和清理策略

中间文件放在独立目录：

- `输出前缀_intermediate`

成功运行结束后，程序会：

- 递归尝试删除空目录
- 只删除空目录
- 如果目录里还有残留文件，就保留现场，不强删

这意味着：

- 正常跑完后，如果没有残留，中间目录会消失
- 如果中途失败或有未清理文件，中间目录会保留，便于排查


## 错误处理

当前版本对多进程子任务做了错误上报增强。

如果某个子进程失败，父进程现在会抛出包含以下信息的错误：

- 失败的任务类型
- 对应输入文件
- 异常类型
- 异常信息
- 子进程 traceback

日志文件默认写到仓库目录下：

- [error.log](error.log)


## 如何检查输出文件

仓库里有一个辅助脚本：

- [check_binidx.py](check_binidx.py)

不过它当前更像个人调试脚本，里面还写了固定路径示例，不适合作为稳定 CLI 工具。

如果只是验证是否生成成功，最简单的方法是直接检查：

- `.bin` 是否存在
- `.idx` 是否存在

在测试里也使用了 `indexed_dataset.IndexedDataset(...)` 直接读取生成结果，验证样本数和内容。


## 当前目录下的可用示例资源

### 测试语料

- [data](data/)
- [tests/fixtures/real_data_excerpt.jsonl](tests/fixtures/real_data_excerpt.jsonl)

### RWKV 词表

- [rwkv_vocab_v20240530.txt](rwkv_vocab_v20240530.txt)
- [rwkv_vocab_v20250609.txt](rwkv_vocab_v20250609.txt)

### Hugging Face tokenizer 目录

- [Qwen1.5-14B-Chat](Qwen1.5-14B-Chat/)
- [chatglm3-6b-128k](chatglm3-6b-128k/)
- [llama-3-8b-Instruct-chinese](llama-3-8b-Instruct-chinese/)
- [yi](yi/)


## 一个推荐的实际命令

如果你当前主要就是 RWKV 预训练数据处理，建议直接从这个模板出发：

```bash
python preprocess_data.py ^
  --input data ^
  --output-prefix data\rwkv_dataset ^
  --tokenizer-type RWKVTokenizer ^
  --vocab-file rwkv_vocab_v20240530.txt ^
  --json-keys text ^
  --workers 8 ^
  --max-processes 2 ^
  --append-eod
```

如果是先做小规模验证，可以把 `--input` 换成单个 `.jsonl` 文件，先确认输出和速度都符合预期。


## 已知说明

- 当前仓库只支持 `.jsonl`
- `HFTokenizer` 现在推荐使用 `--tokenizer-model`
- `HFTokenizer` 仍兼容旧的 `--vocab-file` 写法
- `rwkv_vocab_v20250609.txt` 当前已纳入仓库，但在现有 `pyrwkv_tokenizer` 后端上还没有通过直接加载验证
- CLI 参数里虽然还保留了部分历史 tokenizer 名称，但当前代码并没有完整实现这些分支
- `check_binidx.py` 目前是调试脚本，不是正式文档化工具


## 开发和测试

当前仓库已经补充了较完整的单元测试与集成测试。

常用测试命令：

```bash
python -m pytest -q
```

覆盖率：

```bash
python -m pytest --cov=preprocess_data --cov-report=term-missing -q
```

如果你使用当前约定的 conda 环境：

```bash
D:\anaconda\envs\model\python.exe -m pytest -q
```


## 总结

如果只记住三件事：

1. 输入只放 `.jsonl`
2. RWKV 用 `--vocab-file` 指向 `.txt` 词表
3. HF 用 `--tokenizer-model` 指向 tokenizer 目录

按这三条来用，当前仓库就会比较稳定、直观。
