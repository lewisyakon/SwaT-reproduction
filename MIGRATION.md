# 独立仓库与迁移说明

本仓库是 SwaT 当前复现状态的独立快照，不继承上游 Git 历史或 remote。仓库同时保存：

- SwaT renderer 的原始训练实现与本次 M1/M2/M3 修改；
- `reproduction/current/` 下的统一协议 evaluator、无 AM 队列和两档 AM 队列；
- 显存友好的流式 head pairing 实现（`receiver-chunk=1` 可在 3090 上运行）；
- `variants/SwaT-Code/` 的代码数据特化配置与脚本；
- 固定提示词、正式结果、实验计划与跨方法对比文档。

已完成的原始 renderer、AM Query Transfer 和 SwaT-Code renderer 已剥离 Adam 状态后
通过 Git LFS 纳入 `weights/`；推理张量和 dtype 均保持不变。模型、Hugging Face
数据、可恢复训练状态和逐样本结果仍按用途列在 `migration/external_assets.json`。

克隆后执行：

```bash
git lfs install
git lfs pull
python tools/verify_included_weights.py
```

## 迁移流程

```bash
python tools/verify_external_assets.py --mode current --profile evaluate
python tools/verify_external_assets.py --mode current --profile am
bash tools/create_code_bundle.sh /path/to/export
```

Git bundle 只包含 LFS 指针，不包含权重实体；完整离线迁移还需复制 `weights/`，
或者在有网络时从 GitHub 执行 `git lfs pull`。

按 manifest 复制资产到 `recommended_path` 后：

```bash
python tools/verify_external_assets.py --mode restored --profile evaluate
python tools/verify_external_assets.py --mode restored --profile am
```

本次统一实验脚本保留了旧机器路径作为可审计快照。迁移后先 dry-run 路径改写：

```bash
python tools/relocate_snapshot.py --root reproduction/current \
  --map /home/lewis/Subproblem4UnifiedReproduction="$PWD/reproduction/current" \
  --map /mnt/hd2/lewis/Subproblem4UnifiedReproduction="$PWD" \
  --map /mnt/hd2/lewis/SwaTReproduction="$PWD" \
  --map /home/lewis/DenseLatentCommunication="$PWD/reproduction"

python tools/relocate_snapshot.py --root variants/SwaT-Code \
  --map /mnt/hd2/lewis/SwaT-Code="$PWD/variants/SwaT-Code" \
  --map /mnt/hd2/lewis/SwaTReproduction="$PWD" \
  --map /home/lewis/Subproblem4UnifiedReproduction="$PWD/reproduction/current"
```

确认列表后追加 `--write`。主 evaluator 已支持 `SWAT_ROOT`、
`SUBPROBLEM4_PROJECT_ROOT` 和 `SUBPROBLEM4_ASSET_ROOT`，也可以不改文件而通过环境变量
指定新路径。固定提示词位于 `reproduction/prompts/提示词模板.zip`。

