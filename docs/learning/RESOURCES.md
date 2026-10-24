# Drafter foundations resources

## Knowledge

- [GLM mHC implementation](https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/models/glm5_next/modeling_glm5_next.py#L218-L336)
  Exact pre/post gates, residual mixing, and final unweighted stream average.
- [mHC paper, revised version](https://arxiv.org/html/2512.24880v2#S4)
  Why residual mixing is constrained and how iterative normalization works.
- [Native DeepSeek DSpark implementation](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/blob/62af8fffb2f7030cac4de2f0169f5b8d1101b646/inference/model.py#L795-L935)
  Target-stream averaging, the drafter's own four streams, and learned final reduction.
- [GLU Variants Improve Transformer](https://arxiv.org/html/2002.05202v1#S2)
  Gated feed-forward networks and the SwiGLU form of a gated SiLU MLP.
- [GQA paper](https://arxiv.org/html/2305.13245v3#S2.SS2)
  Multiple query heads sharing fewer key/value heads.
- [RoFormer paper](https://arxiv.org/html/2104.09864v5#S3.SS2.SSS2)
  Position-dependent coordinate rotations and their frequency-base parameter.
- [GLM token mappings](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/46a050bec13002ad29ee795c16e61cf4fa3c4a9c/tokenizer.json#L5-L94)
  Exact strings for stop tokens; compare with the pinned generation configuration.
- [Speculators reference losses](https://github.com/vllm-project/speculators/blob/b9c51bcfc134923eae2b6a6f8246ae3500d5c42f/src/speculators/losses/eager.py#L92-L142)
  Teacher-top-token cross-entropy and total-variation distance.
- [Surfing the Activation Wave](https://arxiv.org/abs/2610.12939v1)
  ELSI/ELSI-C layer selection, which is separate from the MLP activation function.

## Gaps

- The user has not yet demonstrated that the four-stream and constraint explanation is clear.
- These sources do not establish the best architecture or feature set for our custom run.
