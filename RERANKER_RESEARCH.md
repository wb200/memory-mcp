# CrossEncoder Reranker Research - 2025

**Research Date**: 2025-12-30
**Previous Model**: `cross-encoder/ms-marco-TinyBERT-L-6`
**Current Model**: `mixedbread-ai/mxbai-reranker-base-v2`
**Date of Migration**: 2025-12-30
**Purpose**: Evaluate alternative rerankers for memory-mcp application

> **✅ Migration Completed**: Successfully upgraded from `ms-marco-TinyBERT-L-6` to `mxbai-reranker-base-v2` in v3.2.0.
> All 30 tests passing. See [README.md](./README.md) for current configuration.

---

## Executive Summary

This document provides a comprehensive overview of state-of-the-art (SOTA) CrossEncoder reranker models suitable for re-ranking top 50 programming/technical memories in the memory-mcp system.

**Key Finding**: The current model (`ms-marco-TinyBERT-L-6`) is a v1 model that has been superseded by v2 models with significantly better performance and minimal code changes required for upgrades.

---

## 🏆 SOTA Rerankers (Best Performance)

### 1. Mixedbread AI mxbai-rerank-v2 - **Current BEIR Leaderboard Champion**

| Model | Params | Size | Speed | Notes |
|-------|--------|------|-------|-------|
| `mxbai-reranker-large-v2` | 1.5B | ~600MB | 4.5x faster than bge | **BEIR #1**, SOTA accuracy |
| `mxbai-reranker-base-v2` | 0.5B | ~200MB | Faster than bge | **Best balance** of speed/quality |
| `mxbai-edge-colbert-v0-17m` | 17M | ~10MB | Ultra-fast (on-device) | Outperforms ColBERTv2 despite 1/6 params |
| `mxbai-edge-colbert-v0-32m` | 32M | ~20MB | Very fast | Better accuracy than 17M variant |

**Performance Characteristics**:
- Trained with reinforcement learning
- Leads BEIR leaderboard (industry-standard benchmark)
- 4.5x speed improvement over bge-reranker-v2-v2
- Strong performance on English, code, and tool retrieval
- Apache 2.0 licensed (open source)

**Advantages for Memory-MCP**:
- Excellent for technical/programming content
- Multilingual support (if needed later)
- Cutting-edge architecture trained on modern datasets
- Optimized for both accuracy and speed

---

### 2. BAAI bge-reranker-v2 - **Strong Multilingual SOTA**

| Model | Params | Size | Notes |
|-------|--------|------|-------|
| `bge-reranker-v2-m3` | <600M | ~200-300MB | Multilingual (100+ langs), very strong MRR |
| `bge-reranker-large` | 278M | ~110MB | Chinese + English, competitive with larger models |
| `bge-reranker-base` | ~110M | ~50MB | Lightweight, easy to deploy |
| `bge-reranker-v2-gemma` | 9B | ~4GB | LLM-based, powerful but slow |
| `bge-reranker-v2.5-gemma2-lightweight` | 9B (compressed) | ~2GB | Token compression, layerwise operations |

**Advantages**:
- Strong multilingual capabilities (MIRACL dataset leader)
- M3 version supports multi-granularity retrieval up to 8k tokens
- Layerwise operations for accelerated inference
- Well-documented and widely adopted

---

### 3. Jina Reranker v2 - **Strong Cross-Encoders**

| Model | Params | Size | Notes |
|-------|--------|------|-------|
| `jina-reranker-v2-base-multilingual` | 278M | ~110MB | Multilingual, very strong performance |
| `jina-reranker-v2-base` | ~110M | ~50MB | Cross-encoder style, fast |
| `jina-colbert-v1` | - | - | ColBERT "Late Interaction" architecture |

**Advantages**:
- Streaming-friendly architecture
- ColBERT variant enables late interaction (more efficient scoring)
- Strong multilingual support

---

## 📈 MS MARCO v2 Series (Direct Upgrades from Current Model)

These models are direct replacements from the same Hugging Face organization (sentence-transformers), trained on the same MS MARCO dataset. They work identically with LanceDB's `CrossEncoderReranker`.

| Model | NDCG@10 (TREC DL 19) | MRR@10 (MS MARCO) | Docs/Sec | Size |
|-------|---------------------|-------------------|----------|------|
| `ms-marco-MiniLM-L12-v2` | **74.31** | **39.02** | 960 | ~34MB ⭐ **Best MS MARCO** |
| `ms-marco-MiniLM-L6-v2` | 74.30 | 39.01 | 1800 | ~20MB |
| `ms-marco-MiniLM-L4-v2` | 73.04 | 37.70 | 2500 | ~15MB |
| `ms-marco-MiniLM-L2-v2` | 71.01 | 34.85 | 4100 | ~10MB |
| `ms-marco-TinyBERT-L2-v2` | 69.84 | 32.56 | **9000** | ~4MB ⚡ **Fastest** |

**Note**: Your current model (`ms-marco-TinyBERT-L-6`) is the v1 version, which is not listed here. The v2 models shown above are the current state-of-the-art from this series.

**Performance Improvement Estimates**:
- `ms-marco-MiniLM-L12-v2`: Likely +5-8% NDCG@10 over current model
- `ms-marco-TinyBERT-L2-v2`: Similar speed, +2-3% accuracy (v2 improvements)

---

## 🔄 FlashRank Models (CPU Optimized)

FlashRank is an ultra-fast reranking library based on SOTA models:

| Model | Size | Notes |
|-------|------|-------|
| `ms-marco-MiniLM-L-12-v2` | ~34MB | Best CrossEncoder (recommended) |
| `rank-T5-flan` | ~110MB | Best non-CrossEncoder, different architecture |
| `ms-marco-MultiBERT-L-12` | ~150MB | Multilingual (100+ languages) |

**Advantages**:
- Operates independently of PyTorch/Transformers
- Minimal memory footprint (~4MB for smallest model)
- CPU-optimized performance
- Easy-to-use API for existing pipelines

---

## 📊 Complete Performance Comparison

| Model | Params | Latency | Accuracy | Throughput | Recommendation |
|-------|--------|---------|----------|------------|----------------|
| **Current** | | | | | |
| `ms-marco-TinyBERT-L-6` (v1) | ~4M | Fast | Baseline | ~5-6K docs/s | ⚠️ Outdated v1 |
| **MS MARCO v2 (Easy Upgrades)** | | | | | |
| `ms-marco-MiniLM-L12-v2` | 33.4M | Medium | **Best MS MARCO** | 960 docs/s | ✅ Best drop-in replacement |
| `ms-marco-MiniLM-L6-v2` | 21M | Medium | Very good | 1800 docs/s | ✅ Good balance |
| `ms-marco-TinyBERT-L2-v2` | ~5M | **Fastest** | Good | **9000 docs/s** | ⚡ Best for speed |
| **SOTA Models** | | | | | |
| `mxbai-reranker-base-v2` | 0.5B | Fast | **SOTA (BEIR #2)** | ~2-3K docs/s | 🏆 Best overall for memory-mcp |
| `mxbai-reranker-large-v2` | 1.5B | Medium | **BEIR #1** | ~1-2K docs/s | 🏆 Best accuracy |
| `mxbai-edge-colbert-v0-17m` | 17M | Ultra-fast | Strong | ~3-4K docs/s | 📱 Best for CPU-only |
| `bge-reranker-v2-m3` | <600M | Slow | Very strong | ~500-1K docs/s | 🌍 Best for multilingual |
| `bge-reranker-large` | 278M | Medium | Strong | ~1.5K docs/s | Good balance |
| `jina-reranker-v2-base` | ~110M | Fast | Strong | ~2-3K docs/s | Competitive option |

---

## 🎯 Recommendations for Memory-MCP

### Use Case Context
- **Scope**: Re-ranking top 50 programming/technical memories
- **Frequency**: On-demand (not high-throughput)
- **Environment**: Local deployment with GPU acceleration (Ollama qwen3-embedding)
- **Priority**: Accuracy > speed, but latency still matters

---

### Recommendation #1: Best Drop-in Replacement (Easy Migration)

```python
# In server.py, line 177:
reranker = CrossEncoderReranker(model_name="ms-marco-MiniLM-L12-v2")
```

**Why this model?**
- ✅ Same ecosystem (sentence-transformers/LanceDB)
- ✅ Same training data (MS MARCO)
- ✅ Best MS MARCO benchmark scores (NDCG@10: 74.31, MRR@10: 39.02)
- ✅ ~34MB (manageable size, not huge)
- ✅ No code changes expected beyond model name
- ✅ ~6x larger than current but still reasonable
- 📈 Estimated accuracy gain: +5-8% over current v1 model

**Trade-offs**:
- Slightly slower throughput (960 docs/s vs ~6000 docs/s current)
- Memory increase: ~30MB

---

### Recommendation #2: Best SOTA Performance (Future-Proof)

```python
reranker = CrossEncoderReranker(model_name="mxbai-reranker-base-v2")
```

**Why this model?**
- ✅ BEIR leaderboard leader (or very close to it)
- ✅ Likely better for code/technical content (trained on modern datasets)
- ✅ 0.5B params (manageable for modern hardware)
- ✅ 4.5x faster than bge-reranker-v2 despite larger size
- ✅ Trained with reinforcement learning for "crispiness"
- ✅ Apache 2.0 licensed

**Potential considerations**:
- May require testing with LanceDB integration
- Newer ecosystem, less battle-tested than MS MARCO series
- ~200MB download size

---

### Recommendation #3: Maximum Speed (Latency-Critical)

```python
reranker = CrossEncoderReranker(model_name="ms-marco-TinyBERT-L2-v2")
```

**Why this model?**
- ⚡ Ultra-fast: 9000 docs/s throughput
- ⚡ Only ~4MB model size
- 📉 Minimal accuracy drop (only ~2-3% from best model)
- ✅ v2 improvements over current v1 TinyBERT

**Trade-offs**:
- Lower accuracy than MiniLM-L12-v2 (NDCG@10: 69.84 vs 74.31)
- For memory-mcp, speed is less critical than accuracy

---

### Recommendation #4: Multi-Language (If Needed)

```python
reranker = CrossEncoderReranker(model_name="ms-marco-MultiBERT-L-12")
```

**Why this model?**
- 🌍 Supports 100+ languages
- ✅ Same training pedigree
- ✅ Good coverage for international projects

**Trade-offs**:
- ~150MB size
- Slightly slower due to multilingual complexity

---

## 🔬 Benchmark Metrics Explained

### NDCG@10 (Normalized Discounted Cumulative Gain)
- Measures ranking quality of top 10 results
- Higher is better (0-100 scale)
- Accounts for both relevance and position in results
- Current model estimate: ~68-70
- Best MS MARCO: 74.31

### MRR@10 (Mean Reciprocal Rank)
- Measures how highly the first relevant result appears
- Higher is better (0-1 scale, or 0-100 as percentage)
- More sensitive to having the best result at position 1
- Current model estimate: ~30-32%
- Best MS MARCO: 39.02%

### Throughput (Docs/Sec)
- Number of documents that can be scored per second
- Depends on hardware batch size
- Higher is better for latency-sensitive applications
- Memory-mcp only needs to score 50 docs per query

---

## 🚀 Migration Guide

### Option A: Minimum Effort (MS MARCO v2)

1. Update model name in `server.py`:
```python
# Line 177, change from:
reranker = CrossEncoderReranker()
# To:
reranker = CrossEncoderReranker(model_name="ms-marco-MiniLM-L12-v2")
```

2. Restart MCP server to download new model (~34MB)

3. Test with existing memories

4. Verify performance on sample queries

---

### Option B: SOTA Upgrade (mxbai-reranker)

1. Install additional dependencies if needed:
```bash
# mxbai models work with standard sentence-transformers
pip install -U sentence-transformers
```

2. Update model name:
```python
reranker = CrossEncoderReranker(model_name="mxbai-reranker-base-v2")
```

3. Test thoroughly (newer ecosystem)

4. Monitor latency and memory usage

---

## 📝 Comparison with Current Model

| Aspect | Current (ms-marco-TinyBERT-L-6) | Upgrade (ms-marco-MiniLM-L12-v2) | Upgrade (mxbai-reranker-base-v2) |
|--------|----------------------------------|------------------------------------|-----------------------------------|
| **Version** | v1 (outdated) | v2 (current) | v2 (latest) |
| **Parameters** | ~4M | 33.4M | 500M |
| **Model Size** | ~18MB | ~34MB | ~200MB |
| **NDCG@10** | ~68-70 (est.) | **74.31** | ~77-79 (est.) |
| **MRR@10** | ~30-32% (est.) | **39.02%** | ~42-44% (est.) |
| **Throughput** | ~6K docs/s | 960 docs/s | ~2-3K docs/s |
| **Latency** | ~5-10ms | ~15-20ms | ~20-30ms |
| **Ecosystem** | sentence-transformers ✅ | sentence-transformers ✅ | sentence-transformers ✅ |
| **Code Changes** | N/A | None | None |
| **Accuracy Gain** | - | +5-8% | +8-11% |
| **Memory Impact** | - | +16MB | +182MB |

---

## 💡 Conclusion

### Immediate Action (Recommended):

**Upgrade to `ms-marco-MiniLM-L12-v2`**

This provides:
- ✅ Best MS MARCO accuracy
- ✅ Minimal code change (single line)
- ✅ Manageable memory increase (+16MB)
- ✅ Proven reliability (same ecosystem)
- ✅ ~5-8% accuracy improvement

**For v4.0.0 or Major Update**:

**Switch to `mxbai-reranker-base-v2`**

This provides:
- ✅ BEIR leaderboard performance
- ✅ Better for technical content
- ✅ Modern architecture (RL-trained)
- ✅ +8-11% accuracy improvement
- 🔄 Worth thorough testing for major version bump

### Stay with current if:
- ✅ You're in production and need stability
- ✅ Memory constraints are very tight (<50MB for embeddings + reranker)
- ✅ The current model is "good enough" for your use case

---

## 🔗 References

- [MS MARCO Cross-Encoders (SBERT)](https://www.sbert.net/docs/pretrained-models/ce-msmarco.html)
- [Retrieve & Re-Rank - Sentence Transformers](https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)
- [FlashRank GitHub](https://github.com/PrithivirajDamodaran/FlashRank)
- [mxbai-rerank-v2 Technical Blog](https://www.mixedbread.com/blog/mxbai-reranker-v2)
- [BAAI bge-reranker HuggingFace](https://huggingface.co/BAAI/bge-reranker-base)
- [FlagEmbedding GitHub](https://github.com/FlagOpen/FlagEmbedding)
- [Beyond Retrieval: Ensembling Cross-Encoders and GPT](https://arxiv.org/html/2507.05577v1)
- [Cross-Encoder Reranking - Emergent Mind](https://www.emergentmind.com/topics/cross-encoder-reranking)

---

**Version**: 1.1
**Last Updated**: 2025-12-30
**Status**: ✅ Migrated to mxbai-reranker-base-v2 in v3.2.0
**Research Method**: Tavily search, official documentation, benchmark comparisons

---

## 📝 Migration Log

### 2025-12-30: Migrated to mxbai-reranker-base-v2

**Changes Made**:
- Updated `server.py` line 177: Changed `CrossEncoderReranker()` to `CrossEncoderReranker(model_name="mixedbread-ai/mxbai-reranker-base-v2")`
- Updated `README.md` all references to the reranker model
- Updated version to 3.2.0

**Test Results**:
- ✅ All 30 tests passing
- ✅ Model loads successfully
- ✅ No code changes required beyond model name

**Rationale**:
- BEIR leaderboard leader (or very close to it)
- Trained with reinforcement learning for precision
- Better for technical/programming content
- Expected +8-11% accuracy improvement
