<div align="center">
 👋 Hi, everyone! 
    <br>
    We are <b>ByteDance Seed team.</b>
</div>

<p align="center">
  You can get to know us better through the following channels👇
  <br>
  <a href="https://team.doubao.com/">
    <img src="https://img.shields.io/badge/Website-%231e37ff?style=for-the-badge&logo=bytedance&logoColor=white"></a>
  <a href="https://github.com/user-attachments/assets/93481cda-a7f3-47f3-b333-fe6b3da86b78">
    <img src="https://img.shields.io/badge/WeChat-07C160?style=for-the-badge&logo=wechat&logoColor=white"></a>
 <a href="https://www.xiaohongshu.com/user/profile/668e7e15000000000303157d?xsec_token=ABl2-aqekpytY6A8TuxjrwnZskU-6BsMRE_ufQQaSAvjc%3D&xsec_source=pc_search">
    <img src="https://img.shields.io/badge/Xiaohongshu-%23FF2442?style=for-the-badge&logo=xiaohongshu&logoColor=white"></a>
  <a href="https://www.zhihu.com/org/dou-bao-da-mo-xing-tuan-dui/">
    <img src="https://img.shields.io/badge/zhihu-%230084FF?style=for-the-badge&logo=zhihu&logoColor=white"></a>
</p>

![seed logo](https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216)

# ByteCheckpoint: A Unified Checkpointing System for Large Foundation Model Development
<p align="center">
  <a href="https://arxiv.org/pdf/2407.20143">
    <img src="https://img.shields.io/badge/Paper-NSDI-red"></a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-Apache-blue"></a>
</p>

ByteCheckpoint is a unified, efficient and production-grade checkpointing system for large foundation model development.

ByteCheckpoint is the open-source implementation of our research paper:
[ByteCheckpoint: A Unified Checkpointing System for Large Foundation Model Development](https://arxiv.org/abs/2407.20143).

ByteCheckpoint is easy to use and efficient with:

✔ **Framework-Agnostic API**: Provides a unified checkpointing entrypoint, i.e., `bytecheckpoint.save` and `bytecheckpoint.load`, to support various parallelism configurations across different frameworks.

✔ **Load-time Checkpoint Resharding**: Enables seamless checkpoint reloading with arbitrary new parallelism configurations, eliminating the need for manual resharding scripts.

✔ **Optimized I/O Performance**: Integrates advanced techniques such as asynchronous and parallel I/O, D2H tensor copying with pinned memory, load-balanced checkpointing, decomposed tensor representation.

✔ **Comprehensive Toolset**: Provides utilities for checkpoint merging/conversion/modification and metadata/tensor file inspection. Enables flexible checkpoint transfer and management.

## 📰 News
[2025/04] We officially released ByteCheckpoint! 🔥 

[2024/12] ByteCheckpoint is accepted to NSDI 2025.

## 🚀 Getting started

### Installation

Install ByteCheckpoint from source.
```
git clone https://github.com/ByteDance-Seed/ByteCheckpoint.git
cd ByteCheckpoint
pip install -e .
```

Install ByteCheckpoint from PyPI.
```
pip install bytecheckpoint
```

### Basic Usage

We introduce how to use Bytecheckpoint to save, load, and merge checkpoint.

In ByteCheckpoint, a checkpoint consists of three parts (folders):
- `model`: It contains model checkpoint, including one ``.metadata`` checkpoint metadata file and multiple `.distcp` tensor data files.  
- `optimizer`: It contains optimizer checkpoint, including one ``.metadata`` checkpoint metadata file and multiple `.distcp` tensor data files.   
- `extra_state`: It contains user-saved pickable objects, e.g., the dataloader state dictionary and RNG states.  

#### Save and Load Checkpoint

Get model, optimizer, and extra states (RNG states, learning rate scheduler) from training code.
```python
checkpoint_state = {
    "model": model, 
    "optimizer": optimizer, 
    "extra_state": {'torch_rng_state': torch.get_rng_state()}
}
```
Save them with ByteCheckpoint `save` API.
```python
import bytecheckpoint as bcp
bcp.save(ckpt_path, checkpoint_state, framework="fsdp")
```
Load them with ByteCheckpoint `load` API.
The model and optimizer will be loaded in an in-place manner. 
The extra state will be loaded in `checkpoint_state["extra_state"]`. 
```python
bcp.load(ckpt_path, checkpoint_state, framework="fsdp")
torch.set_rng_state(checkpoint_state["extra_state"]['torch_rng_state'])
```

#### Training Code Example (FSDP)

A simple single-machine FSDP training demo with ByteCheckpoint is on [demo/fsdp_save_reshard.py](demo/fsdp_save_reshard.py)

Start training and save checkpoint at each step:

```bash
# Train on 8 GPUs
torchrun --master_addr=localhost --master_port=6000 --nproc_per_node=8 --nnodes=1 demo/fsdp_save_reshard.py --mode normal
```
Load checkpoint and resume training:
```bash
# Load on 4 GPUs
torchrun --master_addr=localhost --master_port=6000 --nproc_per_node=4 --nnodes=1 demo/fsdp_save_reshard.py --mode resume
```

For multi-machine training, we recommend operating checkpoint in a shared file system that supports POSIX semantics, such as [NFS](https://documentation.ubuntu.com/server/how-to/networking/install-nfs/index.html).

#### Merge Model checkpoint

To merge model checkpoint, you can use `scripts/merge_bcp.py`

Merge saved checkpoint in the demo training code with safetensors format:

```
python3 scripts/merge_bcp.py --framework fsdp \
--ckpt_path tmp_checkpoint_dir_fsdp/global_step_0 \
--output_path merged_ckpt_fsdp \
--safetensors_format \
--model_only
```

## 🔧 Advanced Usage Guide

### API Arguments
- Enable `fast_saving` and `fast_loading` to use asynchronous and parallel I/O techniques.
- Enable `save_decomposed_model_optimizer` and `load_decomposed_model_optimizer` for FSDP (`use_orig_params=True` is required) to obtain model/optimizer state dict without additional communication and GPU-CPU synchronization.
- Pass the `role` keyword (e.g., actor, critic) to support checkpointing in multi-role training scenarios, such as PPO training.
- Enable `strict` in `load` API to check whether the fqns in a given state_dict are strictly the same as those recorded in the .metadata file. 

### Configuration
- Enable `BYTECHECKPOINT_ENABLE_TREE_TOPO` to improve the stability of large-scale planning for model/optimizer planning.
- Enable `BYTECHECKPOINT_ENABLE_PINNED_MEM_D2H` to use the pinned CPU memory pool to accelerate D2H tensor copying.
- Adjust `BYTECHECKPOINT_STORE_WORKER_COUNT` and `BYTECHECKPOINT_LOAD_WORKER_COUNT` to tune the I/O performance.

Please refer to [config.py](bytecheckpoint/config.py) for more details.

## 🤝 Contribution Guide

Community contributions are welcome. Please checkout [Contribution Guidance](CONTRIBUTING.md).

### Code Formatting

We use `ruff` to enforce strict code formatting when reviewing PRs. To reformat your code locally, make sure you have installed the latest version of `ruff`.
```
pip install ruff
```

Then you can format code with:
```
bash format_code.sh
```

### Testing
Run local tests with:
```
bash test.sh
```

## 📄 License
This project is licensed under Apache License 2.0. See the `LICENSE` file for details.

## 😊 Citation and Acknowledgement
If you find this project helpful, please give us a star ⭐ and cite our [paper](https://arxiv.org/pdf/2407.20143):

```bibtex
@article{wan2024bytecheckpoint,
  title={ByteCheckpoint: A Unified Checkpointing System for Large Foundation Model Development},
  author={Borui, Wan and Mingji, Han and Yiyao, Sheng and Yanghua, Peng and Haibin, Lin and Mofan, Zhang and Zhichao, Lai and Menghan, Yu and Junda, Zhang and Zuquan, Song and Xin, Liu and Chuan, Wu},
  journal={arXiv preprint arXiv:2407.20143},
  year={2024}
}
```

ByteCheckpoint is inspired by the design of [PyTorch Distributed Checkpoint (DCP)](https://pytorch.org/docs/stable/distributed.checkpoint.html).

## 🌱 About [ByteDance Seed Team](https://team.doubao.com/)

Founded in 2023, ByteDance Seed Team is dedicated to crafting the industry's most advanced AI foundation models. The team aspires to become a world-class research team and make significant contributions to the advancement of science and society.
