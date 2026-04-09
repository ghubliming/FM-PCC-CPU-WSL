# Investigation Report: Why the "Visual" Pipeline Runs Without Images

## 1. Background Summary

Earlier today, we verified the data storage requirements established by the **Camera Capture and Storage Policy**. We confirmed that the `images/` directory (including `bp-cam/` and `inhand-cam/`) is entirely missing from the `d3il/environments/dataset/data/avoiding/data/` folder.

Despite missing all image data, our fast CPU smoke test for the "visual" environment (`train_FM_v3_avoiding_visual.py` and `eval_FM_v3_avoiding_visual.py`) ran perfectly. 

This naturally raises the question: **Why can we train and evaluate a visual model if it isn't receiving any pictures?**

---

## 2. Technical Investigation

To answer this, I looked into how the dataset and the neural network architecture are currently wired.

### A. The Dataset Loader is Blind to Images
In `flow_matcher_v3_avoiding_visual/datasets/d4rl.py` (lines 136–160), the dataset loader processes both the standard `avoiding-d3il` and the new `avoiding-d3il-visual` environments using the exact same logic:

```python
env_state = pickle.load(f)
robot_des_pos = env_state['robot']['des_c_pos'][:, :2]
robot_c_pos = env_state['robot']['c_pos'][:, :2]
input_state = np.concatenate((robot_des_pos, robot_c_pos), axis=-1)
# ... yields the input_state
```

The data loader literally rips the 2D Cartesian coordinates (shape: 4) directly out of the `env_state` dictionary from the `.pkl` files and throws away the rest. It **never attempts to open an image file** or read camera data.

### B. The Neural Network has No Visual Encoders
In `flow_matcher_v3_avoiding_visual/models/unet1d_temporal_cond.py`, the core architecture (`Flow_matcher_U_Net_v2`) is a 1D Temporal U-Net designed for sequences of numbers, not 2D images. 

During the config printout from our test, we clearly saw:
```text
[utils/config ] Config: <class 'flow_matcher_v3_avoiding_visual.models.unet1d_temporal_cond.Flow_matcher_U_Net_v2'>
    cond_dim: 4
    transition_dim: 6
```
The model expects an observation dimension of exactly 4 (the `robot_des_pos` and `robot_c_pos`) and predicts actions of dimension 2. There are no CNN layers (like ResNet) or Vision Transformers to process high-dimensional pixel arrays.

### C. The Environment Rollout Never Asks for Images
During evaluation, the D3IL environment generates new states to feed into the model. Because the model architecture expects `cond_dim: 4`, the eval loop feeds it the 4-dimensional state directly. The environment is either instantiated without the strict `if_vision=True` requirement, or the visual components are simply ignored during the forward pass.

---

## 3. Conclusion

**The "visual" pipeline is currently a visual pipeline in name only.** 

Right now, the `flow_matching_v3_avoiding_visual` config behaves exactly like the state-based config. The whole stack—from the `d4rl.py` dataset loader to the 1D U-Net model architecture—is hardcoded to process low-dimensional spatial coordinates. 

Because it is purely processing numerical states, it never encounters a `FileNotFoundError` or shape mismatch for missing images, allowing the scripts to run perfectly.

As the operational rules in the policy specify, the current run constitutes a test of the **"visual-infrastructure-only"**. It verifies that new config blocks, dataset aliases, and routing scripts work. However, to achieve a *true* visual policy, significant architectural changes are required:
1. Rewriting the data loader to fetch and batch `.png`/`.jpg` files from the camera folders.
2. Injecting a visual encoder (e.g., an ImageNet-pretrained ResNet or ViT) into the U-Net so it accepts `(Batch, Channels, Height, Width)` inputs instead of a 4-element array.
3. Regenerating the D3IL dataset to ensure those images actually exist on disk.
