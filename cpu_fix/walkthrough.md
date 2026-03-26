# Walkthrough: Fixing Device Agnostic Model Loading

## Issue
When running `FM_test/eval_FM_CPU.py` on a CPU-only environment, the model deserialization failed because `torch.load` defaulted to restoring tensors to the GPU (where they were initially trained). Flow Matching was lacking `map_location` logic that was present in the Diffusion baseline.

## Changes Made
1. **`flow_matcher/utils/training.py`**: Added `map_location=self.device` to the `Trainer.load()` method. This ensures that weights are mapped to the correct runtime device (e.g. `cpu`) instead of crashing when attempting to find `cuda:0`.

```diff
-        data = torch.load(loadpath)
+        data = torch.load(loadpath, map_location=self.device)
```

2. **`flow_matcher/utils/serialization.py`**: The previous implementation overrode the string variable `device` but failed to inject it inside the pickled `trainer_config._dict`. I added the missing overrides identical to those found in `diffuser/utils/serialization.py`:

```diff
-    ## Override pickled device with the requested device
-    model_config._device = device
-    diffusion_config._device = device
+    if hasattr(model_config, '_device'):
+        model_config._device = device
+    if hasattr(diffusion_config, '_device'):
+        diffusion_config._device = device
+    if hasattr(trainer_config, '_device'):
+        trainer_config._device = None
+
     trainer_config._dict['results_folder'] = os.path.join(*loadpath)
+    trainer_config._dict['train_device'] = device
```

With these fixes, the serialized Trainer object will respect the device you requested (e.g., CPU) when unpacking `torch.load` weights.
