# 多 GPU 训练

使用 `HSIDataParallel` 将模型分配到多张 GPU。以下以两张 GPU 为例，输入与标注格式见 [README 的接入说明](../README.md#接入)。

通过 `build_rfdetr_model(..., device="cuda:0")` 构建模型和损失，将整批 `cubes` 与 `targets` 放在 `cuda:0`，再包装模型：

```python
from hsi_refine import HSIDataParallel

model = HSIDataParallel(model, device_ids=[0, 1], output_device=0)
model.train()
criterion.train()
outputs = model(cubes, targets)
losses, roi_stats = criterion(outputs, targets)
loss = criterion.total(losses)
loss.backward()
```

`HSIDataParallel` 按图片分配完整标注，并在输出合并时恢复整批图片编号。使用 DataParallel 时采用此封装；原生 `torch.nn.DataParallel` 无法按检测任务的图片边界拆分标注。

损失模块保持单实例，在 `output_device` 上对完整输出和原始标注计算一次。推理使用 `model.eval()` 和 `model(cubes)`。

测试覆盖 CPU 上的并发、标注拆分、输出合并、损失和梯度回归；双 GPU 测试在具备两张 CUDA 设备时运行。
