==========================================!!!START TRAINING!!!==========================================
Traceback (most recent call last):
  File "main.py", line 344, in <module>
    criterion, optimizer, epoch, args, batch_logger, epoch_logger, memory_bank)
  File "main.py", line 106, in train
    unnormed_vec, normed_vec = model(data)
  File "/home/ubuntu/anaconda3/envs/torch140/lib/python3.7/site-packages/torch/nn/modules/module.py", line 532, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/ubuntu/anaconda3/envs/torch140/lib/python3.7/site-packages/torch/nn/parallel/data_parallel.py", line 150, in forward
    return self.module(*inputs[0], **kwargs[0])
  File "/home/ubuntu/anaconda3/envs/torch140/lib/python3.7/site-packages/torch/nn/modules/module.py", line 532, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/ubuntu/codes/driver_monitor/Driver-Anomaly-Detection/models/efficientnet.py", line 191, in forward
    x = self.extract_features(inputs)
  File "/home/ubuntu/codes/driver_monitor/Driver-Anomaly-Detection/models/efficientnet.py", line 173, in extract_features
    x = self._swish(self._bn0(self._conv_stem(inputs)))
  File "/home/ubuntu/anaconda3/envs/torch140/lib/python3.7/site-packages/torch/nn/modules/module.py", line 532, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/ubuntu/codes/driver_monitor/Driver-Anomaly-Detection/models/utils.py", line 154, in forward
    x = F.conv3d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
RuntimeError: Given groups=1, weight of size 32 3 3 3 3, expected input[90, 1, 17, 113, 113] to have 3 channels, but got 1 channels instead


==========================================!!!START TRAINING!!!==========================================
-------
torch.Size([90, 1, 16, 112, 112])
torch.Size([90, 2048, 1, 4, 4])
torch.Size([90, 512, 1, 4, 4])
torch.Size([90, 512])
torch.Size([90, 512])
normalization constant Z is set to 30730534.0
-------
torch.Size([10, 1, 16, 112, 112])
torch.Size([10, 2048, 1, 4, 4])
torch.Size([10, 512, 1, 4, 4])
torch.Size([10, 512])
torch.Size([10, 512])
Training Process is running: 1/250  | Batch: 0 | Loss: 5.558858394622803 (5.558858394622803) | Probs: 0.01215553842484951 (0.01215553842484951)
-------
torch.Size([90, 1, 16, 112, 112])
torch.Size([90, 2048, 1, 4, 4])
torch.Size([90, 512, 1, 4, 4])
torch.Size([90, 512])
torch.Size([90, 512])
-------
torch.Size([10, 1, 16, 112, 112])
torch.Size([10, 2048, 1, 4, 4])
torch.Size([10, 512, 1, 4, 4])
torch.Size([10, 512])
torch.Size([10, 512])
Training Process is running: 1/250  | Batch: 1 | Loss: 5.6625447273254395 (5.610701560974121) | Probs: 0.01313744392246008 (0.012646491173654795)
-------
torch.Size([90, 1, 16, 112, 112])
torch.Size([90, 2048, 1, 4, 4])
torch.Size([90, 512, 1, 4, 4])
torch.Size([90, 512])
torch.Size([90, 512])
-------
torch.Size([10, 1, 16, 112, 112])
torch.Size([10, 2048, 1, 4, 4])
torch.Size([10, 512, 1, 4, 4])
torch.Size([10, 512])
torch.Size([10, 512])
Training Process is running: 1/250  | Batch: 2 | Loss: 5.564438343048096 (5.595280488332112) | Probs: 0.011484349146485329 (0.012259110497931639)
-------
torch.Size([90, 1, 16, 112, 112])
torch.Size([90, 2048, 1, 4, 4])
torch.Size([90, 512, 1, 4, 4])
torch.Size([90, 512])
torch.Size([90, 512])
-------
torch.Size([10, 1, 16, 112, 112])
torch.Size([10, 2048, 1, 4, 4])
torch.Size([10, 512, 1, 4, 4])
torch.Size([10, 512])
torch.Size([10, 512])
Training Process is running: 1/250  | Batch: 3 | Loss: 6.269207954406738 (5.763762354850769) | Probs: 0.011962785385549068 (0.012185029219835997)
-------
torch.Size([90, 1, 16, 112, 112])
torch.Size([90, 2048, 1, 4, 4])
torch.Size([90, 512, 1, 4, 4])
torch.Size([90, 512])
torch.Size([90, 512])
-------
torch.Size([10, 1, 16, 112, 112])
torch.Size([10, 2048, 1, 4, 4])
torch.Size([10, 512, 1, 4, 4])
torch.Size([10, 512])
torch.Size([10, 512])
Training Process is running: 1/250  | Batch: 4 | Loss: 6.015622615814209 (5.814134407043457) | Probs: 0.012230778113007545 (0.012194178998470306)
-------
torch.Size([90, 1, 16, 112, 112])
torch.Size([90, 2048, 1, 4, 4])
torch.Size([90, 512, 1, 4, 4])
torch.Size([90, 512])
torch.Size([90, 512])
-------
