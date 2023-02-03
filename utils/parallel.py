
# MIT License

# Copyright (c) 2022 Shariq Farooq Bhat

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch.distributed as dist
import torch.nn
import torch.nn as nn
import torch.utils.data.distributed

def parallelize(config, model, find_unused_parameters=True):

    if config.gpu is not None:
        torch.cuda.set_device(config.gpu)
        model = model.cuda(config.gpu)

    config.multigpu = False
    if config.distributed:
        # Use DDP
        config.multigpu = True
        config.rank = config.rank * config.ngpus_per_node + config.gpu
        dist.init_process_group(backend=config.dist_backend, init_method=config.dist_url,
                                world_size=config.world_size, rank=config.rank)
        config.batch_size = int(config.batch_size / config.ngpus_per_node)
        # config.batch_size = 8
        config.workers = int(
            (config.num_workers + config.ngpus_per_node - 1) / config.ngpus_per_node)
        print("Device", config.gpu, "Rank",  config.rank, "batch size",
              config.batch_size, "Workers", config.workers)
        torch.cuda.set_device(config.gpu)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda(config.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu], output_device=config.gpu,
                                                          find_unused_parameters=find_unused_parameters)

    elif config.gpu is None:
        # Use DP
        config.multigpu = True
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    return model