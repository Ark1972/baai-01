Here is some performance testing results for /rerank/query endpoint:
20 passages:    35.10s
10 passages:    18.39s 
 5 passages:     9.71s
 5 passages (Shorter passages): 3.22s

- I am running this on a VM with the following specs:
```
"PublishAllPorts": false,
		"ReadonlyRootfs": false,
		"SecurityOpt": null,
		"UTSMode": "",
		"UsernsMode": "",
		"ShmSize": 67108864,
		"Runtime": "runc",
		"Isolation": "",
		"CpuShares": 0,
		"Memory": 4294967296,
		"NanoCpus": 2000000000,
		"CgroupParent": "",
		"BlkioWeight": 0,
		"BlkioWeightDevice": null,
		"BlkioDeviceReadBps": null,
		"BlkioDeviceWriteBps": null,
		"BlkioDeviceReadIOps": null,
		"BlkioDeviceWriteIOps": null,
		"CpuPeriod": 0,
		"CpuQuota": 0,
		"CpuRealtimePeriod": 0,
		"CpuRealtimeRuntime": 0,
		"CpusetCpus": "",
		"CpusetMems": "",
		"Devices": null,
		"DeviceCgroupRules": null,
		"DeviceRequests": null,
		"MemoryReservation": 2147483648,
		"MemorySwap": 8589934592,
		"MemorySwappiness": null,
```
- re-ranker runs on CPU

I am interesting in my abilities in Microsoft Azure:
- Which calculation serviceses can I use?
- Can I run re-ranking CUDA?
- What is the recommended instance size for this kind of workload?

Let's try to evaluate the approximate execution time reducing?

What kind of protective services should I consider (I am not familiar with Azure, 
but have AWS experiences)?
- API Gateway
- VPC
- Policies
- Security groups
- IAM roles
- etc.

---
# Current Performance Analysis

My CPU performance (2 cores, 4GB RAM):
- 20 passages: 35.10s (1.76s per passage)
- 10 passages: 18.39s (1.84s per passage)
- 5 passages: 9.71s (1.94s per passage)
- 5 short passages: 3.22s (0.64s per passage)

# Azure Compute Options

## 1. Azure Container Apps with Serverless GPU ⭐ RECOMMENDED

Best for: Production workloads with variable demand

GPU Options:
- NVIDIA T4 (16GB) - Inference-optimized, cost-effective
- NVIDIA A100 (40GB/80GB) - High performance for large models

Advantages:
✅ Scale-to-zero - Pay only when processing requests✅ Per-second billing - No idle costs✅        
Auto-scaling - Handles traffic spikes automatically✅ Fully managed - No VM maintenance✅
Latest CUDA support

Disadvantages:
❌ Cold start latency (mitigated with min replicas)

## 2. Azure Virtual Machines (NC-series)

Best for: Predictable, sustained workloads

Options:
- NCasT4_v3 (Tesla T4): $384/month ($0.53/hour)
- NCv3 (Tesla V100): ~$3.06/hour

Advantages:
✅ Predictable performance✅ No cold starts✅ Full control over environment

Disadvantages:
❌ Pay 24/7 even when idle❌ Manual scaling required❌ VM management overhead

## Realistic expectation: 
5-8x speedup with T4 GPU and moderate optimization effort.
10-20x faster with A100 GPU and significant optimization.

