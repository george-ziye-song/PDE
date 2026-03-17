"""Simple multi-node connectivity test."""
import os
import torch
import torch.distributed as dist

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Each rank sends its rank number, all-reduce to sum
    tensor = torch.tensor([rank], dtype=torch.float32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    hostname = os.uname().nodename
    print(f"[{hostname}] rank={rank}/{world_size}, local_rank={local_rank}, "
          f"gpu={torch.cuda.get_device_name(local_rank)}, "
          f"all_reduce_sum={tensor.item():.0f} (expect {world_size*(world_size-1)/2:.0f})")

    dist.barrier()
    if rank == 0:
        print("Multi-node test PASSED!")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
