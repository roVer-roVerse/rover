FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update -y \
    && apt-get install -y git curl


RUN --mount=type=cache,target=/root/.cache/pip \
  pip install lighteval accelerate vllm ray nvitop

ADD recommended_set.txt /

RUN --mount=type=cache,target=/root/.cache \
    lighteval accelerate \
        --model_args "pretrained=TinyPixel/small-1,dtype=bfloat16" \
        --tasks "/recommended_set.txt" \
        --override_batch_size 8 \
        --output_dir="/outputs/small-1"
