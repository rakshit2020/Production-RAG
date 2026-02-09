## Downlaod the NIM based Models

```
# Set your desired folder path
export NGC_API_KEY=<PASTE_API_KEY_HERE>
export LOCAL_NIM_CACHE=/path/to/your/folder/embedding-model  # Change this to your desired path

# Example: Save in your project folder
# export LOCAL_NIM_CACHE=/home/rakshit/rag-pipeline/models/embedding

# Create the directory
mkdir -p "$LOCAL_NIM_CACHE"

# Run the container - it will download and cache the model in your folder
docker run -it --rm \
    --gpus all \
    --shm-size=16GB \
    -e NGC_API_KEY \
    -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
    -u $(id -u) \
    -p 8000:8000 \
    nvcr.io/nim/nvidia/llama-3.2-nemoretriever-300m-embed-v2:latest
```


## Downaload vLLM model
### Method 1: Using huggingface-cli (Recommended)

```
# Install huggingface_hub if not already installed
pip install huggingface_hub

# Set your model download folder
export MODEL_FOLDER=/home/rakshit/models/llm/ministral-3-8b
mkdir -p "$MODEL_FOLDER"

# Download the model to specific folder
huggingface-cli download mistralai/Ministral-3-8B-Instruct-2512 \
  --local-dir "$MODEL_FOLDER" \
  --local-dir-use-symlinks False

# Wait for download to complete (this will take several minutes)
```

### Then run vLLM with the local folder:
```
# Use the downloaded model folder
vllm serve "$MODEL_FOLDER" \
  --host 0.0.0.0 \
  --port 8003 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --trust-remote-code \
  --dtype auto
```


## Method 1: Using Official vLLM Docker Image

```
# Set your model folder path
export LLM_FOLDER=/home/rakshit/models/llm/ministral-3-8b
export HF_HOME=/home/rakshit/models/hf-cache

# Create cache folder
mkdir -p "$HF_HOME"

# Pull vLLM Docker image
docker pull vllm/vllm-openai:latest

# Run vLLM with your local model
docker run -d \
  --name vllm-server \
  --runtime nvidia \
  --gpus all \
  --shm-size=16g \
  -v "$LLM_FOLDER:/model" \
  -v "$HF_HOME:/root/.cache/huggingface" \
  -p 8003:8000 \
  -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
  vllm/vllm-openai:latest \
  --model /model \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --trust-remote-code \
  --dtype auto
```
