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
