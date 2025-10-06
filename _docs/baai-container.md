I want to use BAAI's FlagEmbedding models https://huggingface.co/BAAI/bge-reranker-v2-m3#usage
here is the code snippet from the model card:
```python
from FlagEmbedding import FlagReranker
reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

score = reranker.compute_score(['query', 'passage'])
print(score) # -5.65234375

# You can map the scores into 0-1 by set "normalize=True", which will apply sigmoid function to the score
score = reranker.compute_score(['query', 'passage'], normalize=True)
print(score) # 0.003497010252573502

scores = reranker.compute_score([['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']])
print(scores) # [-8.1875, 5.26171875]

# You can map the scores into 0-1 by set "normalize=True", which will apply sigmoid function to the score
scores = reranker.compute_score([['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']], normalize=True)
print(scores) # [0.00027803096387751553, 0.9948403768236574]

```

My goal is to build a containerized application that uses this model to provide a web service 
for reranking text pairs. 
The application should expose an endpoint where users can send 
POST requests with JSON payloads containing text pairs, 
and the service will respond with the reranking scores.

Container should be deployable on a cloud platform Azure.
Please provide a complete example including:
1. The application code (e.g., using Flask or FastAPI).
2. The Dockerfile to containerize the application.
   - Ollama enviromnetment with FlagEmbedding installed.
3. Instructions to build and run the Docker container. 
4. Terraform script to deploy the container on Azure.
5. Any additional configuration files needed.

Let's plan the steps to achieve this goal.

--------------

=> CACHED [builder 4/5] COPY requirements.txt .                                                                                                                         0.0s
=> [builder 5/5] RUN pip install --no-cache-dir --upgrade pip &&     pip install --no-cache-dir --user -r requirements.txt                                            424.4s
=> => # Downloading pydantic_core-2.14.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.1 MB)                                                                
=> => #    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/2.1 MB 2.2 MB/s  0:00:00                                                                                            
=> => # Downloading anyio-3.7.1-py3-none-any.whl (80 kB)                                                                                                                    
=> => # Downloading pluggy-1.6.0-py3-none-any.whl (20 kB)                                                                                                                   
=> => # Downloading starlette-0.27.0-py3-none-any.whl (66 kB)                                                                                                               
=> => # Downloading torch-2.8.0-cp310-cp310-manylinux_2_28_x86_64.whl (888.0 MB)                             

It is hanging on step Downloading pydantic_core-2.14.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.1 MB) 
2.1 MB is not as huge, but it is taking too long time.

---------
Use this ollama model:
   ollama pull xitao/bge-reranker-v2-m3

Hybrid: FastAPI + FlagEmbedding running alongside Ollama server


