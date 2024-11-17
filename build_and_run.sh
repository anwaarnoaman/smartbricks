version="1.0.0" 
repo=""
 
docker build -t "$repo"sb-app:"$version" -f Dockerfile.Gradio . 
docker build -t "$repo"sb-api:"$version" -f Dockerfile.Flask . 
 

 