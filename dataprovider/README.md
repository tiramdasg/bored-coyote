Build image, remove dangling images and run with mounted (your/local/data).
```cmd
sudo docker build . --tag data-provider:latest  && sudo docker rmi -f $(sudo docker images -f "dangling=true" -q && sudo docker run --rm -v /home/gabst/projects/big_data/data/:/data data-provider)

```
Above command comprises three chained commands and will stop execution, if any fails.
This can happen e.g. when no dangling images are found. Then you might want to call them independently. 
```cmd
sudo docker build . --tag data-provider:latest
sudo docker rmi -f $(sudo docker images -f "dangling=true" -q)
sudo docker run --rm -v /home/gabst/projects/big_data/data/:/data data-provider

```
