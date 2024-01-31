# project3
### How to Run
- Clone this repository with `git clone https://github.com/WalkerRout/project3.git`
- Navigate into the project with `cd project3/`
- Ensure the docker daemon is installed and running (simply install and run docker desktop https://docs.docker.com/get-docker/)
- Run the project and install the model
	- `docker compose up --build`
  - (in another shell) `docker exec ollama ollama pull mistral`
- Open `http://localhost:5000/` in browser
> Stop the project docker containers with `docker compose down`
> 
> Remove docker generated images/containers/caches with `docker system prune`