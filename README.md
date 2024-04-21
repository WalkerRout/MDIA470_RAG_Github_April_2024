# project3
### Setup and Run
- Clone this repository with; \
  `git clone https://github.com/WalkerRout/project3.git`
- Navigate into the project with; \
  `cd project3/`
- Ensure the docker daemon is installed and running (simply install and run docker desktop https://docs.docker.com/get-docker/)
- Build the project with; \
  `$ docker compose up --build app`
- This will likely give an error ("dependency failed to start: container ollama is unhealthy"), but that is expected; images are built, but not yet set up. The language model needs to be pulled to make the service healthy for the app. In another terminal window/tab, run; \
  `$ docker exec ollama ollama pull mistral`
- Re-build other services and run the application with; \
  `$ docker compose up`
  - Make sure to wait until the policies are embedded, currently the user can use the application before this step is complete which leads to Qdrant error "Not found: Collection `ubc_pdf_policies` doesn't exist!"}. By separating the app and pull_policies services, development changes can be made quickly without having to re-pull and embed policies for each new change
- View the application's website at `http://localhost:5000/` in a web browser

Of course, if you want to run the application with a single command, just chain all the above together and run with; \
  `$ docker compose up --build app; docker exec ollama ollama pull mistral; docker compose up` \
(**Don't forget to wait a bit for pull_policies to generate embeddings!**)

### Shutting Down and Reclaiming Space
- Stop the project docker containers with; \
  `$ docker compose down`
- And remove docker generated images/containers/caches with; \
  `$ docker system prune`
- It is recommended to use the 'Clean/Purge data' option in Docker Desktop to fully reclaim space

### Pitfalls/Tips
- Trying to submit a prompt before the policies are finished pulling will cause the app to crash
  - Wait for the pull_policies container to output "DONE PULLING AND EMBEDDING POLICIES"
- Trying to run the project on a machine without a separate, dedicated GPU will cause errors if GPU support is enabled for the ollama container in docker-compose.yml
  - Delete or comment out all the "enable GPU support" lines between the '-----------------------------' comments
- If you need to stop the app and want to re-run it afterwards (**after everything has been pulled and setup**), just use;\
  `$ docker compose up app`

### Support
- Contact `walkerrout04@gmail.com` for any questions.