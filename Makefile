APPNAME = mlapp
PORT = 8501

build-image:
	docker build -t $(APPNAME) .

launch:
	docker run --rm -p $(PORT):$(PORT) --name="$(APP_NAME)" $(APPNAME)

stop: ## Stop and remove a running container
	docker stop $(APP_NAME); docker rm $(APP_NAME)

rm-image:
	docker image rm $(APPNAME)