APPNAME = streamlit_app
PORT = 8501

build-image:
	docker build -t $(APPNAME) .

launch:
	docker run --rm -p $(PORT):$(PORT) --name="$(APP_NAME)" $(APPNAME)

stop: ## Stop and remove a running container
	docker stop $(APP_NAME); docker rm $(APP_NAME)

rm-image:
	docker image rm $(APPNAME)

package-wheel:
	pip install wheel && python setup.py bdist_wheel

install-wheel:
	pip install dist/ml_for_nature-1.0-py3-none-any.whl

install-dev-mod:
	pip install -e .
