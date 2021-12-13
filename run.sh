NAME=prognostics

# Build Image
echo "Building image..."
docker build -t $NAME .

# Run Container
echo "Running container..."
docker run \
  -it -d --rm \
  --name $NAME \
  -u $(id -u):$(id -g) \
  -v $PWD:/work \
  -p 8888:8888 \
  --gpus all \
  $NAME

# Show logs
sleep 2 && docker logs $NAME
