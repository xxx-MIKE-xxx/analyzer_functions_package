import importlib.metadata
for dist in importlib.metadata.distributions():
    try:
        print(dist.metadata['Name'])
    except Exception as e:
        print(f'❌ Error in {dist._path}: {e}')

