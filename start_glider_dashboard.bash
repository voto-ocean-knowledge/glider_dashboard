python initialize.py
python parquet_converter.py
panel serve glider_explorer.py --port 5006 --warm --use-xheaders --allow-websocket-origin='*' #--admin --profiler=snakeviz
