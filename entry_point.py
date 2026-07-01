import subprocess
import subprocess as subp
import time

if __name__ == "__main__":
    while True:
        # while True:
        # p = subp.Popen(["python3", "glider_explorer.py"] + sys.argv[1:])
        subprocess.run(["python", "initialize.py"])
        subprocess.run(["python", "parquet_converter.py"])
        p = subp.Popen(
            [
                "panel",
                "serve",
                "glider_explorer.py",
                "--port",
                "5006",
                "--allow-websocket-origin='*'",
            ]
        )
        time.sleep(7200)
        print("time is up. restarting service")
        p.terminate()
        p.wait()
