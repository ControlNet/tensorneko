import subprocess


ffmpeg_available = subprocess.run('ffmpeg -version', stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL, shell=True).returncode == 0
