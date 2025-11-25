import os
import shutil

SRC = "raw_videos"
DST = "videos_100"

os.makedirs(DST, exist_ok=True)

for fname in os.listdir(SRC):
    if fname.lower().endswith((".mp4", ".mkv")):
        src_path = os.path.join(SRC, fname)
        dst_path = os.path.join(DST, fname)
        if not os.path.exists(dst_path):
            shutil.copyfile(src_path, dst_path)

print("Videos listos en:", DST)


