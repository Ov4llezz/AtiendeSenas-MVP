import os
import json
import time
import sys
import urllib.request
import random
import logging
import subprocess

# ---------------------------------------
# CONFIG
# ---------------------------------------
NSLT_JSON = "nslt_100.json"
WLASL_JSON = "WLASL_v0.3.json"
SAVE_DIR = "raw_videos_wlasl100"

youtube_downloader = "yt-dlp"


# ---------------------------------------
# LOGGING SIN UNICODE (Windows-friendly)
# ---------------------------------------
logging.basicConfig(
    filename="download_{}.log".format(int(time.time())),
    filemode="w",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(message)s"))
logging.getLogger().addHandler(handler)


# ---------------------------------------
# Load allowed video ids
# ---------------------------------------
def load_allowed_video_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ids = set(data.keys())
    logging.info("[INFO] Video IDs en nslt_100.json: {}".format(len(ids)))
    return ids


# ---------------------------------------
# Build video_id → url map
# ---------------------------------------
def build_id_to_url_map(wlasl_json_path, allowed_ids):
    with open(wlasl_json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    id2url = {}

    for entry in dataset:
        for inst in entry.get("instances", []):
            vid = str(inst.get("video_id"))
            url = inst.get("url")

            if vid in allowed_ids and vid not in id2url:
                id2url[vid] = url

    logging.info("[INFO] IDs encontrados en WLASL_v0.3.json: {}".format(len(id2url)))

    missing = allowed_ids - set(id2url.keys())
    logging.info("[INFO] IDs faltantes: {}".format(len(missing)))

    return id2url


# ---------------------------------------
# Normal requests (ASLPro + others)
# ---------------------------------------
def request_video(url, referer=None):
    headers = {"User-Agent": "Mozilla/5.0"}
    if referer:
        headers["Referer"] = referer

    req = urllib.request.Request(url, headers=headers)
    resp = urllib.request.urlopen(req)
    data = resp.read()
    return data


def save_binary(data, path):
    with open(path, "wb+") as f:
        f.write(data)


# ---------------------------------------
# YouTube download
# ---------------------------------------
def download_youtube(url, saveto, video_id):
    output = os.path.join(saveto, "{}.mp4".format(video_id))

    if os.path.exists(output):
        logging.info("[SKIP] {} ya existe (YouTube)".format(video_id))
        return True

    cmd = [
        youtube_downloader,
        "-o", output,
        "-f", "mp4/best",
        "--no-warnings",
        "--quiet",
        "--retries", "10",
        url
    ]

    result = subprocess.run(cmd)

    return result.returncode == 0


# ---------------------------------------
# ASLPro
# ---------------------------------------
def download_aslpro(url, saveto, video_id):
    output = os.path.join(saveto, "{}.swf".format(video_id))

    if os.path.exists(output):
        logging.info("[SKIP] {} ya existe (ASLPro)".format(video_id))
        return True

    try:
        data = request_video(url, referer="http://www.aslpro.com/")
        save_binary(data, output)
        return True
    except:
        return False


# ---------------------------------------
# Other .mp4 links
# ---------------------------------------
def download_other(url, saveto, video_id):
    output = os.path.join(saveto, "{}.mp4".format(video_id))

    if os.path.exists(output):
        logging.info("[SKIP] {} ya existe (other)".format(video_id))
        return True

    try:
        data = request_video(url)
        save_binary(data, output)
        return True
    except:
        return False


# ---------------------------------------
# Select method
# ---------------------------------------
def select_method(url):
    if "youtube" in url or "youtu.be" in url:
        return download_youtube
    if "aslpro" in url:
        return download_aslpro
    return download_other


# ---------------------------------------
# MAIN DOWNLOAD LOOP (Windows-safe)
# ---------------------------------------
def download_all(id2url, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    total = len(id2url)
    ok = 0
    fail = 0

    for i, (vid, url) in enumerate(id2url.items(), start=1):

        safe_msg = "[{}/{}] Descargando video_id={} URL={}".format(i, total, vid, url)
        logging.info(safe_msg)

        method = select_method(url)
        success = method(url, save_dir, vid)

        if success:
            ok += 1
        else:
            fail += 1

        time.sleep(random.uniform(1.0, 2.0))

    logging.info("[FIN] Exitos: {}  Fallos: {}  Total: {}".format(ok, fail, total))


# ---------------------------------------
# MAIN
# ---------------------------------------
if __name__ == "__main__":
    logging.info("[INIT] Cargando IDs WLASL100...")
    allowed_ids = load_allowed_video_ids(NSLT_JSON)

    logging.info("[INIT] Encontrando URLs...")
    id2url = build_id_to_url_map(WLASL_JSON, allowed_ids)

    logging.info("[INIT] Descargando SOLO WLASL100...")
    download_all(id2url, SAVE_DIR)



