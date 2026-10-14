"""
Công cụ tự động hóa đẩy và quản lý Notebook chạy trên Kaggle GPU qua Kaggle REST API.
Hỗ trợ: push, status, output.
"""
import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
import zipfile
import io

ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")
NOTEBOOK_PATH = os.path.join(os.path.dirname(__file__), "slq_multilayer_kaggle.ipynb")
DEFAULT_SLUG = "slq-multi-layer-internvl"
DEFAULT_TITLE = "SLQ Multi-Layer InternVL"
DATASET_SOURCE = "awsaf49/coco-2017-dataset"


def load_credentials():
    env_vars = {}
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env_vars[k.strip()] = v.strip().strip('"').strip("'")

    username = env_vars.get("KAGGLE_USERNAME") or os.environ.get("KAGGLE_USERNAME")
    api_key = env_vars.get("KAGGLE_API_KEY") or os.environ.get("KAGGLE_API_KEY")

    if not username or not api_key:
        print("LỖI: Chưa cấu hình KAGGLE_USERNAME hoặc KAGGLE_API_KEY trong .env")
        sys.exit(1)

    return username, api_key


def get_headers(api_key):
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "Antigravity-KaggleRunner/1.0",
    }


def push_kernel(slug=DEFAULT_SLUG, title=DEFAULT_TITLE, notebook_path=None):
    username, api_key = load_credentials()
    nb_path = notebook_path or NOTEBOOK_PATH
    if not os.path.exists(nb_path):
        print(f"LỖI: Không tìm thấy file notebook tại {nb_path}")
        sys.exit(1)

    with open(nb_path) as f:
        notebook_content = f.read()

    full_slug = f"{username}/{slug}"
    payload = {
        "slug": full_slug,
        "newTitle": title,
        "text": notebook_content,
        "language": "python",
        "kernelType": "notebook",
        "isPrivate": True,
        "enableGpu": True,
        "enableInternet": True,
        "datasetDataSources": [DATASET_SOURCE],
    }

    url = "https://www.kaggle.com/api/v1/kernels/push"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=get_headers(api_key), method="POST")

    print(f"Đang đẩy notebook lên Kaggle: {full_slug} (GPU: True, Internet: True)...")
    try:
        with urllib.request.urlopen(req) as resp:
            body = resp.read().decode("utf-8")
            res = json.loads(body)
            print("ĐẨY THÀNH CÔNG!")
            print(f"URL: {res.get('url', f'https://www.kaggle.com/code/{full_slug}')}")
            print(f"Phiên bản: {res.get('versionNumber', 'Mới nhất')}")
            print("\nKaggle đã bắt đầu kích hoạt và xếp lịch chạy trên GPU.")
            print(f"Để kiểm tra tiến độ: python SLQ_reproduce/kaggle_runner.py status")
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8")
        print(f"LỖI HTTP {e.code}: {err}")
        sys.exit(1)


def get_status(slug=DEFAULT_SLUG):
    username, api_key = load_credentials()
    full_slug = f"{username}/{slug}"
    url = f"https://www.kaggle.com/api/v1/kernels/status?userName={username}&kernelSlug={slug}"

    req = urllib.request.Request(url, headers=get_headers(api_key), method="GET")
    try:
        with urllib.request.urlopen(req) as resp:
            body = resp.read().decode("utf-8")
            res = json.loads(body)
            status = res.get("status", "unknown")
            msg = res.get("failureMessage")
            print(f"Trạng thái notebook '{full_slug}': {status.upper()}")
            if msg:
                print(f"Chi tiết thông báo: {msg}")
            return status
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8")
        print(f"LỖI HTTP {e.code}: {err}")
        sys.exit(1)


def get_output(slug=DEFAULT_SLUG, out_dir="results_kaggle"):
    username, api_key = load_credentials()
    full_slug = f"{username}/{slug}"
    url = f"https://www.kaggle.com/api/v1/kernels/output?userName={username}&kernelSlug={slug}"

    req = urllib.request.Request(url, headers=get_headers(api_key), method="GET")
    print(f"Đang tải output của notebook '{full_slug}'...")
    try:
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
            # Kaggle output có thể trả về json hoặc zip file
            os.makedirs(out_dir, exist_ok=True)
            try:
                # Kiểm tra nếu là file zip
                z = zipfile.ZipFile(io.BytesIO(data))
                z.extractall(out_dir)
                print(f"Đã giải nén toàn bộ kết quả vào thư mục: {out_dir}")
                for f in z.namelist():
                    print(f"  - {f}")
            except zipfile.BadZipFile:
                # Nếu là JSON chứa danh sách files hoặc log
                res = json.loads(data.decode("utf-8"))
                out_path = os.path.join(out_dir, "output.json")
                with open(out_path, "w") as f:
                    json.dump(res, f, indent=2)
                print(f"Đã lưu thông tin output vào: {out_path}")
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8")
        print(f"LỖI HTTP {e.code}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kaggle Notebook Runner")
    parser.add_argument("action", choices=["push", "status", "output"], help="Hành động cần thực hiện")
    parser.add_argument("--slug", default=DEFAULT_SLUG, help="Slug của kernel")
    parser.add_argument("--title", default=DEFAULT_TITLE, help="Tiêu đề notebook")
    parser.add_argument("--out_dir", default="results_kaggle", help="Thư mục lưu output khi tải về")
    parser.add_argument("--notebook", default=None, help="Đường dẫn tới file notebook cần đẩy")

    args = parser.parse_args()
    if args.action == "push":
        push_kernel(args.slug, args.title, args.notebook)
    elif args.action == "status":
        get_status(args.slug)
    elif args.action == "output":
        get_output(args.slug, args.out_dir)
