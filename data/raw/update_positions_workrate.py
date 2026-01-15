import sys
import pandas as pd
import time
import re
import random
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

try:
    import cloudscraper

    USE_CLOUDSCRAPER = True
    print("✓ Cloudscraper đã được cài đặt và sẽ được sử dụng")
except ImportError:
    import requests

    USE_CLOUDSCRAPER = False
    print("⚠ Cloudscraper chưa cài đặt. Đang sử dụng requests thông thường.")
    print("Chạy: pip install cloudscraper để cải thiện khả năng bypass Cloudflare")

# Lock để đồng bộ hóa việc ghi vào DataFrame và in console
write_lock = Lock()
print_lock = Lock()


def create_scraper():
    """Tạo scraper session"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": "en-US,en;q=0.9,vi;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
        "sec-ch-ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
    }

    if USE_CLOUDSCRAPER:
        try:
            scraper = cloudscraper.create_scraper(
                browser={"browser": "chrome", "platform": "windows", "desktop": True},
                delay=10,
            )
            scraper.headers.update(headers)
            return scraper
        except Exception as e:
            print(f"⚠ Lỗi cloudscraper: {e}, chuyển sang requests")
            scraper = requests.Session()
            scraper.headers.update(headers)
            return scraper
    else:
        scraper = requests.Session()
        scraper.headers.update(headers)
        return scraper


def warm_up_session(scraper):
    """Visit trang chủ để lấy cookies trước khi crawl player page"""
    try:
        # Visit trang chủ
        response = scraper.get("https://sofifa.com", timeout=15)
        time.sleep(random.uniform(2, 4))

        # Visit trang players list
        response = scraper.get("https://sofifa.com/players", timeout=15)
        time.sleep(random.uniform(2, 4))

        return True
    except Exception as e:
        print(f"  ⚠ Không thể warm up session: {e}")
        return False


def crawl_position_and_workrate(player_url, player_name, idx, total):
    """
    Crawl position và work_rate của một cầu thủ từ trang cá nhân
    Tạo scraper riêng cho mỗi request và có retry logic

    Returns:
        dict: {'idx': int, 'positioning': str, 'work_rate': str}
    """
    # Tạo scraper riêng cho thread này
    scraper = create_scraper()

    # Warm up session lần đầu tiên
    if not warm_up_session(scraper):
        return {"idx": idx, "positioning": None, "work_rate": None}

    # Retry logic với exponential backoff
    max_retries = 5
    base_delay = 5

    for attempt in range(max_retries):
        try:
            full_url = f"https://sofifa.com{player_url}"

            with print_lock:
                if attempt > 0:
                    print(
                        f"[{idx + 1}/{total}] {player_name} - Thử lại lần {attempt + 1}"
                    )
                else:
                    print(f"[{idx + 1}/{total}] {player_name}")
                print(f"  Đang crawl: {full_url}")

            # Random delay trước khi request để tránh pattern detection
            time.sleep(random.uniform(4, 8))

            # Cập nhật Referer trước khi request
            scraper.headers.update(
                {
                    "Referer": "https://sofifa.com/players",
                    "Sec-Fetch-Site": "same-origin",
                }
            )

            response = scraper.get(full_url, timeout=30)

            if response.status_code == 429:
                with print_lock:
                    print(
                        f"  ⚠ Lỗi 429 (Too Many Requests) - Chờ {base_delay * (2 ** attempt)} giây..."
                    )
                time.sleep(base_delay * (2**attempt))
                continue

            if response.status_code == 403:
                with print_lock:
                    print(
                        f"  ⚠ Lỗi 403 (Forbidden) - Warm up lại session và thử lại..."
                    )
                # Tạo scraper mới và warm up lại
                scraper = create_scraper()
                warm_up_session(scraper)
                time.sleep(base_delay * (2**attempt))
                continue

            if response.status_code != 200:
                with print_lock:
                    print(f"  ❌ Không thể truy cập: {response.status_code}")
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2**attempt))
                    continue
                return {"idx": idx, "positioning": None, "work_rate": None}

            soup = BeautifulSoup(response.content, "html.parser")
            result = {"idx": idx, "positioning": None, "work_rate": None}

            # Lấy Positions từ <span class="pos posXX"> trong các thẻ <a>
            positions = []

            # Tìm tất cả thẻ span có class bắt đầu bằng "pos"
            pos_spans = soup.find_all("span", class_=lambda x: x and "pos" in x.split())
            for span in pos_spans:
                pos_text = span.get_text().strip()
                # Lọc các position hợp lệ (2-3 ký tự, chữ hoa)
                if (
                    pos_text
                    and 2 <= len(pos_text) <= 3
                    and pos_text.isupper()
                    and pos_text not in positions
                ):
                    positions.append(pos_text)
                if len(positions) >= 5:  # Lấy tối đa 5 positions
                    break

            if positions:
                result["positioning"] = ", ".join(positions)
                with print_lock:
                    print(f"  ✓ Positioning: {result['positioning']}")
            else:
                with print_lock:
                    print(f"  ⚠ Không tìm thấy positioning")

            # Lấy Work Rate - tìm trong các thẻ li hoặc label
            work_rate = None

            # Cách 1: Tìm trong các thẻ có chứa "Att. Work Rate" và "Def. Work Rate"
            att_rate = None
            def_rate = None

            # Tìm trong tất cả các thẻ li
            for li in soup.find_all(["li", "div", "p"]):
                text = li.get_text()
                if "Att. Work Rate" in text or "Attacking Work Rate" in text:
                    # Tìm các từ như Low, Medium, High
                    rate_match = re.search(r"(Low|Medium|High)", text, re.IGNORECASE)
                    if rate_match:
                        att_rate = rate_match.group(1).capitalize()
                if "Def. Work Rate" in text or "Defensive Work Rate" in text:
                    rate_match = re.search(r"(Low|Medium|High)", text, re.IGNORECASE)
                    if rate_match:
                        def_rate = rate_match.group(1).capitalize()

            if att_rate and def_rate:
                work_rate = f"{att_rate}/{def_rate}"
            else:
                # Cách 2: Tìm pattern "Medium/Medium" trực tiếp
                page_text = soup.get_text()
                work_rate_match = re.search(
                    r"(Low|Medium|High)\s*/\s*(Low|Medium|High)",
                    page_text,
                    re.IGNORECASE,
                )
                if work_rate_match:
                    work_rate = f"{work_rate_match.group(1).capitalize()}/{work_rate_match.group(2).capitalize()}"

            if work_rate:
                result["work_rate"] = work_rate
                with print_lock:
                    print(f"  ✓ Work Rate: {result['work_rate']}")
            else:
                with print_lock:
                    print(f"  ⚠ Không tìm thấy work rate")

            # Thành công, thoát khỏi retry loop
            return result

        except Exception as e:
            with print_lock:
                print(f"  ❌ Lỗi khi crawl (lần {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2**attempt))
            else:
                return {"idx": idx, "positioning": None, "work_rate": None}

    # Nếu hết retries mà không thành công
    return {"idx": idx, "positioning": None, "work_rate": None}


def update_csv_with_features(
    csv_path="sofifa_players.csv",
    output_path=None,
    start_index=0,
    max_players=None,
    num_threads=10,
):
    """
    Đọc CSV hiện tại, crawl positioning và work_rate cho mỗi cầu thủ, và cập nhật CSV
    Sử dụng multi-threading để tăng tốc độ crawl

    Args:
        csv_path: Đường dẫn file CSV đầu vào
        output_path: Đường dẫn file CSV đầu ra (None = ghi đè file gốc)
        start_index: Vị trí bắt đầu crawl (để tiếp tục từ giữa chừng)
        max_players: Số lượng cầu thủ tối đa cần crawl (None = crawl hết)
        num_threads: Số lượng threads để crawl song song
    """
    # Đọc CSV
    print(f"📖 Đang đọc file: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Đã load {len(df)} cầu thủ")
    except Exception as e:
        print(f"❌ Lỗi khi đọc file: {e}")
        return

    # Kiểm tra cột Player_URL
    if "Player_URL" not in df.columns:
        print("❌ Không tìm thấy cột 'Player_URL' trong CSV")
        return

    # Tạo cột mới nếu chưa có (với dtype object)
    if "positioning" not in df.columns:
        df["positioning"] = pd.Series(dtype="object")
    if "work_rate" not in df.columns:
        df["work_rate"] = pd.Series(dtype="object")

    # Xác định phạm vi crawl
    end_index = min(start_index + max_players, len(df)) if max_players else len(df)
    total_to_crawl = end_index - start_index

    print(
        f"\n📊 Bắt đầu crawl từ cầu thủ #{start_index + 1} đến #{end_index} ({total_to_crawl} cầu thủ)"
    )
    print(f"🧵 Sử dụng {num_threads} threads để tăng tốc độ")
    print("=" * 70)

    success_count = 0
    error_count = 0
    skip_count = 0

    # Tạo danh sách các công việc cần crawl
    tasks = []
    for idx in range(start_index, end_index):
        player_url = df.at[idx, "Player_URL"]
        player_name = df.at[idx, "Name"] if "Name" in df.columns else f"Player {idx+1}"

        # Bỏ qua nếu thiếu URL
        if pd.isna(player_url) or not player_url:
            skip_count += 1
            continue

        # Chỉ crawl nếu positioning chưa có dữ liệu
        if pd.notna(df.at[idx, "positioning"]):
            skip_count += 1
            continue

        tasks.append((idx, player_url, player_name))

    print(f"\n🎯 Tổng số cầu thủ cần crawl: {len(tasks)}")
    print(f"⊗ Đã bỏ qua: {skip_count} cầu thủ\n")

    if not tasks:
        print("✓ Không có cầu thủ nào cần crawl!")
        return

    # Crawl song song với multi-threading (giảm số threads để tránh bị ban)
    completed = 0
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit tất cả các tasks
        future_to_task = {
            executor.submit(
                crawl_position_and_workrate, player_url, player_name, idx, len(df)
            ): (idx, player_name)
            for idx, player_url, player_name in tasks
        }

        # Xử lý kết quả khi hoàn thành
        for future in as_completed(future_to_task):
            idx, player_name = future_to_task[future]
            try:
                result = future.result()

                # Cập nhật DataFrame với thread-safe lock
                with write_lock:
                    if result["positioning"]:
                        df.at[result["idx"], "positioning"] = result["positioning"]
                    if result["work_rate"]:
                        df.at[result["idx"], "work_rate"] = result["work_rate"]

                    if result["positioning"] or result["work_rate"]:
                        success_count += 1
                    else:
                        error_count += 1

                    completed += 1

                    # Auto-save mỗi 10 cầu thủ
                    if completed % 10 == 0:
                        temp_output = output_path or csv_path
                        df.to_csv(temp_output, index=False, encoding="utf-8-sig")
                        with print_lock:
                            print(
                                f"\n💾 Tự động lưu: {completed}/{len(tasks)} cầu thủ đã xử lý"
                            )
                            print(
                                f"   ✓ Thành công: {success_count} | ❌ Lỗi: {error_count} | ⊗ Bỏ qua: {skip_count}"
                            )
                            print()

            except Exception as e:
                with print_lock:
                    print(f"\n❌ Lỗi khi xử lý {player_name}: {e}\n")
                error_count += 1

    # Lưu file cuối cùng
    output_file = output_path or csv_path
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print(f"✅ HOÀN TẤT!")
    print(f"📊 Thống kê:")
    print(f"   - Tổng số xử lý: {len(tasks)}")
    print(f"   - Thành công: {success_count}")
    print(f"   - Lỗi: {error_count}")
    print(f"   - Bỏ qua: {skip_count}")
    print(f"   - Đã lưu vào: {output_file}")

    # Hiển thị preview
    print(f"\n👀 Preview dữ liệu mới:")
    cols_to_show = ["Name", "positioning", "work_rate"]
    available_cols = [col for col in cols_to_show if col in df.columns]
    print(df[available_cols].tail(10))


if __name__ == "__main__":
    # CẤU HÌNH - Thay đổi các giá trị này theo nhu cầu
    # ============================================================
    CSV_INPUT = "sofifa_players.csv"  # File CSV đầu vào (chứa Player_URL)
    CSV_OUTPUT = "sofifa_players.csv"  # File CSV đầu ra (None = ghi đè file gốc)
    START_INDEX = 0  # Vị trí bắt đầu (0 = từ đầu)
    MAX_PLAYERS = None  # Số lượng tối đa (None = crawl hết)
    NUM_THREADS = 1  # Số threads (giảm xuống 1 để tránh lỗi 403/429)
    # ============================================================

    print("=" * 70)
    print("  CRAWL POSITIONING & WORK RATE TỪ PLAYER_URL")
    print("=" * 70)
    print(f"\n⚙ Cấu hình:")
    print(f"   - File đầu vào: {CSV_INPUT}")
    print(f"   - File đầu ra: {CSV_OUTPUT}")
    print(f"   - Bắt đầu từ: cầu thủ #{START_INDEX + 1}")
    print(f"   - Số lượng: {'Tất cả' if MAX_PLAYERS is None else MAX_PLAYERS}")
    print(f"   - Số threads: {NUM_THREADS} (crawl song song)")
    print(f"   - Nguồn: Sử dụng Player_URL có sẵn trong CSV")
    print()

    update_csv_with_features(
        csv_path=CSV_INPUT,
        output_path=CSV_OUTPUT,
        start_index=START_INDEX,
        max_players=MAX_PLAYERS,
        num_threads=NUM_THREADS,
    )
