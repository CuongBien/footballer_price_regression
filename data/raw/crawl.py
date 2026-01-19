import sys

try:
    import cloudscraper

    USE_CLOUDSCRAPER = True
    print("✓ Cloudscraper đã được cài đặt và sẽ được sử dụng")
except ImportError:
    import requests

    USE_CLOUDSCRAPER = False
    print("⚠ Cloudscraper chưa cài đặt. Đang sử dụng requests thông thường.")
    print("Chạy: pip install cloudscraper để cải thiện khả năng bypass Cloudflare")

from bs4 import BeautifulSoup
import pandas as pd
import time
import re


def parse_value(value_str):
    """Chuyển đổi chuỗi €110.5M hoặc €500K thành số thực"""
    if "M" in value_str:
        return float(re.sub(r"[€M]", "", value_str)) * 1000000
    elif "K" in value_str:
        return float(re.sub(r"[€K]", "", value_str)) * 1000
    return 0


def crawl_player_details(scraper, player_url):
    """Crawl thông tin chi tiết của một cầu thủ từ trang cá nhân"""
    try:
        full_url = f"https://sofifa.com{player_url}",
        response = scraper.get(full_url, timeout=15)

        if response.status_code != 200:
            print(f"  Không thể truy cập trang cầu thủ: {response.status_code}")
            return {}

        soup = BeautifulSoup(response.content, "html.parser")
        details = {}

        # Phương pháp 1: Lấy từ JSON-LD (structured data)
        json_ld = soup.find("script", {"type": "application/ld+json"})
        if json_ld:
            import json

            try:
                data = json.loads(json_ld.string)
                if "height" in data:
                    height_match = re.search(r"(\d+)", data["height"])
                    if height_match:
                        details["Height_cm"] = int(height_match.group(1))
                if "weight" in data:
                    weight_match = re.search(r"(\d+)", data["weight"])
                    if weight_match:
                        details["Weight_kg"] = int(weight_match.group(1))
                if "nationality" in data:
                    details["Nationality"] = data["nationality"]
            except:
                pass

        # Phương pháp 2: Parse thông tin từ profile section
        # Tìm Preferred Foot
        page_text = soup.get_text()
        preferred_foot_match = re.search(r"Preferred foot\s+(Left|Right)", page_text)
        if preferred_foot_match:
            details["Preferred_Foot"] = preferred_foot_match.group(1)

        # Đếm stars cho Skill Moves và Weak Foot
        profile_section = soup.find("div", class_="attribute")
        if profile_section:
            paragraphs = profile_section.find_all("p")
            for p in paragraphs:
                p_text = p.get_text()
                if "Skill moves" in p_text:
                    stars = p.find_all("svg", class_="star")
                    details["Skill_Moves"] = len(stars)
                elif "Weak foot" in p_text:
                    stars = p.find_all("svg", class_="star")
                    details["Weak_Foot"] = len(stars)

        # Phương pháp 3: Lấy chi tiết stats từ các thẻ <em>
        # Tìm tất cả stats có dạng: <em title="value">value</em>
        stat_names = [
            "Crossing",
            "Finishing",
            "Heading accuracy",
            "Short passing",
            "Volleys",
            "Dribbling",
            "Curve",
            "FK Accuracy",
            "Long passing",
            "Ball control",
            "Acceleration",
            "Sprint speed",
            "Agility",
            "Reactions",
            "Balance",
            "Shot power",
            "Jumping",
            "Stamina",
            "Strength",
            "Long shots",
            "Aggression",
            "Interceptions",
            "Positioning",
            "Vision",
            "Penalties",
            "Composure",
            "Marking",
            "Standing tackle",
            "Sliding tackle",
        ]

        # Tìm stats từ các cột
        cols = soup.find_all("div", class_="col")
        for col in cols:
            paragraphs = col.find_all("p")
            for p in paragraphs:
                # Tìm thẻ em chứa giá trị
                em = p.find("em")
                # Tìm span hoặc text chứa tên stat
                spans = p.find_all("span")
                for span in spans:
                    span_text = span.get_text().strip()
                    if span_text in stat_names and em:
                        stat_value = em.get("title") or em.get_text()
                        if stat_value.isdigit():
                            safe_name = span_text.replace(" ", "_").replace(".", "")
                            details[safe_name] = int(stat_value)

        # Lấy Positions từ các thẻ position
        positions = []
        pos_divs = soup.find_all("div", class_="pos")
        for pos_div in pos_divs[:10]:  # Lấy tối đa 10 positions đầu tiên
            pos_text = pos_div.get_text().strip()
            if pos_text and len(pos_text) <= 4 and pos_text not in positions:
                positions.append(pos_text)
        if positions:
            details["Positions"] = ", ".join(positions[:5])

        # Work Rate (Attack/Defense)
        work_rate_match = re.search(
            r"Work Rate.*?(\w+)\s*/\s*(\w+)", page_text, re.IGNORECASE
        )
        if work_rate_match:
            details["Work_Rate"] = (
                f"{work_rate_match.group(1)}/{work_rate_match.group(2)}"
            )

        return details

    except Exception as e:
        print(f"  Lỗi khi crawl chi tiết cầu thủ: {e}")
        import traceback

        traceback.print_exc()
        return {}


def load_existing_data(csv_path="sofifa_players.csv"):
    """Load dữ liệu đã crawl trước đó để tiếp tục"""
    try:
        if pd.io.common.file_exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"✓ Đã load {len(df)} cầu thủ từ file {csv_path}")
            return df
    except Exception as e:
        print(f"⚠ Không thể load file {csv_path}: {e}")
    return pd.DataFrame()


def get_existing_player_urls(df):
    """Lấy danh sách URL cầu thủ đã crawl để tránh duplicate"""
    if df.empty or "Player_URL" not in df.columns:
        return set()
    return set(df["Player_URL"].dropna().tolist())


def crawl_sofifa(pages=5, detailed=False, start_page=0, resume=False, csv_path="sofifa_players.csv", save_interval=10):
    """
    Crawl dữ liệu cầu thủ từ SoFIFA

    Args:
        pages: Số trang cần crawl (mỗi trang có 60 cầu thủ)
        detailed: Nếu True, sẽ crawl thông tin chi tiết từ trang cá nhân của mỗi cầu thủ (chậm hơn)
        start_page: Trang bắt đầu crawl (0-indexed)
        resume: Nếu True, sẽ load dữ liệu cũ và bỏ qua cầu thủ đã có
        csv_path: Đường dẫn file CSV để lưu/load
        save_interval: Số trang sau mỗi lần tự động lưu (để tránh mất dữ liệu)
    """
    all_players = []
    existing_urls = set()
    
    # Load dữ liệu cũ nếu resume
    if resume:
        existing_df = load_existing_data(csv_path)
        if not existing_df.empty:
            all_players = existing_df.to_dict("records")
            existing_urls = get_existing_player_urls(existing_df)
            print(f"📌 Đã có {len(existing_urls)} URL cầu thủ - sẽ bỏ qua nếu gặp lại")

    # Tạo scraper session
    if USE_CLOUDSCRAPER:
        try:
            scraper = cloudscraper.create_scraper(
                browser={"browser": "chrome", "platform": "windows", "desktop": True},
                delay=10,
            )
            print("✓ Đang sử dụng cloudscraper")
        except Exception as e:
            print(f"Lỗi cloudscraper: {e}, chuyển sang requests")
            scraper = requests.Session()
            scraper.headers.update(
                {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                    "Referer": "https://sofifa.com/",
                }
            )
    else:
        scraper = requests.Session()
        scraper.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Referer": "https://sofifa.com/",
            }
        )

    new_players_count = 0
    skipped_count = 0
    
    for page in range(start_page, start_page + pages):
        # Mỗi trang của SoFIFA hiển thị 60 cầu thủ, offset tăng dần 60
        offset = page * 60
        url = f"https://sofifa.com/players?offset={offset}"

        print(f"Đang crawl trang {page + 1} (offset={offset})...")

        try:
            response = scraper.get(url, timeout=15)

            if response.status_code != 200:
                print(
                    f"Không thể truy cập website - Status code: {response.status_code}"
                )
                print(f"URL: {url}")
                break
        except Exception as e:
            print(f"Lỗi kết nối: {e}")
            break

        soup = BeautifulSoup(response.content, "html.parser")
        # Tìm bảng - website đã thay đổi, không còn class 'table-hover'
        table = soup.find("table")

        if not table:
            print("Không tìm thấy bảng dữ liệu trên trang")
            print("Có thể website đã thay đổi cấu trúc hoặc yêu cầu xác thực")
            break

        tbody = table.find("tbody")
        if not tbody:
            print("Không tìm thấy tbody trong bảng")
            break

        rows = tbody.find_all("tr")

        if not rows:
            print("Không tìm thấy dữ liệu cầu thủ trên trang")
            break

        for row in rows:
            try:
                cols = row.find_all("td")

                if len(cols) < 8:
                    continue

                # Trích xuất dữ liệu - cấu trúc cột:
                # 0: Picture, 1: Name+Position, 2: Age, 3: Overall+Change, 4: Potential+Change
                # 5: Team, 6: Value, 7: Wage, 8+: Stats

                # Lấy tên cầu thủ và URL từ cột 1
                name_col = cols[1]
                name_link = name_col.find("a")
                if not name_link:
                    continue

                # Tên có thể nằm trong nhiều thẻ, lấy text và làm sạch
                name = (
                    name_link.get("title")
                    or name_link.text.strip().split("\n")[0].strip()
                )
                player_url = name_link.get("href", "")  # URL đến trang chi tiết cầu thủ

                # Bỏ qua nếu đã có trong dataset
                if resume and player_url in existing_urls:
                    skipped_count += 1
                    continue

                age = cols[2].text.strip()
                ovr_text = cols[3].text.strip()  # Có thể là "75+2", "75-2" hoặc "75"
                # Tách số overall, bỏ qua +/- thay đổi
                ovr = re.split(r"[+\-]", ovr_text)[0]

                pot_text = cols[4].text.strip()  # Có thể là "88+2", "88-2" hoặc "88"
                # Tách số potential, bỏ qua +/- thay đổi
                pot = re.split(r"[+\-]", pot_text)[0]

                team_link = cols[5].find("a")
                team = team_link.text.strip() if team_link else "Free Agent"

                value = cols[6].text.strip()
                wage = cols[7].text.strip()

                player_data = {
                    "Name": name,
                    "Age": int(age) if age.isdigit() else 0,
                    "Overall": int(ovr) if ovr.isdigit() else 0,
                    "Potential": int(pot) if pot.isdigit() else 0,
                    "Team": team,
                    "Value_Raw": value,
                    "Wage_Raw": wage,
                    "Value_Numeric": parse_value(value),
                    "Wage_Numeric": parse_value(wage),
                    "Player_URL": player_url,
                }

                # Nếu yêu cầu thông tin chi tiết, crawl trang cá nhân của cầu thủ
                if detailed and player_url:
                    print(f"  → Đang lấy thông tin chi tiết của {name}...")
                    detailed_info = crawl_player_details(scraper, player_url)
                    player_data.update(detailed_info)
                    time.sleep(1)  # Nghỉ thêm khi crawl chi tiết để tránh bị ban

                all_players.append(player_data)
                existing_urls.add(player_url)  # Thêm vào set để tránh duplicate trong cùng session
                new_players_count += 1
            except Exception as e:
                print(f"Lỗi khi xử lý một cầu thủ: {e}")
                continue

        # Auto-save sau mỗi save_interval trang
        pages_crawled = page - start_page + 1
        if save_interval > 0 and pages_crawled % save_interval == 0:
            temp_df = pd.DataFrame(all_players)
            temp_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"💾 Tự động lưu: {len(all_players)} cầu thủ (mới: {new_players_count}, bỏ qua: {skipped_count})")

        # Nghỉ một chút để tránh bị ban IP
        time.sleep(2)

    print(f"\n📊 Kết quả crawl: {new_players_count} cầu thủ mới, {skipped_count} bỏ qua (đã có)")
    return pd.DataFrame(all_players)


def test_connection():
    """Kiểm tra kết nối đến website"""
    if USE_CLOUDSCRAPER:
        try:
            scraper = cloudscraper.create_scraper(
                browser={"browser": "chrome", "platform": "windows", "desktop": True},
                delay=10,
            )
            print("→ Đang sử dụng cloudscraper để bypass Cloudflare")
        except Exception as e:
            print(f"Lỗi khởi tạo cloudscraper: {e}")
            scraper = requests.Session()
            scraper.headers.update(
                {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                }
            )
    else:
        scraper = requests.Session()
        scraper.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Referer": "https://sofifa.com/",
            }
        )
        print("→ Đang sử dụng requests với headers mô phỏng trình duyệt")

    url = "https://sofifa.com/players"
    print(f"Đang kiểm tra kết nối đến: {url}")

    try:
        response = scraper.get(url, timeout=15)
        print(f"Status code: {response.status_code}")
        print(f"Content length: {len(response.content)} bytes")

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")

            # Debug: In ra tất cả các class của bảng để kiểm tra
            tables = soup.find_all("table")
            print(f"Tìm thấy {len(tables)} bảng")
            for i, tbl in enumerate(tables):
                print(f"  Bảng {i+1}: class={tbl.get('class')}")

            table = soup.find("table", {"class": "table-hover"})
            if not table:
                # Thử tìm bảng với class khác
                table = soup.find("table")

            print(f"Tìm thấy bảng: {table is not None}")
            if table:
                tbody = table.find("tbody")
                rows = tbody.find_all("tr") if tbody else []
                print(f"Số dòng dữ liệu: {len(rows)}")

                # Debug: In ra cấu trúc của dòng đầu tiên
                if rows:
                    first_row = rows[0]
                    cols = first_row.find_all(["td", "th"])
                    print(f"Số cột: {len(cols)}")
                    for idx, col in enumerate(cols[:5]):  # In 5 cột đầu
                        print(f"  Cột {idx}: {col.text.strip()[:50]}")
        return response
    except Exception as e:
        print(f"Lỗi: {e}")
        return None


# Kiểm tra kết nối trước
print("=== KIỂM TRA KẾT NỐI ===")
test_response = test_connection()
print()

if test_response and test_response.status_code == 200:
    print("=== BẮT ĐẦU CRAWL DỮ LIỆU ===")
    print()

    # CẤU HÌNH CRAWL - Thay đổi các giá trị này theo nhu cầu
    # ============================================================
    RESUME_MODE = True      # True = tiếp tục crawl từ dữ liệu cũ, False = bắt đầu từ đầu
    START_PAGE = 141         # Trang bắt đầu (0-indexed). 5442 records / 60 = ~91 trang đã crawl
    NUM_PAGES = 50          # Số trang cần crawl thêm (mỗi trang 60 cầu thủ)
    DETAILED_MODE = True    # True = lấy thông tin chi tiết (chậm hơn), False = chỉ lấy thông tin cơ bản
    SAVE_INTERVAL = 5       # Tự động lưu sau mỗi bao nhiêu trang (0 = không auto-save)
    CSV_PATH = "sofifa_players.csv"
    # ============================================================

    print(f"📊 Cấu hình:")
    print(f"   - Resume mode: {'BẬT' if RESUME_MODE else 'TẮT'}")
    print(f"   - Trang bắt đầu: {START_PAGE + 1} (offset={START_PAGE * 60})")
    print(f"   - Số trang crawl: {NUM_PAGES}")
    print(f"   - Mode: {'CHI TIẾT' if DETAILED_MODE else 'CƠ BẢN'}")
    print(f"   - Auto-save: mỗi {SAVE_INTERVAL} trang")
    print(f"   - Ước tính thêm: ~{NUM_PAGES * 60} cầu thủ mới")
    if DETAILED_MODE:
        print("⚠ Chế độ chi tiết sẽ mất nhiều thời gian hơn")
    print()

    df = crawl_sofifa(
        pages=NUM_PAGES, 
        detailed=DETAILED_MODE, 
        start_page=START_PAGE,
        resume=RESUME_MODE,
        csv_path=CSV_PATH,
        save_interval=SAVE_INTERVAL
    )
else:
    print("Không thể kết nối đến website. Tạo DataFrame rỗng.")
    df = pd.DataFrame()

# Lưu dữ liệu ra CSV
if not df.empty:
    df.to_csv("sofifa_players.csv", index=False, encoding="utf-8-sig")
    print(f"\n✓ Đã lưu {len(df)} cầu thủ vào file sofifa_players.csv")
    print(f"\n📋 Các cột dữ liệu: {', '.join(df.columns.tolist())}")
    print(f"\n👀 Preview 5 cầu thủ cuối cùng:")
    print(df.tail())
else:
    print("\n❌ Không có dữ liệu để lưu")
