"""
Crawl cải tiến với nhiều kỹ thuật tránh bị chặn
"""
import sys
import random
import time
from fake_useragent import UserAgent

try:
    import cloudscraper
    USE_CLOUDSCRAPER = True
    print("✓ Cloudscraper đã được cài đặt")
except ImportError:
    import requests
    USE_CLOUDSCRAPER = False
    print("⚠ Cloudscraper chưa cài đặt")

from bs4 import BeautifulSoup
import pandas as pd
import re

# Danh sách User-Agent để rotate
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
]

def parse_value(value_str):
    """Chuyển đổi chuỗi €110.5M hoặc €500K thành số thực"""
    if "M" in value_str:
        return float(re.sub(r"[€M]", "", value_str)) * 1000000
    elif "K" in value_str:
        return float(re.sub(r"[€K]", "", value_str)) * 1000
    return 0


def create_scraper_with_retry():
    """Tạo scraper giống hệt crawl.py"""
    if USE_CLOUDSCRAPER:
        try:
            scraper = cloudscraper.create_scraper(
                browser={"browser": "chrome", "platform": "windows", "desktop": True},
                delay=10,
            )
            print("✓ Đang sử dụng cloudscraper")
            return scraper
        except Exception as e:
            print(f"Lỗi cloudscraper: {e}, chuyển sang requests")
            scraper = requests.Session()
            scraper.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Referer": "https://sofifa.com/",
            })
    else:
        scraper = requests.Session()
        scraper.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": "https://sofifa.com/",
        })
    return scraper


def get_with_retry(scraper, url, max_retries=5):
    """GET request với retry và exponential backoff"""
    for attempt in range(max_retries):
        try:
            # Rotate User-Agent mỗi lần retry
            if hasattr(scraper, 'headers'):
                scraper.headers['User-Agent'] = random.choice(USER_AGENTS)
            
            response = scraper.get(url, timeout=20)
            
            if response.status_code == 200:
                return response
            elif response.status_code == 403:
                wait_time = (2 ** attempt) * 10  # Exponential backoff: 10s, 20s, 40s, 80s, 160s
                print(f"  ⚠ 403 Forbidden - Đợi {wait_time}s trước khi thử lại (lần {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            elif response.status_code == 429:
                wait_time = 60
                print(f"  ⚠ 429 Too Many Requests - Đợi {wait_time}s")
                time.sleep(wait_time)
            else:
                print(f"  ⚠ Status {response.status_code} - Thử lại sau 10s")
                time.sleep(10)
                
        except requests.exceptions.Timeout:
            print(f"  ⚠ Timeout - Thử lại (lần {attempt + 1}/{max_retries})")
            time.sleep(5)
        except Exception as e:
            print(f"  ⚠ Lỗi: {e}")
            time.sleep(10)
    
    print(f"  ❌ Không thể truy cập sau {max_retries} lần thử")
    return None


def crawl_player_details(scraper, player_url):
    """Crawl thông tin chi tiết của một cầu thủ"""
    try:
        full_url = f"https://sofifa.com{player_url}"
        response = get_with_retry(scraper, full_url, max_retries=3)
        
        if not response or response.status_code != 200:
            return {}

        soup = BeautifulSoup(response.content, "html.parser")
        details = {}

        # Lấy dữ liệu từ JSON-LD
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

        # Lấy Preferred Foot
        page_text = soup.get_text()
        preferred_foot_match = re.search(r"Preferred foot\s+(Left|Right)", page_text)
        if preferred_foot_match:
            details["Preferred_Foot"] = preferred_foot_match.group(1)

        # Skill Moves và Weak Foot
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

        # Lấy stats
        stat_names = [
            "Crossing", "Finishing", "Heading accuracy", "Short passing", "Volleys",
            "Dribbling", "Curve", "FK Accuracy", "Long passing", "Ball control",
            "Acceleration", "Sprint speed", "Agility", "Reactions", "Balance",
            "Shot power", "Jumping", "Stamina", "Strength", "Long shots",
            "Aggression", "Interceptions", "Positioning", "Vision", "Penalties",
            "Composure", "Marking", "Standing tackle", "Sliding tackle",
            "GK Diving", "GK Handling", "GK Kicking", "GK Positioning", "GK Reflexes",
        ]

        cols = soup.find_all("div", class_="col")
        for col in cols:
            paragraphs = col.find_all("p")
            for p in paragraphs:
                em = p.find("em")
                spans = p.find_all("span")
                for span in spans:
                    span_text = span.get_text().strip()
                    if span_text in stat_names and em:
                        stat_value = em.get("title") or em.get_text()
                        if stat_value.isdigit():
                            safe_name = span_text.replace(" ", "_").replace(".", "")
                            details[safe_name] = int(stat_value)

        # Lấy Positions
        positions = []
        pos_spans = soup.find_all("span", class_="pos")
        for pos_span in pos_spans[:10]:
            pos_text = pos_span.get_text().strip()
            if pos_text and len(pos_text) <= 4 and pos_text not in positions:
                positions.append(pos_text)
        
        if not positions:
            pos_divs = soup.find_all("div", class_="pos")
            for pos_div in pos_divs[:10]:
                pos_text = pos_div.get_text().strip()
                if pos_text and len(pos_text) <= 4 and pos_text not in positions:
                    positions.append(pos_text)
        
        if positions:
            details["Positions"] = ", ".join(positions[:5])
        else:
            details["Positions"] = "Unknown"

        return details

    except Exception as e:
        print(f"  Lỗi crawl chi tiết: {e}")
        return {}


def crawl_sofifa_improved(pages=5, start_page=0, csv_path="sofifa_players.csv"):
    """Crawl với kỹ thuật cải tiến"""
    print("🚀 Bắt đầu crawl với kỹ thuật cải tiến\n")
    
    # Load dữ liệu cũ
    all_players = []
    existing_urls = set()
    
    try:
        if pd.io.common.file_exists(csv_path):
            df = pd.read_csv(csv_path)
            all_players = df.to_dict("records")
            existing_urls = set(df["Player_URL"].dropna().tolist())
            print(f"✓ Đã load {len(existing_urls)} cầu thủ từ file cũ\n")
    except:
        pass
    
    # Tạo scraper
    scraper = create_scraper_with_retry()
    
    new_count = 0
    skip_count = 0
    
    for page in range(start_page, start_page + pages):
        offset = page * 60
        url = f"https://sofifa.com/players?offset={offset}"
        
        print(f"📄 Trang {page + 1} (offset={offset})")
        
        # Random delay giữa các trang (8-12 giây - chậm hơn)
        if page > start_page:
            delay = random.uniform(8, 12)
            print(f"   ⏱ Đợi {delay:.1f}s...")
            time.sleep(delay)
        
        response = get_with_retry(scraper, url)
        
        if not response:
            print(f"❌ Không thể crawl trang {page + 1}, dừng lại\n")
            break
        
        print(f"   ✓ Status: {response.status_code}, Size: {len(response.content)} bytes")
        
        # Parse giống crawl.py gốc
        soup = BeautifulSoup(response.content, "html.parser")
        table = soup.find("table")
        
        # Debug: nếu không tìm thấy bảng
        if not table:
            print(f"   ⚠ Không tìm thấy bảng")
            print(f"   Tìm thấy {len(soup.find_all('table'))} bảng")
        
        if not table:
            print("❌ Không tìm thấy bảng dữ liệu\n")
            break
        
        tbody = table.find("tbody")
        rows = tbody.find_all("tr") if tbody else []
        
        print(f"   Tìm thấy {len(rows)} cầu thủ")
        
        for row in rows:
            try:
                cols = row.find_all("td")
                if len(cols) < 8:
                    continue
                
                name_col = cols[1]
                name_link = name_col.find("a")
                if not name_link:
                    continue
                
                name = name_link.get("title") or name_link.text.strip().split("\n")[0].strip()
                player_url = name_link.get("href", "")
                
                if player_url in existing_urls:
                    skip_count += 1
                    continue
                
                age = cols[2].text.strip()
                ovr = re.split(r"[+\-]", cols[3].text.strip())[0]
                pot = re.split(r"[+\-]", cols[4].text.strip())[0]
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
                
                # Crawl chi tiết
                print(f"   → {name}...", end=" ")
                detailed_info = crawl_player_details(scraper, player_url)
                player_data.update(detailed_info)
                print("✓")
                
                all_players.append(player_data)
                existing_urls.add(player_url)
                new_count += 1
                
                # Random delay giữa các cầu thủ (5-8 giây - chậm hơn)
                time.sleep(random.uniform(5, 8))
                
            except Exception as e:
                print(f"   ⚠ Lỗi: {e}")
                continue
        
        # Auto-save mỗi trang
        if all_players:
            temp_df = pd.DataFrame(all_players)
            temp_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"   💾 Đã lưu: {len(all_players)} tổng ({new_count} mới, {skip_count} bỏ qua)\n")
    
    return pd.DataFrame(all_players)


# Chạy
if __name__ == "__main__":
    print("=" * 60)
    print("CRAWL CẢI TIẾN VỚI KỸ THUẬT ANTI-BAN")
    print("=" * 60)
    print()
    
    # Cấu hình
    START_PAGE = 119  # Trang bắt đầu
    NUM_PAGES = 50  # Số trang (bắt đầu với ít trang để test)
    
    print(f"📊 Cấu hình:")
    print(f"   - Trang bắt đầu: {START_PAGE + 1}")
    print(f"   - Số trang: {NUM_PAGES}")
    print(f"   - Retry logic: BẬT")
    print(f"   - User-Agent rotation: BẬT")
    print(f"   - Random delays: BẬT")
    print(f"   - Exponential backoff: BẬT")
    print()
    
    df = crawl_sofifa_improved(
        pages=NUM_PAGES,
        start_page=START_PAGE,
        csv_path="sofifa_players.csv"
    )
    
    if not df.empty:
        print(f"\n✅ Hoàn tất! Đã lưu {len(df)} cầu thủ")
    else:
        print("\n❌ Không có dữ liệu")
