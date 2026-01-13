try:
    import cloudscraper
    USE_CLOUDSCRAPER = True
except ImportError:
    import requests
    USE_CLOUDSCRAPER = False
    print("Cài đặt cloudscraper để vượt qua Cloudflare protection: pip install cloudscraper")

from bs4 import BeautifulSoup
import pandas as pd
import time
import re

def parse_value(value_str):
    """Chuyển đổi chuỗi €110.5M hoặc €500K thành số thực"""
    if 'M' in value_str:
        return float(re.sub(r'[€M]', '', value_str)) * 1000000
    elif 'K' in value_str:
        return float(re.sub(r'[€K]', '', value_str)) * 1000
    return 0

def crawl_sofifa(pages=5):
    all_players = []
    
    # Tạo scraper session
    if USE_CLOUDSCRAPER:
        scraper = cloudscraper.create_scraper(
            browser={
                'browser': 'chrome',
                'platform': 'windows',
                'desktop': True
            }
        )
    else:
        scraper = requests.Session()
        scraper.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })

    for page in range(0, pages):
        # Mỗi trang của SoFIFA hiển thị 60 cầu thủ, offset tăng dần 60
        offset = page * 60
        url = f"https://sofifa.com/players?offset={offset}"
        
        print(f"Đang crawl trang {page + 1}...")
        
        try:
            response = scraper.get(url, timeout=15)
            
            if response.status_code != 200:
                print(f"Không thể truy cập website - Status code: {response.status_code}")
                print(f"URL: {url}")
                break
        except Exception as e:
            print(f"Lỗi kết nối: {e}")
            break
            
        soup = BeautifulSoup(response.content, 'html.parser')
        # Tìm bảng - website đã thay đổi, không còn class 'table-hover'
        table = soup.find('table')
        
        if not table:
            print("Không tìm thấy bảng dữ liệu trên trang")
            print("Có thể website đã thay đổi cấu trúc hoặc yêu cầu xác thực")
            break
            
        tbody = table.find('tbody')
        if not tbody:
            print("Không tìm thấy tbody trong bảng")
            break
            
        rows = tbody.find_all('tr')
        
        if not rows:
            print("Không tìm thấy dữ liệu cầu thủ trên trang")
            break

        for row in rows:
            try:
                cols = row.find_all('td')
                
                if len(cols) < 8:
                    continue
                
                # Trích xuất dữ liệu - cấu trúc cột:
                # 0: Picture, 1: Name+Position, 2: Age, 3: Overall+Change, 4: Potential+Change
                # 5: Team, 6: Value, 7: Wage, 8+: Stats
                
                # Lấy tên cầu thủ từ cột 1
                name_col = cols[1]
                name_link = name_col.find('a')
                if not name_link:
                    continue
                
                # Tên có thể nằm trong nhiều thẻ, lấy text và làm sạch
                name = name_link.get('title') or name_link.text.strip().split('\n')[0].strip()
                
                age = cols[2].text.strip()
                ovr_text = cols[3].text.strip()  # Có thể là "75+2" hoặc "75"
                ovr = ovr_text.split('+')[0] if '+' in ovr_text else ovr_text
                
                pot_text = cols[4].text.strip()  # Có thể là "88+2" hoặc "88"
                pot = pot_text.split('+')[0] if '+' in pot_text else pot_text
                
                team_link = cols[5].find('a')
                team = team_link.text.strip() if team_link else "Free Agent"
                
                value = cols[6].text.strip()
                wage = cols[7].text.strip()
                
                player_data = {
                    'Name': name,
                    'Age': int(age) if age.isdigit() else 0,
                    'Overall': int(ovr) if ovr.isdigit() else 0,
                    'Potential': int(pot) if pot.isdigit() else 0,
                    'Team': team,
                    'Value_Raw': value,
                    'Wage_Raw': wage,
                    'Value_Numeric': parse_value(value),
                    'Wage_Numeric': parse_value(wage)
                }
                all_players.append(player_data)
            except Exception as e:
                print(f"Lỗi khi xử lý một cầu thủ: {e}")
                continue
        
        # Nghỉ một chút để tránh bị ban IP
        time.sleep(2)

    return pd.DataFrame(all_players)

def test_connection():
    """Kiểm tra kết nối đến website"""
    if USE_CLOUDSCRAPER:
        scraper = cloudscraper.create_scraper(
            browser={
                'browser': 'chrome',
                'platform': 'windows',
                'desktop': True
            }
        )
        print("Sử dụng cloudscraper để bypass Cloudflare")
    else:
        scraper = requests.Session()
        scraper.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })
        print("Sử dụng requests thông thường (có thể bị chặn)")
    
    url = "https://sofifa.com/players"
    print(f"Đang kiểm tra kết nối đến: {url}")
    
    try:
        response = scraper.get(url, timeout=15)
        print(f"Status code: {response.status_code}")
        print(f"Content length: {len(response.content)} bytes")
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Debug: In ra tất cả các class của bảng để kiểm tra
            tables = soup.find_all('table')
            print(f"Tìm thấy {len(tables)} bảng")
            for i, tbl in enumerate(tables):
                print(f"  Bảng {i+1}: class={tbl.get('class')}")
            
            table = soup.find('table', {'class': 'table-hover'})
            if not table:
                # Thử tìm bảng với class khác
                table = soup.find('table')
            
            print(f"Tìm thấy bảng: {table is not None}")
            if table:
                tbody = table.find('tbody')
                rows = tbody.find_all('tr') if tbody else []
                print(f"Số dòng dữ liệu: {len(rows)}")
                
                # Debug: In ra cấu trúc của dòng đầu tiên
                if rows:
                    first_row = rows[0]
                    cols = first_row.find_all(['td', 'th'])
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
    # Chạy thử nghiệm crawl 2 trang đầu tiên
    print("=== BẮT ĐẦU CRAWL DỮ LIỆU ===")
    df = crawl_sofifa(pages=2)
else:
    print("Không thể kết nối đến website. Tạo DataFrame rỗng.")
    df = pd.DataFrame()

# Lưu dữ liệu ra CSV
df.to_csv('sofifa_players.csv', index=False, encoding='utf-8-sig')
print("Đã lưu dữ liệu vào file sofifa_players.csv")
print(df.head())