import cloudscraper
from bs4 import BeautifulSoup

# Test với Messi
url = "https://sofifa.com/player/158023/lionel-messi/260015/"
print(f"Testing: {url}\n")

scraper = cloudscraper.create_scraper()
response = scraper.get(url, timeout=20)

print(f"Status: {response.status_code}")

if response.status_code == 200:
    soup = BeautifulSoup(response.content, "html.parser")

    # Test 1: Tìm div.pos
    pos_divs = soup.find_all("div", class_="pos")
    print(f"\n1. Found {len(pos_divs)} div.pos elements")
    if pos_divs:
        for i, p in enumerate(pos_divs[:5]):
            print(f"   {i+1}. {p.get_text().strip()}")

    # Test 2: Tìm tất cả div có chứa "pos" trong class
    all_pos = soup.find_all("div", class_=lambda x: x and "pos" in x)
    print(f"\n2. Found {len(all_pos)} divs with 'pos' in class")
    for i, d in enumerate(all_pos[:5]):
        print(f"   {i+1}. class={d.get('class')}, text={d.get_text().strip()[:30]}")

    # Test 3: Tìm positions section
    print(f"\n3. Searching for position keywords...")
    text_content = soup.get_text()
    if "Position" in text_content:
        print("   ✓ Found 'Position' in text")

    # Test 4: Tìm bất kỳ text nào giống position (ST, CAM, etc)
    import re

    positions_found = re.findall(
        r"\b(ST|CF|LW|RW|CAM|CM|CDM|CB|LB|RB|GK|LM|RM)\b", text_content
    )
    print(f"\n4. Position codes found: {set(positions_found)}")

    # Test 5: Xem cấu trúc HTML gần chỗ có position
    print(f"\n5. HTML structure near positions:")
    # Tìm tất cả thẻ span
    for span in soup.find_all("span")[:20]:
        text = span.get_text().strip()
        if text in ["ST", "CF", "LW", "RW", "CAM", "CM"]:
            print(f"   Found '{text}' in span")
            print(f"   Parent: {span.parent.name}, class={span.parent.get('class')}")
            print(f"   Full element: {span.parent}")
            break
else:
    print(f"Failed to load page: {response.status_code}")
