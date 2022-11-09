from shopee_crawler.toolkit import crawl_by_cat_url
import time
import os
import json
import requests

category_urls = ["https://shopee.sg/Women's-Apparel-cat.11012819", "https://shopee.sg/Men's-Wear-cat.11012963", 'https://shopee.sg/Mobile-Gadgets-cat.11013350', 'https://shopee.sg/Home-Living-cat.11000001', 'https://shopee.sg/Computers-Peripherals-cat.11013247', 'https://shopee.sg/Beauty-Personal-Care-cat.11012301', 'https://shopee.sg/Home-Appliances-cat.11027421', 'https://shopee.sg/Health-Wellness-cat.11027491', 'https://shopee.sg/Food-Beverages-cat.11011871', 'https://shopee.sg/Toys-Kids-Babies-cat.11011538', 'https://shopee.sg/Kids-Fashion-cat.11012218', 'https://shopee.sg/Video-Games-cat.11013478', 'https://shopee.sg/Sports-Outdoors-cat.11012018', 'https://shopee.sg/Hobbies-Books-cat.11011760', 'https://shopee.sg/Cameras-Drones-cat.11013548', 'https://shopee.sg/Pet-Food-Supplies-cat.11012453', "https://shopee.sg/Women's-Bags-cat.11012592", "https://shopee.sg/Men's-Bags-cat.11012659", 'https://shopee.sg/Jewellery-Accessories-cat.11013077', 'https://shopee.sg/Watches-cat.11012515', "https://shopee.sg/Women's-Shoes-cat.11012698", "https://shopee.sg/Men's-Shoes-cat.11012767", 'https://shopee.sg/Automotive-cat.11000002', 'https://shopee.sg/ShopeePay-Near-Me-cat.11080712', 'https://shopee.sg/Dining-Travel-Services-cat.11012255', 'https://shopee.sg/Travel-Luggage-cat.11012566', 'https://shopee.sg/Miscellaneous-cat.11029718']

def get_category_data(dir, category_url):
    data = crawl_by_cat_url('shopee.sg', cat_url=category_url)
    cat = category_url.split('/')[-1].split('.')[0]
    with open(os.path.join(dir, f"{cat}.json"), 'w') as f:
        json.dump(data, f)

def save_image_from_url(url, file_name):
    headers = {
        "User-Agent": "Chrome/51.0.2704.103",
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        with open(file_name, "wb") as f:
            f.write(response.content)
    else:
        print(response.status_code)

def download_images(dir):
    for fn in os.listdir(dir):
        if fn[0] == '.':
            continue
        fp = os.path.join(dir, fn)
        if not os.path.isfile(fp):
            continue
        image_path = os.path.join(dir, 'images', fn.split('.')[0])
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        print('downloading images for', fn)
        with open(fp, 'r') as f:
            data = json.load(f)
            for p in data:
                save_path = os.path.join(image_path, f"{p['product_id']}.png")
                if os.path.isfile(save_path):
                    continue
                save_image_from_url(p["product_image"], save_path)
            time.sleep(1)

def product_category_and_names(dir):
    categories = {}
    product_names = {}
    for fn in os.listdir(dir):
        if fn[0] == '.':
            continue
        fp = os.path.join(dir, fn)
        if not os.path.isfile(fp):
            continue
        
        category = fn.split('-cat')[0]
        with open(fp, 'r') as f:
            categories[category] = []
            data = json.load(f)
            for p in data:
                pid = p['product_id']
                categories[category].append(pid)
                product_names[str(pid)] = p['product_name']

    return categories, product_names

# if __name__ == '__main__':
#     for c in category_urls:
#         get_category_data('../data', c)
#     download_images('../data')