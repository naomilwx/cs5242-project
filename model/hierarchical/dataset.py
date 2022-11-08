from model.dataset import DataSet as BaseDataset, all_categories

category_groups = {
    'cars': ['Automotive'],
    'bags': ["Men's-Bags", "Women's-Bags"],
    'apparel-footwear': ["Women's-Apparel", "Men's-Wear", 'Kids-Fashion', "Men's-Shoes", "Women's-Shoes"],
    'food': ['Pet-Food-Supplies', 'Food-Beverages'],
    'accessories': ['Jewellery-Accessories', 'Watches'],
    'gadgets': ['Mobile-Gadgets', 'Computers-Peripherals', 'Cameras-Drones'],
    'health': ['Beauty-Personal-Care', 'Health-Wellness'],
    'hobbies': ['Video-Games', 'Travel-Luggage', 'Sports-Outdoors', 'Hobbies-Books'],
    'home-n-kids': ['Home-Living', 'Home-Appliances', 'Toys-Kids-Babies']
}

def category_to_group_map():
    m = {}
    for (g, items) in category_groups.items():
        for item in items:
            m[item] = g
    return m

    
class DataSet(BaseDataset):
    def __init__(self, path = None, max_num_img=None, crop=0.75):
        super(DataSet, self).__init__(path, max_num_img, crop, categories=all_categories)
        self.groups = list(category_groups.keys())
        self.catgroup_map = category_to_group_map()
        
    def group_id_from_category_id(self, cat_id):
        group = self.catgroup_map[self.categories[cat_id]]
        return self.groups.index(group)

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        g_id = self.group_id_from_category_id(label)

        return image, g_id, label
